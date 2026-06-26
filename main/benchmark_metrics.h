#ifndef BENCHMARK_METRICS_H
#define BENCHMARK_METRICS_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>   // isinf, isnan

// ---------------------------------------------------------------------------
// Timing on this target
// ---------------------------------------------------------------------------
// esp-tflite-micro links a micro_time.cc whose ticks_per_second() == 1000000,
// confirmed at runtime. That means 1 profiler tick == 1 microsecond. There is
// NO relationship to the 240 MHz CPU clock; do not divide ticks by 240.
//
// We still read ticks_per_second() at runtime (see ticks_to_us below) so the
// pipeline stays correct even if the timer source ever changes.
// ---------------------------------------------------------------------------

// A sentinel meaning "this configuration was not run" (e.g. SRAM run skipped
// because the model did not fit). Kept distinct from a legitimate 0.
static const int32_t BM_NOT_RUN_I = -1;
static const float   BM_NOT_RUN_F = -1.0f;

// SQNR uses INFINITY (perfect reconstruction) and NAN (no signal) as in-band
// values, so it CANNOT share the -1 not-run sentinel. We use NAN for not-run
// too and distinguish "not run" from "no signal" via the run flags / skip
// reason already on the row. Both render as an empty CSV cell.
static const float   BM_SQNR_NOT_RUN = NAN;

// Convert profiler ticks to microseconds using the live tick rate.
static inline float bm_ticks_to_us(int32_t ticks, uint32_t ticks_per_second) {
    if (ticks < 0) return BM_NOT_RUN_F;             // not-run sentinel passthrough
    if (ticks_per_second == 0) return BM_NOT_RUN_F; // timer not implemented
    return (float)ticks * 1000000.0f / (float)ticks_per_second;
}

// One CSV row per (model_config, quant_type).
typedef struct {
    int         model_config_index;   // 0..(BENCHMARK_MODEL_COUNT/2 - 1); == original sorted index
    const char* quant_type;           // "float32" or "int8"

    uint32_t    model_size_bytes;     // from BENCHMARK_MODEL_SIZES
    size_t      arena_used_bytes;     // measured via interpreter->arena_used_bytes()

    // --- PSRAM run ---
    int32_t     psram_inf_ticks;      // profiler GetTotalTicks (== microseconds here)
    float       psram_inf_us;         // ticks converted via live tick rate
    float       psram_mse;
    float       psram_sqnr_db;        // scale-free quantization quality, dB

    // --- SRAM run ---
    int         sram_eligible;        // 1 if our measure-then-decide said "try it"
    int         sram_ran;             // 1 if it actually allocated + ran
    int32_t     sram_inf_ticks;
    float       sram_inf_us;
    float       sram_mse;
    float       sram_sqnr_db;

    // Provenance / diagnostics
    uint32_t    ticks_per_second;     // captured once; documents the time base
    const char* sram_skip_reason;     // why SRAM was skipped, or "" if it ran
} BenchmarkMetrics;

static inline void bm_init_row(BenchmarkMetrics* r) {
    memset(r, 0, sizeof(*r));
    r->arena_used_bytes  = 0;
    r->psram_inf_ticks   = BM_NOT_RUN_I;
    r->psram_inf_us      = BM_NOT_RUN_F;
    r->psram_mse         = BM_NOT_RUN_F;
    r->psram_sqnr_db     = BM_SQNR_NOT_RUN;
    r->sram_eligible     = 0;
    r->sram_ran          = 0;
    r->sram_inf_ticks    = BM_NOT_RUN_I;
    r->sram_inf_us       = BM_NOT_RUN_F;
    r->sram_mse          = BM_NOT_RUN_F;
    r->sram_sqnr_db      = BM_SQNR_NOT_RUN;
    r->sram_skip_reason  = "";
}

// Emit the CSV header. Call once at the top of the run.
static inline int bm_csv_header(char* buf, int cap) {
    return snprintf(buf, cap,
        "model_index,quant_type,"
        "model_size_bytes,arena_used_bytes,"
        "psram_inf_ticks,psram_inf_us,psram_mse,psram_sqnr_db,"
        "sram_eligible,sram_ran,sram_inf_ticks,sram_inf_us,sram_mse,sram_sqnr_db,"
        "ticks_per_second,sram_skip_reason\n");
}

// Format an SQNR value. NaN (not-run / no-signal) -> empty cell. +Inf (perfect
// reconstruction, e.g. float32) -> "inf" so pandas reads it as np.inf, which
// you can filter or treat as a ceiling. Otherwise the dB value.
static inline void bm_fmt_sqnr(char* dst, int cap, float v) {
    if (isnan(v))       dst[0] = '\0';
    else if (isinf(v))  snprintf(dst, cap, "inf");
    else                snprintf(dst, cap, "%.3f", v);
}

// Append one row. A field that was not run is written empty so pandas reads it
// as NaN (read_csv default), which is exactly what you want for "didn't run".
static inline int bm_csv_row(char* buf, int cap, const BenchmarkMetrics* r) {
    char psram_ticks[24], psram_us[24], psram_mse[24], psram_sqnr[24];
    char sram_ticks[24],  sram_us[24],  sram_mse[24],  sram_sqnr[24];
    char arena[24];

    if (r->psram_inf_ticks < 0) psram_ticks[0] = '\0';
    else snprintf(psram_ticks, sizeof psram_ticks, "%ld", (long)r->psram_inf_ticks);
    if (r->psram_inf_us < 0)    psram_us[0] = '\0';
    else snprintf(psram_us, sizeof psram_us, "%.3f", r->psram_inf_us);
    if (r->psram_mse < 0)       psram_mse[0] = '\0';
    else snprintf(psram_mse, sizeof psram_mse, "%.6f", r->psram_mse);
    bm_fmt_sqnr(psram_sqnr, sizeof psram_sqnr, r->psram_sqnr_db);

    if (r->sram_inf_ticks < 0)  sram_ticks[0] = '\0';
    else snprintf(sram_ticks, sizeof sram_ticks, "%ld", (long)r->sram_inf_ticks);
    if (r->sram_inf_us < 0)     sram_us[0] = '\0';
    else snprintf(sram_us, sizeof sram_us, "%.3f", r->sram_inf_us);
    if (r->sram_mse < 0)        sram_mse[0] = '\0';
    else snprintf(sram_mse, sizeof sram_mse, "%.6f", r->sram_mse);
    bm_fmt_sqnr(sram_sqnr, sizeof sram_sqnr, r->sram_sqnr_db);

    snprintf(arena, sizeof arena, "%zu", r->arena_used_bytes);

    return snprintf(buf, cap,
        "%d,%s,"
        "%lu,%s,"
        "%s,%s,%s,%s,"
        "%d,%d,%s,%s,%s,%s,"
        "%lu,%s\n",
        r->model_config_index,
        r->quant_type ? r->quant_type : "",
        (unsigned long)r->model_size_bytes, arena,
        psram_ticks, psram_us, psram_mse, psram_sqnr,
        r->sram_eligible, r->sram_ran, sram_ticks, sram_us, sram_mse, sram_sqnr,
        (unsigned long)r->ticks_per_second,
        r->sram_skip_reason ? r->sram_skip_reason : "");
}

#endif // BENCHMARK_METRICS_H