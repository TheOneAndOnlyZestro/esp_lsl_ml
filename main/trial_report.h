// ============================================================
// trial_report.h  -- CSV reporting for the benchmark pipeline.
//
// One row per trial. Per-window quantities (stage ticks, ELU us, whole-
// window us, mse) are reduced to a plain (unweighted) average and a max
// across all windows in the trial. LSTM/regressor per-step ticks are
// reduced to avg/max across every step taken in the trial.
//
// Stage inference times are in PROFILER TICKS (getTotalTicks()).
// ELU times, window time and trial total are in microseconds (esp_timer).
// ============================================================
#ifndef TRIAL_REPORT_H
#define TRIAL_REPORT_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ---- a tiny running avg+max accumulator ----
struct AvgMax {
    double   sum   = 0.0;
    double   maxv  = 0.0;
    uint32_t count = 0;
    bool     seen  = false;

    inline void add(double v) {
        sum += v;
        if (!seen || v > maxv) { maxv = v; seen = true; }
        count++;
    }
    inline double avg() const { return count ? sum / (double)count : 0.0; }
    inline double max() const { return seen ? maxv : 0.0; }
    inline void reset() { sum = 0.0; maxv = 0.0; count = 0; seen = false; }
};

// ---- per-window stage measurements handed to the accumulator ----
struct WindowStageTimes {
    int32_t infa_ticks   = 0;   // model[0] getTotalTicks()
    int32_t infb_ticks   = 0;   // model[1] getTotalTicks()
    uint64_t elua_us      = 0;  // first ELU (esp_timer)
    uint64_t elub_us      = 0;  // second ELU (esp_timer)
    uint64_t window_us    = 0;  // whole-window wall clock (infa+elua+infb+elub+loop)
    double   mse          = 0.0;
    // LSTM/regressor steps are added individually via TrialStats::add_step()
};

// ---- per-trial accumulator ----
struct TrialStats {
    // identity / config (set once per trial)
    int    config       = 0;
    long   window_len   = 0;
    int    step         = 0;
    const char* quant_a    = "f32";
    const char* quant_b    = "f32";
    const char* quant_lstm = "f32";
    const char* quant_reg  = "f32";
    double size_a_kb    = 0.0;
    double size_b_kb    = 0.0;
    double size_lstm_kb = 0.0;
    double size_reg_kb  = 0.0;

    // per-window reductions
    AvgMax infa, infb;          // stage ticks
    AvgMax elua, elub;          // us
    AvgMax window_us;           // us
    AvgMax mse;

    // per-step reductions (over all steps of all windows in the trial)
    AvgMax lstm_step;           // ticks
    AvgMax reg_step;            // ticks

    // trial-level scalar
    uint64_t trial_total_us = 0;

    void reset_accumulators() {
        infa.reset(); infb.reset();
        elua.reset(); elub.reset();
        window_us.reset(); mse.reset();
        lstm_step.reset(); reg_step.reset();
        trial_total_us = 0;
    }

    inline void add_window(const WindowStageTimes& w) {
        infa.add((double)w.infa_ticks);
        infb.add((double)w.infb_ticks);
        elua.add((double)w.elua_us);
        elub.add((double)w.elub_us);
        window_us.add((double)w.window_us);
        mse.add(w.mse);
    }

    inline void add_lstm_step(int32_t ticks) { lstm_step.add((double)ticks); }
    inline void add_reg_step(int32_t ticks)  { reg_step.add((double)ticks); }
};

// ---- header: write ONCE at the very top of the report buffer ----
inline void trial_report_header(char* buf, int size)
{
    int n = strlen(buf);
    snprintf(buf + n, size - n,
        "config,window_len,step,"
        "quant_a,quant_b,quant_lstm,quant_reg,"
        "size_a_kb,size_b_kb,size_lstm_kb,size_reg_kb,"
        "infa_ticks_avg,infa_ticks_max,elua_us_avg,elua_us_max,"
        "infb_ticks_avg,infb_ticks_max,elub_us_avg,elub_us_max,"
        "lstm_step_ticks_avg,lstm_step_ticks_max,"
        "reg_step_ticks_avg,reg_step_ticks_max,"
        "window_us_avg,window_us_max,"
        "trial_total_us,"
        "mse_avg,mse_max\n");
}

// ---- one CSV row per trial ----
inline void trial_report_row(char* buf, int size, const TrialStats& t)
{
    int n = strlen(buf);
    snprintf(buf + n, size - n,
        "%d,%ld,%d,"
        "%s,%s,%s,%s,"
        "%.3f,%.3f,%.3f,%.3f,"
        "%.2f,%.0f,%.2f,%.0f,"
        "%.2f,%.0f,%.2f,%.0f,"
        "%.2f,%.0f,"
        "%.2f,%.0f,"
        "%.2f,%.0f,"
        "%llu,"
        "%.6e,%.6e\n",
        t.config, t.window_len, t.step,
        t.quant_a, t.quant_b, t.quant_lstm, t.quant_reg,
        t.size_a_kb, t.size_b_kb, t.size_lstm_kb, t.size_reg_kb,
        t.infa.avg(), t.infa.max(), t.elua.avg(), t.elua.max(),
        t.infb.avg(), t.infb.max(), t.elub.avg(), t.elub.max(),
        t.lstm_step.avg(), t.lstm_step.max(),
        t.reg_step.avg(), t.reg_step.max(),
        t.window_us.avg(), t.window_us.max(),
        (unsigned long long)t.trial_total_us,
        t.mse.avg(), t.mse.max());
}

#endif // TRIAL_REPORT_H