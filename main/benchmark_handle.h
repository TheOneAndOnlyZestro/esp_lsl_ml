#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binary_manifests/dense_layers_seed1/models_manifest_0.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"
#include "MicroInterface/Model.h"

#include "model_flash.h"
#include "lsl_handle.h"

class BenchmarkHandle {
    private:
        LSLHandle* m_lsl_handle;

        Model* m_model;
        
        uint8_t* m_model_ptr; // IN RAM
        bool inPSRAM = false;
        ModelFlash* m_model_flash;
        
        const char* models_partition;
        
        const uint8_t** models_ptrs; // ON FLASH

        // Persistent buffers — allocated ONCE, reused for every model, freed
        // only in the destructor. This is what prevents heap fragmentation.
        uint8_t* m_arena_sram  = nullptr;   // internal-RAM arena (180 KB)
        uint8_t* m_arena_psram = nullptr;   // PSRAM arena (CONFIG_ARENA_SIZE)
        uint8_t* m_model_buf   = nullptr;   // model-copy buffer (worst-case)

        static const int SRAM_ARENA_BYTES  = 180 * 1024;
        static const int PSRAM_ARENA_BYTES = CONFIG_ARENA_SIZE * 1024;
        size_t   m_model_buf_cap = 0;       // capacity of m_model_buf
    
    public:
        BenchmarkHandle(const char* models_partition);
        ~BenchmarkHandle();

        void init_model_refs();
        // void init_model(int model_index, bool usePSRAM, int input_size,
        //                          int output_size, char* report_buffer, int size,
        //                          size_t arena_bytes);

        void init_model(int model_index, bool usePSRAM, int input_size,
                                 int output_size, char* report_buffer, int size,
                                 int arena_bytes = 0);

        float print_output(const float* output_window, int output_size, const float* correct_window);

        void clear_models();

        void run_inference(const float* input_ptr, const int* input_sizes, float* output_ptr, int* output_sizes
        ,char* report_buffer, int size);

public:
    // True iff the most recently init'd model constructed and allocated tensors.
    bool model_ok() const {
        return m_model != nullptr && m_model->isInitialized();
    }
 
    // Measured arena usage of the current model (0 if none / not initialized).
    size_t arena_used() const {
        return m_model ? m_model->getArenaUsedBytes() : 0;
    }
 
    // Run one inference and return the profiler tick total for THAT invocation.
    // Clears the profiler first so the value reflects exactly one Invoke().
    // ticks == microseconds on this target (ticks_per_second == 1000000).
    // Returns -1 on failure.
    int32_t run_inference_ticks(const float* input_ptr, const int* input_sizes,
                                float* output_ptr, int* output_sizes) {
        if (!model_ok()) return -1;
        m_model->ClearProfiler();
        bool ok = m_model->predict(input_ptr, input_sizes, output_ptr, output_sizes);
        if (!ok) return -1;
        return m_model->getTotalTicks();
    }
 
    // Mean squared error, no printf.
    static float compute_mse(const float* out, int n, const float* correct) {
        float mse = 0.0f;
        for (int j = 0; j < n; j++) {
            const float d = out[j] - correct[j];
            mse += d * d;
        }
        return (n > 0) ? mse / (float)n : 0.0f;
    }
 
    // Signal-to-quantization-noise ratio in dB. Scale-free, so it stays
    // interpretable even when output magnitude drifts with depth.
    //   SQNR_dB = 10 * log10( sum(correct^2) / sum((out-correct)^2) )
    //   error == 0 (float32, exact)  -> +INFINITY   (CSV writes "inf")
    //   signal == 0 (degenerate)     -> NAN         (CSV writes empty cell)
    // Accumulate in double: deep stacks can have large output magnitudes.
    static float compute_sqnr_db(const float* out, int n, const float* correct) {
        if (n <= 0) return NAN;
        double sig = 0.0, err = 0.0;
        for (int j = 0; j < n; j++) {
            const double c = (double)correct[j];
            const double d = (double)out[j] - c;
            sig += c * c;
            err += d * d;
        }
        if (sig <= 0.0) return NAN;
        if (err <= 0.0) return INFINITY;
        return (float)(10.0 * log10(sig / err));
    }

};