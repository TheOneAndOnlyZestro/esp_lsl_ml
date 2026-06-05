#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "benchmark_models_manifest.h"

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
    
    public:
        BenchmarkHandle(const char* models_partition);
        ~BenchmarkHandle(){};

        void init_model_refs();
        void init_model(int model_index, bool usePSRAM, int input_size, int output_size);

        float print_output(const float* output_window, int output_size, const float* correct_window);

        void clear_models();

        void run_inference(const float* input_ptr, const int* input_sizes, float* output_ptr, int* output_sizes
        ,char* report_buffer, int size);

};