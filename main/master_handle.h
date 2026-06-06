#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "MicroInterface/Model.h"

#include "models_manifest.h"
#include "model_flash.h"

#include "model_flash.h"
#include "lsl_handle.h"

class MasterHandle {
    private:
        LSLHandle* m_lsl_handle;

        Model** m_model;
        uint8_t* m_psram_model_ptr[2];
        ModelFlash* m_model_flash;
        // Window buffer
      
        int m_input_window_size;
        int m_output_window_size;

        const char* models_partition;
        
        uint32_t* input_sizes;
        uint32_t* output_sizes;


        const uint8_t** models_ptrs;
    
    public:
        MasterHandle(const char* models_partition);
        ~MasterHandle(){};

        void init_model_refs();
        void init_model(int model_index, int internal_index, char* report_buffer, int size);
        void init_models(int model_1_index, int model_2_index, char* report_buffer, int size);
        void update_input_window();

        float print_output(const float* output_window, int window_len, const float* correct_window);

        void push_output_window();
        void reset_for_next_window();

        inline bool is_input_window_filled() const { return m_input_window_size >= CONFIG_INPUT_WINDOW_SIZE; }
        inline bool is_output_window_filled() const { return m_output_window_size >= CONFIG_OUTPUT_WINDOW_SIZE; }
        
        void clear_models();
        void run_inference();

        void dual_inference(const float* input_ptr, int input_window_size, float* output_ptr, int output_window_size,
        char* report_buffer, int size);
};