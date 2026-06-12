#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "MicroInterface/Model.h"

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
        
        uint32_t* input_sizes;
        uint32_t* output_sizes;

        int models_count;
        const uint8_t** flash_models_ptrs;
        const uint32_t* model_input_sizes;
        const uint32_t* model_output_sizes;
        const uint32_t* model_sizes;

        const int feature_ch = 56;
    public:
        MasterHandle(const int models_count,
            const uint8_t** flash_models_ptrs,
            const uint32_t* model_input_sizes,
            const uint32_t* model_output_sizes,
            const uint32_t* model_sizes
        );

        ~MasterHandle(){};
        void init_model(int model_index, int internal_index,
        char* report_buffer, int size);
        void init_models(int model_1_index, int model_2_index, char* report_buffer, int size);
        void update_input_window();

        float print_output(const float* output_window, int output_size, const float* correct_window);

        void push_output_window();
        void reset_for_next_window();

        inline bool is_input_window_filled() const { return m_input_window_size >= CONFIG_INPUT_WINDOW_SIZE; }
        inline bool is_output_window_filled() const { return m_output_window_size >= CONFIG_OUTPUT_WINDOW_SIZE; }
        
        void clear_models();
        void run_inference();

        void dual_inference(const float* input_ptr, int input_size,
                                  float* output_ptr, int output_size,
                                  int intermediate_size, int first_out_len,
                                  char* report_buffer, int size);
};