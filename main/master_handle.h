#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "MicroInterface/Model.h"
#include "model_flash.h"

#include "model_flash.h"
#include "lsl_handle.h"

class MasterHandle {
    private:
        LSLHandle* m_lsl_handle;

        Model** m_model;
        ModelFlash* m_model_flash;
        // Window buffer
        float m_input_window[CONFIG_INPUT_CHANNELS][CONFIG_INPUT_WINDOW_SIZE];
        float m_output_window[CONFIG_OUTPUT_CHANNELS][CONFIG_OUTPUT_WINDOW_SIZE];
        
        int m_input_window_size;
        int m_output_window_size;

        char** model_partitions;
        uint32_t* model_partition_sizes;
        int partition_count;
        // 2821448
    
    public:
        MasterHandle(const char** model_partitions, const uint32_t* model_partition_sizes, const int count);
        ~MasterHandle(){};

        void init_models();
        void update_input_window();

        void push_output_window();
        void reset_for_next_window();

        inline bool is_input_window_filled() const { return m_input_window_size >= CONFIG_INPUT_WINDOW_SIZE; }
        inline bool is_output_window_filled() const { return m_output_window_size >= CONFIG_OUTPUT_WINDOW_SIZE; }
        
        void run_inference();

        void dual_inference();
};