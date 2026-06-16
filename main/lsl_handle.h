#pragma once

extern "C"{
#include "lsl_esp32.h"
#include "wifi_helper.h"
}

#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"


class LSLHandle {
    private:
        lsl_esp32_inlet_t m_inlet;
        lsl_esp32_stream_info_t m_inlet_info;
        

        float* input_window;
        int capacity;
        int size;
        double m_inlet_last_timestamp;
        double m_outlet_last_timestamp;

        lsl_esp32_outlet_t m_outlet;
        lsl_esp32_stream_info_t m_outlet_info;

    public:

        LSLHandle(int window_size);
        ~LSLHandle();

        void init_outlet();
        void init_inlet();
        
        void init_wifi();

        const char* get_inlet_info() const;
        const char* get_outlet_info() const;
        void init_timestep();

        lsl_esp32_err_t pull_samples(float* buffer);
        lsl_esp32_err_t push_samples(float* buffer);

        bool add_to_window();
        void print_current_window();
        float* expose_window();
};