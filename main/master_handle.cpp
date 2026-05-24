#include "master_handle.h"

MasterHandle::MasterHandle() {
    m_input_window_size = 0;
    m_output_window_size = 0;
    
    init_model();
    m_lsl_handle = new LSLHandle();
}  

void MasterHandle::init_model() {
    m_model_flash = new ModelFlash();

    const uint8_t* mmaped_pointer = m_model_flash->allocatePointerOnFlash("model");
    if (!mmaped_pointer) {
        ESP_LOGE("MASTERHandle", "Could not initialize mmaped pointer");
        return;
    }
    uint8_t* psram_model_pointer = m_model_flash->allocatePointerOnPSRAM(m_model_flash->GetModelSize());

    ESP_LOGI("MASTERHandle", "FREE_HEAP,%u",
            (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));

    ESP_LOGI("MASTERHandle", "FREE_PSRAM_HEAP_AFTER_ALLOCATION,%u",
            (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    // Transfer model from flash to PSRAM
    memcpy(psram_model_pointer, mmaped_pointer, m_model_flash->GetModelSize());

    uint64_t startInit = esp_timer_get_time();
    m_model = new Model(m_model_flash,psram_model_pointer, CONFIG_ARENA_SIZE * 1024);
    uint64_t durationinit = esp_timer_get_time() - startInit;

    float durationInMs = durationinit / 1000;

    ESP_LOGI("MASTERHandle", "MODEL TOOK: %lld micro seconds, %0.4f ms, to init", durationinit, durationInMs);

    ESP_LOGI("MASTERHandle", "MODEL CREATED");

    if (m_model->isInitialized()) {
        ESP_LOGI("MASTERHandle", "MODEL INTIALIZED SUCCESSFULLY");
    } else {
            ESP_LOGE("MASTERHandle", "Model was not initialized successfully");
        
    }
}

void MasterHandle::update_input_window() {
    if (m_input_window_size >= CONFIG_INPUT_WINDOW_SIZE) return;  // full, wait for inference

    float new_sample[CONFIG_INPUT_CHANNELS];

    while (m_input_window_size < CONFIG_INPUT_WINDOW_SIZE) {
        //ESP_LOGI("MASTERHandle", "Pulling samples for input window...");
        if (lsl_esp32_err_t err = m_lsl_handle->pull_samples(new_sample); err == LSL_ESP32_OK) {
            ESP_LOGI("MASTERHandle", "Success pulling sample for input window");
            for (int ch = 0; ch < CONFIG_INPUT_CHANNELS; ch++) {
                m_input_window[ch][m_input_window_size] = new_sample[ch];
            }
            m_input_window_size++;
        }else{
            if(err != LSL_ESP32_ERR_TIMEOUT) { // timeout is expected when no new samples are available, so only log other errors
                ESP_LOGE("MASTERHandle", "Error pulling sample for input window: %d", err);
            }
        }
    }
}

void MasterHandle::reset_for_next_window() {
    if (!is_input_window_filled()) return;  // can't reset until first window is filled
    m_input_window_size  = 0;
    m_output_window_size = 0;
}

void MasterHandle::push_output_window() {
    if (!is_output_window_filled()) {
        // Not enough data to push
        return;
    }
    // Push the entire output window sample by sample
    for (int i = 0; i < CONFIG_OUTPUT_WINDOW_SIZE; i++) {
        float sample[CONFIG_OUTPUT_CHANNELS];
        for (int ch = 0; ch < CONFIG_OUTPUT_CHANNELS; ch++) {
            sample[ch] = m_output_window[ch][i];
        }
        m_lsl_handle->push_samples(sample);
    }
}


void MasterHandle::run_inference() {

    if (!is_input_window_filled()) {
        // Not enough data to run inference
        return;
    }

    ESP_LOGI("MASTERHandle", "Running inference on filled input window");
    uint64_t startTime = esp_timer_get_time();
    bool success = m_model->predict(
        &m_input_window[0][0], CONFIG_INPUT_CHANNELS * CONFIG_INPUT_WINDOW_SIZE, &m_output_window[0][0],
        CONFIG_OUTPUT_CHANNELS * CONFIG_OUTPUT_WINDOW_SIZE);
    uint64_t duration = esp_timer_get_time() - startTime;

    if(success) {
        float durationInMs = duration / 1000;
        ESP_LOGI("MASTERHandle", "Inference took: %lld micro seconds, %0.4f ms", duration, durationInMs);
        m_output_window_size = CONFIG_OUTPUT_WINDOW_SIZE;
    } else {
        ESP_LOGE("MASTERHandle", "Inference failed");
    }
}