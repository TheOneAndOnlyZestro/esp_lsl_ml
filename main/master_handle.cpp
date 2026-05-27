#include "master_handle.h"

MasterHandle::MasterHandle(char** model_partitions, const uint32_t* model_partition_sizes,
    const uint32_t* input_sizes, const uint32_t* output_sizes ,const int count) {
    m_input_window_size = 0;
    m_output_window_size = 0;
    
    this->model_partition_sizes = new uint32_t[count];
    for (int i = 0; i < count; i++) {
        this->model_partition_sizes[i] = model_partition_sizes[i];
    }

    this->input_sizes = new uint32_t[count];
    for (int i = 0; i < count; i++) {
        this->input_sizes[i] = input_sizes[i];
    }

    this->output_sizes = new uint32_t[count];
    for (int i = 0; i < count; i++) {
        this->output_sizes[i] = output_sizes[i];
    }   

    this->model_partitions = new char*[count];
    this->partition_count = count;

    for (size_t i = 0; i < count; i++) {
        size_t label_len = strlen(model_partitions[i]) + 1;
        this->model_partitions[i] = new char[label_len];
        strncpy(this->model_partitions[i], model_partitions[i], label_len);
        this->model_partition_sizes[i] = model_partition_sizes[i];
    }
    
    init_models();
    m_lsl_handle = new LSLHandle();
}  

void MasterHandle::init_models() {
    m_model_flash = new ModelFlash(this->model_partitions, 
    static_cast<const uint32_t*>(this->model_partition_sizes), this->partition_count);

    for(int i = 0; i < partition_count; i++) {
        ESP_LOGI("MASTERHandle", "ModelFlash partition %d: label=%s, size=%u", i, this->model_partitions[i], this->model_partition_sizes[i]);
        
        const uint8_t* mmaped_pointer = m_model_flash->allocatePointerOnFlash(i);
        if (!mmaped_pointer) {
            ESP_LOGE("MASTERHandle", "Could not initialize mmaped pointer");
            return;
        }
        uint8_t* psram_model_pointer = m_model_flash->allocatePointerOnPSRAM(m_model_flash->GetModelSize(i));

        ESP_LOGI("MASTERHandle", "FREE_HEAP,%u",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));

        ESP_LOGI("MASTERHandle", "FREE_PSRAM_HEAP_AFTER_ALLOCATION,%u",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

        // Transfer model from flash to PSRAM
        memcpy(psram_model_pointer, mmaped_pointer, m_model_flash->GetModelSize(i));

        uint64_t startInit = esp_timer_get_time();
        m_model[i] = new Model(m_model_flash,psram_model_pointer, CONFIG_ARENA_SIZE * 1024, input_sizes[i], output_sizes[i]);
        uint64_t durationinit = esp_timer_get_time() - startInit;

        float durationInMs = durationinit / 1000;

        ESP_LOGI("MASTERHandle", "MODEL TOOK: %lld micro seconds, %0.4f ms, to init", durationinit, durationInMs);

        ESP_LOGI("MASTERHandle", "MODEL CREATED");

        if (m_model[i]->isInitialized()) {
            ESP_LOGI("MASTERHandle", "MODEL INTIALIZED SUCCESSFULLY");
        } else {
                ESP_LOGE("MASTERHandle", "Model was not initialized successfully");
            
        }
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

    assert(m_model[0] != nullptr); // Ensure the model is initialized

    if (!is_input_window_filled()) {
        // Not enough data to run inference
        return;
    }

    ESP_LOGI("MASTERHandle", "Running inference on filled input window");
    uint64_t startTime = esp_timer_get_time();
    
    // Create input pointers which are of size 1 given there is only one input tensors
    const float* input_ptr = static_cast<const float*>(&m_input_window[0][0]);
    const int* input_lengths = new const int[1]{CONFIG_INPUT_CHANNELS * CONFIG_INPUT_WINDOW_SIZE}; // Array of lengths for each input tensor
    
    // Create output pointers which are of size 1 given there is only one output tensors
    float* output_ptr = static_cast<float*>(&m_output_window[0][0]);
    const int* output_lengths = new const int[1]{CONFIG_OUTPUT_CHANNELS * CONFIG_OUTPUT_WINDOW_SIZE}; // Array of lengths for each output tensor
    
    bool success = m_model[0]->predict(input_ptr, input_lengths, output_ptr, output_lengths);
    
    uint64_t duration = esp_timer_get_time() - startTime;

    if(success) {
        float durationInMs = duration / 1000;
        ESP_LOGI("MASTERHandle", "Inference took: %lld micro seconds, %0.4f ms", duration, durationInMs);
        m_output_window_size = CONFIG_OUTPUT_WINDOW_SIZE;
    } else {
        ESP_LOGE("MASTERHandle", "Inference failed");
    }
}

void MasterHandle::dual_inference()
{
    assert(m_model[0] != nullptr); // Ensure the model is initialized
    assert(m_model[1] != nullptr); // Ensure the second model is initialized

    if (!is_input_window_filled()) {
        // Not enough data to run inference
        return;
    }

    ESP_LOGI("MASTERHandle", "Running inference on filled input window");
    uint64_t startTime_first = esp_timer_get_time();
    
    // Create input pointers which are of size 2 given there are two input tensors
    const float* input_ptr = static_cast<const float*>(&m_input_window[0][0]);
    const int* input_lengths = new const int[1]{CONFIG_INPUT_CHANNELS * CONFIG_INPUT_WINDOW_SIZE}; // Array of lengths for each input tensor
    
    // Create output pointers which are of size 1 given there is only one output tensors
    float* output_ptr = new float[
        56 * CONFIG_OUTPUT_WINDOW_SIZE
    ]; 
    const int* output_lengths = new const int[1]{56 * CONFIG_OUTPUT_WINDOW_SIZE}; // Array of lengths for each output tensor
    
    bool success = m_model[0]->predict(input_ptr, input_lengths, output_ptr, output_lengths);
    
    uint64_t duration_first = esp_timer_get_time() - startTime_first;

    if(success) {
        float durationInMs = duration_first / 1000;
        ESP_LOGI("MASTERHandle", "Inference For First Model took: %lld micro seconds, %0.4f ms", duration_first, durationInMs);
        m_output_window_size = CONFIG_OUTPUT_WINDOW_SIZE;
    } else {
        ESP_LOGE("MASTERHandle", "Inference failed");
    }

    // Onto second model

    uint64_t startTime_second = esp_timer_get_time();
    
    float* second_input_ptr = new float[
        (56 * CONFIG_OUTPUT_WINDOW_SIZE)
        + (2 * 56)
        + (2 * 56)
    ]; // Assuming the second model takes the first model's output plus two additional inputs of size 56 each
    
    const int* second_input_lengths = new const int[3]{
        56 * CONFIG_OUTPUT_WINDOW_SIZE,
        2 * 56,
        2 * 56
    };

   
    float* intermediate_output_ptr = new float[
        (CONFIG_OUTPUT_CHANNELS)
        + (2 * 56)
        + (2 * 56)
    ];

    const int* intermediate_output_lengths = new const int[3]{
        CONFIG_OUTPUT_CHANNELS,
        2 * 56,
        2 * 56
    };
    
    for(int i =0; i < CONFIG_OUTPUT_WINDOW_SIZE; i++) {
        uint64_t intermediateTime_second = esp_timer_get_time();
        // update input to next timestep, h and c (i.e we need to update the second_input_ptr)
        for(int j =0; j < input_lengths[0]; j++) {
            second_input_ptr[j] = output_ptr[j * 56 + i]; 
        }
        // update h and c with prev iteration's h and c
        if(i > 0)
        {
            for(int j = 0; j < input_lengths[1]; j++) {
                second_input_ptr[56 * CONFIG_OUTPUT_WINDOW_SIZE + j] = intermediate_output_ptr[CONFIG_OUTPUT_CHANNELS + j];
            }
            for(int j = 0; j < input_lengths[2]; j++) {
                second_input_ptr[56 * CONFIG_OUTPUT_WINDOW_SIZE + (2 * 56) + j] = intermediate_output_ptr[CONFIG_OUTPUT_CHANNELS + (2 * 56) + j];
            }
        }
        success = m_model[1]->predict(second_input_ptr, second_input_lengths, intermediate_output_ptr, intermediate_output_lengths);
        // Transfer the intermediate into the final output buffer for pushing to LSL
        for(int j = 0; j < CONFIG_OUTPUT_CHANNELS; j++) {
            m_output_window[j][i] = intermediate_output_ptr[j];
        }   

        uint64_t duration_intermediate = esp_timer_get_time() - intermediateTime_second;

        if(success) {
            float durationInMs = duration_intermediate / 1000;

            ESP_LOGI("MASTERHandle", "Inference For Second Model, time_step: %d took: %lld micro seconds, %0.4f ms",
            i, duration_intermediate, durationInMs);

            m_output_window_size = CONFIG_OUTPUT_WINDOW_SIZE;
        } else {
            ESP_LOGE("MASTERHandle", "Inference failed @ time_step: %d", i);
        }
    }


    uint64_t duration_second = esp_timer_get_time() - startTime_second;

    if(success) {
        float durationInMs = duration_second / 1000;
        ESP_LOGI("MASTERHandle", "Inference For Second Model took: %lld micro seconds, %0.4f ms", duration_second, durationInMs);
        m_output_window_size = CONFIG_OUTPUT_WINDOW_SIZE;
    } else {
        ESP_LOGE("MASTERHandle", "Inference failed");
    }

}