    #include "master_handle.h"

    MasterHandle::MasterHandle(const char* models_partition) {
        m_input_window_size = 0;
        m_output_window_size = 0;
        //m_lsl_handle = new LSLHandle();
        this->models_partition = models_partition;
        init_model_refs();
    }  

    void MasterHandle::init_model_refs()
    {
        m_model_flash = new ModelFlash();

        this->models_ptrs = new const uint8_t*[MODEL_COUNT];
        bool success = m_model_flash->allocatePointerOnFlash(this->models_partition, this->models_ptrs, MODEL_COUNT,
             MODEL_OFFSETS, MODEL_SIZES);
        if (!success) {
                ESP_LOGE("MASTERHandle", "Could not initialize mmaped pointers");
                return;
        }
    }
    void MasterHandle::init_model(int model_index, int internal_index) {

        int sizes = internal_index == 0 ? 1 : 3;
        // Initialize
        //ESP_LOGI("MASTERHandle", "Model %d, size=%u", model_index, MODEL_SIZES[model_index]);
        
        m_psram_model_ptr[internal_index] = m_model_flash->allocatePointerOnPSRAM(MODEL_SIZES[model_index]);

        // Transfer model from flash to PSRAM
        memcpy(m_psram_model_ptr[internal_index], this->models_ptrs[model_index], MODEL_SIZES[model_index]);

        uint64_t startInit = esp_timer_get_time();
        
        m_model[internal_index] = new Model(m_model_flash, m_psram_model_ptr[internal_index],
             CONFIG_ARENA_SIZE * 1024, sizes, sizes, true);

        uint64_t durationinit = esp_timer_get_time() - startInit;

        float durationInMs = durationinit / 1000;

        //ESP_LOGI("MASTERHandle", "MODEL TOOK: %lld micro seconds, %0.4f ms, to init", durationinit, durationInMs);

        if (m_model[internal_index]->isInitialized()) {
            //ESP_LOGI("MASTERHandle", "MODEL INTIALIZED SUCCESSFULLY");
        } else {
                ESP_LOGE("MASTERHandle", "Model was not initialized successfully");
            
        }
    }

    void MasterHandle::init_models(int model_1_index, int model_2_index)
    {
        init_model(model_1_index, 0);
        init_model(model_2_index, 1);
    }

    void MasterHandle::update_input_window() {
        // if (m_input_window_size >= CONFIG_INPUT_WINDOW_SIZE) return;  // full, wait for inference

        // float new_sample[CONFIG_INPUT_CHANNELS];

        // while (m_input_window_size < CONFIG_INPUT_WINDOW_SIZE) {
        //     //ESP_LOGI("MASTERHandle", "Pulling samples for input window...");
        //     if (lsl_esp32_err_t err = m_lsl_handle->pull_samples(new_sample); err == LSL_ESP32_OK) {
        //         ESP_LOGI("MASTERHandle", "Success pulling sample for input window");
        //         for (int ch = 0; ch < CONFIG_INPUT_CHANNELS; ch++) {
        //             m_input_window[ch][m_input_window_size] = new_sample[ch];
        //         }
        //         m_input_window_size++;
        //     }else{
        //         if(err != LSL_ESP32_ERR_TIMEOUT) { // timeout is expected when no new samples are available, so only log other errors
        //             ESP_LOGE("MASTERHandle", "Error pulling sample for input window: %d", err);
        //         }
        //     }
        // }
    }

    float MasterHandle::print_output(const float* output_window, int window_len, const float* correct_window)
    {
        printf("OUTPUT DATA (First 5 samples in each channel) ================= \n");

        float mse = 0;
        for(int i =0; i < CONFIG_OUTPUT_CHANNELS; i++)
        {
            for(int j =0; j < window_len; j++)
            {
                //printf("(%d)(%d)[%0.4f] \n", i, j, output_window[(i * window_len) + j]);
                // Calculate MSE
                mse += (output_window[(i * window_len) + j] - correct_window[(i * window_len) + j]) * 
                (output_window[(i * window_len) + j] - correct_window[(i * window_len) + j]);
            }
            printf("==============================\n");
        }

        mse /= (CONFIG_OUTPUT_CHANNELS * window_len);
        return mse;
    }
    void MasterHandle::reset_for_next_window() {
        if (!is_input_window_filled()) return;  // can't reset until first window is filled
        m_input_window_size  = 0;
        m_output_window_size = 0;
    }

    void MasterHandle::push_output_window() {
        // if (!is_output_window_filled()) {
        //     // Not enough data to push
        //     return;
        // }
        // // Push the entire output window sample by sample
        // for (int i = 0; i < CONFIG_OUTPUT_WINDOW_SIZE; i++) {
        //     float sample[CONFIG_OUTPUT_CHANNELS];
        //     for (int ch = 0; ch < CONFIG_OUTPUT_CHANNELS; ch++) {
        //         sample[ch] = m_output_window[ch][i];
        //     }
        //     m_lsl_handle->push_samples(sample);
        // }
    }


    void MasterHandle::run_inference() {

        // assert(m_model[0] != nullptr); // Ensure the model is initialized

        // if (!is_input_window_filled()) {
        //     // Not enough data to run inference
        //     return;
        // }

        // ESP_LOGI("MASTERHandle", "Running inference on filled input window");
        // uint64_t startTime = esp_timer_get_time();
        
        // // Create input pointers which are of size 1 given there is only one input tensors
        // const float* input_ptr = static_cast<const float*>(&m_input_window[0][0]);
        // const int* input_lengths = new const int[1]{CONFIG_INPUT_CHANNELS * CONFIG_INPUT_WINDOW_SIZE}; // Array of lengths for each input tensor
        
        // // Create output pointers which are of size 1 given there is only one output tensors
        // float* output_ptr = static_cast<float*>(&m_output_window[0][0]);
        // const int* output_lengths = new const int[1]{CONFIG_OUTPUT_CHANNELS * CONFIG_OUTPUT_WINDOW_SIZE}; // Array of lengths for each output tensor
        
        // bool success = m_model[0]->predict(input_ptr, input_lengths, output_ptr, output_lengths);
        
        // uint64_t duration = esp_timer_get_time() - startTime;

        // if(success) {
        //     float durationInMs = duration / 1000;
        //     ESP_LOGI("MASTERHandle", "Inference took: %lld micro seconds, %0.4f ms", duration, durationInMs);
        //     m_output_window_size = CONFIG_OUTPUT_WINDOW_SIZE;
        // } else {
        //     ESP_LOGE("MASTERHandle", "Inference failed");
        // }
    }

    void MasterHandle::clear_models()
    {
            delete m_model[0];
            m_model[0] = nullptr;

            delete m_model[1];
            m_model[1] = nullptr;

            heap_caps_free(m_psram_model_ptr[0]);
            m_psram_model_ptr[0] = nullptr;

            heap_caps_free(m_psram_model_ptr[1]);
            m_psram_model_ptr[1] = nullptr;

            ESP_LOGI("MASTERhandle", "2 Models Cleared Successfully");

    }
    void MasterHandle::dual_inference(const float* input_ptr, int input_window_size, float* output_ptr, int output_window_size
    ,char* report_buffer, int size)
    {
        int report_size = strlen(report_buffer);
        assert(m_model[0] != nullptr); // Ensure the model is initialized
        assert(m_model[1] != nullptr); // Ensure the second model is initialized

        // if (!is_input_window_filled()) {
        //     // Not enough data to run inference
        //     return;
        // }

        ESP_LOGI("MASTERHandle", "Running inference on filled input window");
        uint64_t startTime_first = esp_timer_get_time();
        
        // Create input pointers which are of size 2 given there are two input tensors
        const int* input_lengths = new const int[1]{CONFIG_INPUT_CHANNELS * input_window_size}; // Array of lengths for each input tensor
        
        // Create output pointers which are of size 1 given there is only one output tensors
        float* first_output_ptr = new float[
            56 * output_window_size
        ]; 
        const int* output_lengths = new const int[1]{56 * output_window_size}; // Array of lengths for each output tensor
        
        bool success = m_model[0]->predict(input_ptr, input_lengths, first_output_ptr, output_lengths);
        
        uint64_t duration_first = esp_timer_get_time() - startTime_first;

        
        if(success) {
            float durationInMs = duration_first / 1000;
            ESP_LOGI("MASTERHandle", "Inference For First Model took: %lld micro seconds, %0.4f ms", duration_first, durationInMs);
            snprintf(report_buffer + report_size, size - report_size, "0_Model Inf: %lld \xCE\xBCs, %0.2f ms\n", duration_first, durationInMs);
            
            m_output_window_size = output_window_size;
        } else {
            ESP_LOGE("MASTERHandle", "Inference failed");
        }

        // Onto second model
        uint64_t startTime_second = esp_timer_get_time();
        
        float* second_input_ptr = new float[
            (56)
            + (2 * 56)
            + (2 * 56)
        ]{0}; // Assuming the second model takes the first model's output plus two additional inputs of size 56 each
        
        const int* second_input_lengths = new const int[3]{
            56,
            2 * 56,
            2 * 56
        };

    
        float* intermediate_output_ptr = new float[
            (CONFIG_OUTPUT_CHANNELS)
            + (2 * 56)
            + (2 * 56)
        ]{0};

        const int* intermediate_output_lengths = new const int[3]{
            CONFIG_OUTPUT_CHANNELS,
            2 * 56,
            2 * 56
        };
        
        for(int i =0; i < output_window_size; i++) {
            uint64_t intermediateTime_second = esp_timer_get_time();
            // update input to next timestep, h and c (i.e we need to update the second_input_ptr)
            for(int j =0; j < 56; j++) {
                
                second_input_ptr[j] = first_output_ptr[(j * output_window_size) + i]; 
            }
            // update h and c with prev iteration's h and c
            if(i > 0)
            {
                for(int j = 0; j < second_input_lengths[1]; j++) {
                    second_input_ptr[56 + j] = intermediate_output_ptr[CONFIG_OUTPUT_CHANNELS + j];
                }
                for(int j = 0; j < second_input_lengths[2]; j++) {
                    second_input_ptr[56 + (2 * 56) + j] = intermediate_output_ptr[CONFIG_OUTPUT_CHANNELS + (2 * 56) + j];
                }
            }

            success = m_model[1]->predict(second_input_ptr, second_input_lengths, intermediate_output_ptr, intermediate_output_lengths);
            // Transfer the intermediate into the final output buffer for pushing to LSL
            for(int j = 0; j < intermediate_output_lengths[0]; j++) {
                output_ptr[(j * output_window_size) + i] = intermediate_output_ptr[j];
            }   

            uint64_t duration_intermediate = esp_timer_get_time() - intermediateTime_second;
            if(success) {
                float durationInMs = duration_intermediate / 1000;
                //ESP_LOGI("MASTERHandle", "Inference For Second Model, time_step: %d took: %lld micro seconds, %0.4f ms",
                //i, duration_intermediate, durationInMs);
                if (i == 1)
                {
                    report_size = strlen(report_buffer);
                    snprintf(report_buffer + report_size, size - report_size, "1_Model tInf: %lld \xCE\xBCs, %0.2f ms\n",duration_intermediate, durationInMs);

                }
                m_output_window_size++;
            } else {
                ESP_LOGE("MASTERHandle", "Inference failed @ time_step: %d", i);
            }
        }

        uint64_t duration_second = esp_timer_get_time() - startTime_second;

        if(success) {
            float durationInMs = duration_second / 1000;
            //ESP_LOGI("MASTERHandle", "Inference For Second Model took: %lld micro seconds, %0.4f ms", duration_second, durationInMs);
            report_size = strlen(report_buffer);
            snprintf(report_buffer + report_size, size - report_size, "1_Model Inf: %lld \xCE\xBCs, %0.2f ms\n",duration_second, durationInMs);


            m_output_window_size = output_window_size;
        } else {
            ESP_LOGE("MASTERHandle", "Inference failed");
        }
        float durationInMs = (duration_second + duration_first)/1000;
        report_size = strlen(report_buffer);
        snprintf(report_buffer + report_size, size - report_size, "Model Inf: %lld \xCE\xBCs, %0.2f ms\n\n",duration_second + duration_first, durationInMs);

        delete[] input_lengths;              // new'd above
        delete[] first_output_ptr;                 // new'd above (first model's output buffer)
        delete[] output_lengths;             // new'd above
        delete[] second_input_ptr;           // new'd above
        delete[] second_input_lengths;       // new'd above
        delete[] intermediate_output_ptr;    // new'd above
        delete[] intermediate_output_lengths;// new'd above

        ESP_LOGI("MASTERHandle", "CLEARED ALL ARRAYS");
    }