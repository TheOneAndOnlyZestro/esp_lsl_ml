#include "benchmark_handle.h"

BenchmarkHandle::BenchmarkHandle(const char* models_partition) {
    this->models_partition = models_partition;
    //m_lsl_handle = new LSLHandle();
    init_model_refs();
}  

void BenchmarkHandle::init_model_refs()
{
    m_model_flash = new ModelFlash();

    this->models_ptrs = new const uint8_t*[BENCHMARK_MODEL_COUNT];
    bool success = m_model_flash->allocatePointerOnFlash(this->models_partition, this->models_ptrs, BENCHMARK_MODEL_COUNT, BENCHMARK_MODEL_OFFSETS, BENCHMARK_MODEL_SIZES);
    if (!success) {
            ESP_LOGE("MASTERHandle", "Could not initialize mmaped pointers");
            return;
    }
}
void BenchmarkHandle::init_model(int model_index, bool usePSRAM ,int input_size, int output_size) {
    this->inPSRAM = usePSRAM;
    // Initialize
    ESP_LOGI("MASTERHandle", "Model %d, size=%u", model_index, BENCHMARK_MODEL_SIZES[model_index]);
    if(this->inPSRAM)
        m_model_ptr = m_model_flash->allocatePointerOnPSRAM(BENCHMARK_MODEL_SIZES[model_index]);
    else
        m_model_ptr = static_cast<uint8_t*>(malloc(BENCHMARK_MODEL_SIZES[model_index]));

    // Transfer model from flash to PSRAM
    memcpy(m_model_ptr, this->models_ptrs[model_index], BENCHMARK_MODEL_SIZES[model_index]);

    uint64_t startInit = esp_timer_get_time();
    m_model = new Model(m_model_flash, m_model_ptr, CONFIG_ARENA_SIZE * 1024, input_size, output_size, this->inPSRAM);
    uint64_t durationinit = esp_timer_get_time() - startInit;

    float durationInMs = durationinit / 1000;

    ESP_LOGI("MASTERHandle", "MODEL TOOK: %lld micro seconds, %0.4f ms, to init", durationinit, durationInMs);
    if (m_model->isInitialized()) {
        ESP_LOGI("MASTERHandle", "MODEL INTIALIZED SUCCESSFULLY");
    } else {
            ESP_LOGE("MASTERHandle", "Model was not initialized successfully");
        
    }
}


float BenchmarkHandle::print_output(const float* output_window, int output_size, const float* correct_window)
{
    float mse = 0;
    
    for(int j =0; j < output_size; j++)
    {
        if(j > 10 && j < 15)
            printf("(%d)[%0.4f],  (%d)[%0.4f]\n", j, output_window[j], j, correct_window[j]);
        // Calculate MSE
        mse += (output_window[j] - correct_window[j]) * 
        (output_window[j] - correct_window[j]);
    }
    
    mse /= (output_size);
    return mse;
}



void BenchmarkHandle::run_inference(const float* input_ptr, const int* input_sizes, float* output_ptr, int* output_sizes
        ,char* report_buffer, int size) {

        int report_size = strlen(report_buffer);
        assert(m_model != nullptr); // Ensure the model is initialized
        ESP_LOGI("MASTERHandle", "Running inference on filled input window");
        uint64_t startTime_first = esp_timer_get_time();
        
        bool success = m_model->predict(input_ptr, input_sizes, output_ptr, output_sizes);
        
        uint64_t duration_first = esp_timer_get_time() - startTime_first;

        if(success) {
            float durationInMs = duration_first / 1000;
            snprintf(report_buffer + report_size, size - report_size, "0_Model Inf: %lld \xCE\xBCs, %0.2f ms\n", duration_first, durationInMs);
            
        } else {
            ESP_LOGE("MASTERHandle", "Inference failed");
        }
}

void BenchmarkHandle::clear_models()
{
    delete m_model;
    m_model = nullptr;
    
    if(this->inPSRAM)
        heap_caps_free(m_model_ptr);
    else
        free(m_model_ptr);

    m_model_ptr = nullptr;

    ESP_LOGI("MASTERhandle", "Model Cleared Successfully");
}
