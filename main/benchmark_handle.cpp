#include "benchmark_handle.h"
BenchmarkHandle::BenchmarkHandle(const char* models_partition,
                                 int model_count,
                                 const uint32_t* model_offsets,
                                 const uint32_t* model_sizes) {
    this->models_partition = models_partition;

    // Store the injected manifest data. The arrays live in the manifest
    // header included by main.cpp; we just keep pointers to them.
    this->m_model_count   = model_count;
    this->m_model_offsets = model_offsets;
    this->m_model_sizes   = model_sizes;

    init_model_refs();
}

void BenchmarkHandle::init_model_refs()
{
    m_model_flash = new ModelFlash();

    this->models_ptrs = new const uint8_t*[m_model_count];
    bool success = m_model_flash->allocatePointerOnFlash(this->models_partition, this->models_ptrs, m_model_count, m_model_offsets);
    if (!success) {
            ESP_LOGE("MASTERHandle", "Could not initialize mmaped pointers");
            return;
    }

    // Allocate persistent arenas + model buffer ONCE. Reused for all models;
    // never freed during the sweep, so the heap cannot fragment.
    m_arena_psram = m_model_flash->allocatePointerOnPSRAM(PSRAM_ARENA_BYTES);
    m_arena_sram  = static_cast<uint8_t*>(
        heap_caps_malloc(SRAM_ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

    // Model buffer sized to the largest model in the set (worst-case).
    m_model_buf_cap = 0;
    for (int i = 0; i < m_model_count; i++)
        if (m_model_sizes[i] > m_model_buf_cap)
            m_model_buf_cap = m_model_sizes[i];
    m_model_buf = m_model_flash->allocatePointerOnPSRAM(m_model_buf_cap);

    if (!m_arena_psram || !m_arena_sram || !m_model_buf) {
        ESP_LOGE("MASTERHandle", "Persistent buffer alloc failed (psram=%p sram=%p buf=%p)",
                 m_arena_psram, m_arena_sram, m_model_buf);
    } else {
        ESP_LOGI("MASTERHandle", "Persistent: SRAM arena %d B, PSRAM arena %d B, model buf %zu B",
                 SRAM_ARENA_BYTES, PSRAM_ARENA_BYTES, m_model_buf_cap);
    }
}


void BenchmarkHandle::init_model(int model_index, bool usePSRAM, int input_size,
                                 int output_size, char* report_buffer, int size,
                                 int arena_bytes) {
    (void)arena_bytes;   // persistent arenas are fixed-size; full arena exposed
    this->inPSRAM = usePSRAM;
    ESP_LOGI("MASTERHandle", "Model %d, size=%u", model_index,
             (unsigned)m_model_sizes[model_index]);

    // Copy the flatbuffer into the persistent model buffer (no per-model alloc).
    m_model_ptr = m_model_buf;
    memcpy(m_model_ptr, this->models_ptrs[model_index], m_model_sizes[model_index]);

    // Pick the persistent arena for this memory and expose it in full.
    uint8_t* arena = this->inPSRAM ? m_arena_psram : m_arena_sram;
    int      arena_cap = this->inPSRAM ? PSRAM_ARENA_BYTES : SRAM_ARENA_BYTES;

    uint64_t startInit = esp_timer_get_time();
    m_model = new Model(m_model_flash, m_model_ptr, arena, arena_cap,   // borrowing ctor
                        input_size, output_size, this->inPSRAM, report_buffer, size);
    
    // m_model = new Model(m_model_flash, m_model_ptr, arena_cap, 
    //                     input_size, output_size, this->inPSRAM, report_buffer, size);
    
    // if (this->inPSRAM) {
    //     m_model_ptr = m_model_flash->allocatePointerOnPSRAM(m_model_sizes[model_index]);
    //     memcpy(m_model_ptr, this->models_ptrs[model_index], m_model_sizes[model_index]);
    //     m_model = new Model(m_model_flash, m_model_ptr, PSRAM_ARENA_BYTES,
    //                         input_size, output_size, true, report_buffer, size);
    // } else {
    //     m_model_ptr = m_model_buf;
    //     memcpy(m_model_ptr, this->models_ptrs[model_index], m_model_sizes[model_index]);
    //     m_model = new Model(m_model_flash, m_model_ptr, m_arena_sram, SRAM_ARENA_BYTES,
    //                         input_size, output_size, false, report_buffer, size);
    // }
    uint64_t durationinit = esp_timer_get_time() - startInit;
    float durationInMs = durationinit / 1000;

    ESP_LOGI("MASTERHandle", "MODEL TOOK: %lld micro seconds, %0.4f ms, to init",
             durationinit, durationInMs);
    if (m_model->isInitialized()) {
        ESP_LOGI("MASTERHandle", "MODEL INITIALIZED SUCCESSFULLY");
    } else {
        ESP_LOGE("MASTERHandle", "Model was not initialized successfully");
    }
}
float BenchmarkHandle::print_output(const float* output_window, int output_size, const float* correct_window)
{
    float mse = 0;

    for(int j =0; j < output_size; j++)
    {
        if(j< 5)
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

            assert(m_model != nullptr); // Ensure the model is initialized
            ESP_LOGI("MASTERHandle", "Running inference on filled input window");
            uint64_t startTime_first = esp_timer_get_time();

            bool success = m_model->predict(input_ptr, input_sizes, output_ptr, output_sizes);

            uint64_t duration_first = esp_timer_get_time() - startTime_first;

        if(success) {
            float durationInMs = duration_first / 1000;
            // int report_size = strlen(report_buffer);
            // snprintf(report_buffer + report_size, size - report_size, "0_Model Inf: %lld \xCE\xBCs, %0.2f ms\n", duration_first, durationInMs);

        } else {
            ESP_LOGE("MASTERHandle", "Inference failed");
        }
}

void BenchmarkHandle::clear_models()
{
    delete m_model;        // frees interpreter + input/output arrays only;
    m_model = nullptr;     // borrowed arena is left intact (owns_arena == false)

    m_model_ptr = nullptr; // points into persistent m_model_buf; do NOT free

    ESP_LOGI("MASTERhandle", "Model Cleared Successfully");
}

BenchmarkHandle::~BenchmarkHandle()
{
    if (m_arena_psram) heap_caps_free(m_arena_psram);
    if (m_arena_sram)  heap_caps_free(m_arena_sram);
    if (m_model_buf)   heap_caps_free(m_model_buf);
    m_arena_psram = m_arena_sram = m_model_buf = nullptr;
}