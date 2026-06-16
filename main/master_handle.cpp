#include "master_handle.h"

MasterHandle::MasterHandle(const int models_count,
    const uint8_t** flash_models_ptrs,
    const uint32_t* model_input_sizes,
    const uint32_t* model_output_sizes,
    const uint32_t* model_sizes
)
:
models_count(models_count),
flash_models_ptrs(flash_models_ptrs),
model_input_sizes(model_input_sizes),
model_output_sizes(model_output_sizes),
model_sizes(model_sizes)
{
}  

void MasterHandle::init_model(int model_index, int internal_index,
     char* report_buffer, int size) {

    if(this->usePSRAM)
        m_psram_model_ptr[internal_index] = m_model_flash->allocatePointerOnPSRAM(this->model_sizes[model_index]);
    else
    {
        m_psram_model_ptr[internal_index] = (uint8_t*)malloc(this->model_sizes[model_index]);
        printf("MALLOCING");
    }
    // Transfer model from flash to PSRAM
    memcpy(m_psram_model_ptr[internal_index], this->flash_models_ptrs[model_index],
         this->model_sizes[model_index]);

    uint64_t startInit = esp_timer_get_time();
    
    m_model[internal_index] = new Model(
        m_model_flash, 
        m_psram_model_ptr[internal_index],
        CONFIG_ARENA_SIZE * 1024,
        this->model_input_sizes[internal_index],
        this->model_output_sizes[internal_index],
        this->usePSRAM, report_buffer, size
    );

    uint64_t durationinit = esp_timer_get_time() - startInit;

    float durationInMs = durationinit / 1000;

    //ESP_LOGI("MASTERHandle", "MODEL TOOK: %lld micro seconds, %0.4f ms, to init", durationinit, durationInMs);

    if (m_model[internal_index]->isInitialized()) {
        //ESP_LOGI("MASTERHandle", "MODEL INTIALIZED SUCCESSFULLY");
    } else {
            ESP_LOGE("MASTERHandle", "Model was not initialized successfully");
        
    }
}

void MasterHandle::init_models(int model_1_index, int model_2_index,int model_3_index,
    bool usePSRAM, char* report_buffer, int size)
{
    this->usePSRAM = usePSRAM;
    this->m_model = new Model*[3];
    init_model(model_1_index, 0 ,report_buffer, size);
    init_model(model_2_index, 1 ,report_buffer, size);
    init_model(model_3_index, 2 ,report_buffer, size);
}


float MasterHandle::print_output(const float* output_window, int output_size, const float* correct_window)
{
    printf("OUTPUT DATA (5 samples) ================= \n");

    float mse = 0;
    for(int j =0; j < output_size; j++)
    {
        // Calculate MSE
        if(j % (int)(output_size / 5) == 0)
        printf("(%d)[%0.4f],  (%d)[%0.4f]\n", j, output_window[j], j, correct_window[j]);
        // Calculate MSE
        mse += (output_window[j] - correct_window[j]) * 
        (output_window[j] - correct_window[j]);
    }
    printf("==============================\n");
    
    mse /= (output_size);
    return mse;
}


void MasterHandle::clear_models()
{
        delete m_model[0];
        m_model[0] = nullptr;

        delete m_model[1];
        m_model[1] = nullptr;

        delete m_model[2];
        m_model[2] = nullptr;

        if(this->usePSRAM)
            heap_caps_free(m_psram_model_ptr[0]);
        else
            free(m_psram_model_ptr[0]);

        m_psram_model_ptr[0] = nullptr;

        if(this->usePSRAM)
            heap_caps_free(m_psram_model_ptr[1]);
        else
            free(m_psram_model_ptr[1]);

        m_psram_model_ptr[1] = nullptr;

        if(this->usePSRAM)
            heap_caps_free(m_psram_model_ptr[2]);
        else
            free(m_psram_model_ptr[2]);

        m_psram_model_ptr[2] = nullptr;

        ESP_LOGI("MASTERhandle", "2 Models Cleared Successfully");

}

void MasterHandle::display_output(const float* output_window, int output_size)
{
    for(int i =0; i < output_size; i++)
    {
        printf("(%d)[%0.4f]\n", i, output_window[i]);
    }
}

void MasterHandle::build_elu_lut(float in_scale, int32_t in_zp,
                        float out_scale, int32_t out_zp, float alpha)
{
            for (int i = 0; i < 256; ++i) {
                int8_t qx = (int8_t)(i - 128);
                float x = ((int)qx - in_zp) * in_scale;
                float y = (x >= 0.0f) ? x : alpha * (expf(x) - 1.0f);
                int q = (int)lroundf(y / out_scale) + out_zp;
                elu_lut[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
            }
}

void MasterHandle::apply_elu_lut(int8_t *t, int n)
{
            for (int i = 0; i < n; ++i)
                t[i] = elu_lut[(uint8_t)((int)t[i] + 128)];
}

void MasterHandle::apply_elu_lut_int_float(
        int8_t* in, int n, float in_scale, int32_t in_zp, float alpha,
        float* out)
{
    for (int i = 0; i < n; ++i) {
        float x = ((int32_t)in[i] - in_zp) * in_scale;
        out[i] = (x >= 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}
void MasterHandle::dual_inference(const float* input_ptr, int input_size,
                                  float* output_ptr, int output_size,
                                  int intermediate_size, int first_out_len,
                                  char* report_buffer, int size)
{

    int report_size = 0;
    assert(m_model[0] != nullptr); // Ensure the first a model is initialized
    assert(m_model[1] != nullptr); // Ensure the first b model is initialized
    assert(m_model[2] != nullptr); // Ensure the second model is initialized


    const int state_size       = 2 * feature_ch;             // (2, 1, 56) flattened
    const int step             = intermediate_size / feature_ch;
    const int out_len          = output_size / CONFIG_OUTPUT_CHANNELS;   // T_eff
    const int in_len           = input_size / CONFIG_INPUT_CHANNELS;
    const int first_a_out_size = feature_ch * in_len;          // 56 * window_len (56, 50)
    const int first_b_out_size = feature_ch * first_out_len;          // 56 * T

    assert(step >= 1 && intermediate_size % feature_ch == 0);
    assert(output_size % CONFIG_OUTPUT_CHANNELS == 0);
    assert(out_len % step == 0);              // second model runs out_len/step times
    assert(out_len <= first_out_len);         // T_eff = (T/step)*step <= T
    assert(first_out_len - out_len < step);   // only a ragged tail may be skipped
 
    ESP_LOGI("MASTERHandle", "dual_inference: in=%d, out=%d, inter=%d (step=%d, out_len=%d, T=%d)",
             input_size, output_size, intermediate_size, step, out_len, first_out_len);
    
    uint64_t startTime_first_a = esp_timer_get_time();
 
    const int* input_lengths_a = new const int[1]{ input_size };
    
    // PROBLEM
    int8_t* first_output_ptr_a = new int8_t[first_a_out_size];


    const int* output_lengths_a = new const int[1]{ first_a_out_size };
    
    //TEMPORARY
    //assert(input_size == 8 * 50);
    //assert(first_a_out_size == 56 * 50);
    

    bool success = m_model[0]->predict(input_ptr, input_lengths_a, first_output_ptr_a, output_lengths_a);
 
    uint64_t duration_first_a = esp_timer_get_time() - startTime_first_a;

    if (success) {
        float durationInMs = duration_first_a / 1000;
        ESP_LOGI("MASTERHandle", "Inference For First A Model took: %lld micro seconds, %0.4f ms",
            duration_first_a, durationInMs);
            
        
        if(report_buffer != nullptr)
        {
            report_size = strlen(report_buffer);
            snprintf(report_buffer + report_size, size - report_size,
                 "0_Model Infa: %lld \xCE\xBCs, %0.2f ms\n", duration_first_a, durationInMs);
        }
        
    } else {
        ESP_LOGE("MASTERHandle", "First model inference failed");
    }
    m_model[0]->getTotalProfileTimePerOp();
    m_model[0]->ClearProfiler();

    float first_a_scale = m_model[0]->getOutputScale(0);
    int32_t first_a_shift = m_model[0]->getOutputZeroPoint(0);
    
    float first_b_scale = m_model[1]->getInputScale(0);
    int32_t first_b_shift = m_model[1]->getInputZeroPoint(0);

    // Do the ELU LUT layer In-Place
    uint64_t startTime_ELU_A = esp_timer_get_time();
    build_elu_lut(first_a_scale, first_a_shift, first_b_scale, first_b_shift, 1.0);
    apply_elu_lut(first_output_ptr_a, first_a_out_size);
    uint64_t duration_ELU_A = esp_timer_get_time() - startTime_ELU_A;

    if(report_buffer != nullptr)
    {
        report_size = strlen(report_buffer);
        float durationInMs = duration_ELU_A / 1000;
        ESP_LOGI("MASTERHandle", "Inference For first ELU took: %lld micro seconds, %0.4f ms",
            duration_ELU_A, durationInMs);
        snprintf(report_buffer + report_size, size - report_size,
                "0_Model InfELUa: %lld \xCE\xBCs, %0.2f ms\n", duration_ELU_A, durationInMs);
    }
        
    uint64_t startTime_first_b = esp_timer_get_time();
 
    const int* input_lengths_b = new const int[1]{ first_a_out_size };
    
    int8_t* first_output_ptr_b = new int8_t[first_b_out_size];
    const int* output_lengths_b = new const int[1]{ first_b_out_size };
 
    success = m_model[1]->predict(first_output_ptr_a, input_lengths_b, first_output_ptr_b, output_lengths_b);
 
    uint64_t duration_first_b = esp_timer_get_time() - startTime_first_b;
    
    if (success) {
        float durationInMs = duration_first_b / 1000;
        ESP_LOGI("MASTERHandle", "Inference For First B Model took: %lld micro seconds, %0.4f ms",
            duration_first_b, durationInMs);
            
        
        if(report_buffer != nullptr)
        {
            report_size = strlen(report_buffer);
            snprintf(report_buffer + report_size, size - report_size,
                 "0_Model Infb: %lld \xCE\xBCs, %0.2f ms\n", duration_first_b, durationInMs);
        }
        
    } else {
        ESP_LOGE("MASTERHandle", "First model inference failed");
    }

    m_model[1]->getTotalProfileTimePerOp();
    m_model[1]->ClearProfiler();
    
    first_b_scale = m_model[1]->getOutputScale(0);
    first_b_shift = m_model[1]->getOutputZeroPoint(0);
    float* first_output_ptr_b_after_elu = new float[first_b_out_size];
    
    // Do the ELU LUT layer In-Place
    uint64_t startTime_ELU_B = esp_timer_get_time();

    apply_elu_lut_int_float(first_output_ptr_b, first_b_out_size,
    first_b_scale, first_b_shift, 1.0, first_output_ptr_b_after_elu);

    uint64_t duration_ELU_B = esp_timer_get_time() - startTime_ELU_B;

    if(report_buffer != nullptr)
    {
        report_size = strlen(report_buffer);
        float durationInMs = duration_ELU_B / 1000;

        ESP_LOGI("MASTERHandle", "Inference For second ELU took: %lld micro seconds, %0.4f ms",
            duration_ELU_B, durationInMs);

        snprintf(report_buffer + report_size, size - report_size,
                "0_Model InfELUb: %lld \xCE\xBCs, %0.2f ms\n", duration_ELU_B, durationInMs);

    }

    // SECOND MODEL =====================================================
    uint64_t startTime_second = esp_timer_get_time();
 
    // Inputs:  [ x (56*step) | h (2*56) | c (2*56) ]
    float* second_input_ptr = new float[intermediate_size + 2 * state_size]{0};
    const int* second_input_lengths = new const int[3]{
        intermediate_size,
        state_size,
        state_size
    };
 
    // Outputs: [ y (out_ch*step) | h (2*56) | c (2*56) ]
    const int y_chunk = CONFIG_OUTPUT_CHANNELS * step;
    float* intermediate_output_ptr = new float[y_chunk + 2 * state_size]{0};
    const int* intermediate_output_lengths = new const int[3]{
        y_chunk,
        state_size,
        state_size
    };

    bool reported_single_step = false;
 
    for (int t = 0; t < out_len; t += step) {
        uint64_t intermediateTime_second = esp_timer_get_time();
 
        // x slice: channel-major (56, step) taken from first model's
        // channel-major (56, T) output -- stride by T, walk only T_eff
        for (int j = 0; j < feature_ch; j++) {
            for (int s = 0; s < step; s++) {
                second_input_ptr[j * step + s] = first_output_ptr_b_after_elu[j * first_out_len + t + s];
            }
        }
        
        // carry h and c forward from the previous call (zeros on first call)
        if (t > 0) {
            for (int j = 0; j < state_size; j++) {
                second_input_ptr[intermediate_size + j] =
                    intermediate_output_ptr[y_chunk + j];                      // h
                second_input_ptr[intermediate_size + state_size + j] =
                    intermediate_output_ptr[y_chunk + state_size + j];         // c
            }
        }
 
        success = m_model[2]->predict(second_input_ptr, second_input_lengths,
                                      intermediate_output_ptr, intermediate_output_lengths);
        
        // scatter y chunk into the final channel-major output buffer
        for (int j = 0; j < CONFIG_OUTPUT_CHANNELS; j++) {
            for (int s = 0; s < step; s++) {
                output_ptr[j * out_len + t + s] = intermediate_output_ptr[j * step + s];
                //printf("FIRST DUAL (%d)[%0.4f]\n", j, intermediate_output_ptr[j * step + s]);
            }
        }
 
        uint64_t duration_intermediate = esp_timer_get_time() - intermediateTime_second;
        if (success) {
            // report the per-call latency once, using the 2nd call (steady state)
            if (report_buffer != nullptr && t > 0 && !reported_single_step) {
                float durationInMs = duration_intermediate / 1000;
                report_size = strlen(report_buffer);
                snprintf(report_buffer + report_size, size - report_size,
                         "1_Model tInf: %lld \xCE\xBCs, %0.2f ms\n",
                         duration_intermediate, durationInMs);
                reported_single_step = true;
            }
        } else {
            ESP_LOGE("MASTERHandle", "Second model inference failed @ time_step: %d", t);
        }

        m_model[2]->ClearProfiler();
    }
    
 
    uint64_t duration_second = esp_timer_get_time() - startTime_second;
 
    if (success) {
        if(report_buffer != nullptr)
        {
            float durationInMs = duration_second / 1000;
            report_size = strlen(report_buffer);
            snprintf(report_buffer + report_size, size - report_size,
                 "1_Model Inf: %lld \xCE\xBCs, %0.2f ms\n", duration_second, durationInMs);
        }
        
    } else {
        ESP_LOGE("MASTERHandle", "Second model inference failed");
    }
 
    if(report_buffer != nullptr)
    {
        float durationInMs = (duration_second + duration_first_a + duration_first_b) / 1000;
        report_size = strlen(report_buffer);
        snprintf(report_buffer + report_size, size - report_size,
             "Model Inf: %lld \xCE\xBCs, %0.2f ms\n",
             duration_second + duration_first_a + duration_first_b, durationInMs);
    }
    

    delete[] input_lengths_a;
    delete[] first_output_ptr_a;
    delete[] output_lengths_a;

    delete[] input_lengths_b;
    delete[] first_output_ptr_b;
    delete[] output_lengths_b;
    delete[] first_output_ptr_b_after_elu;

    delete[] second_input_ptr;
    delete[] second_input_lengths;
    delete[] intermediate_output_ptr;
    delete[] intermediate_output_lengths;
 
    ESP_LOGI("MASTERHandle", "CLEARED ALL ARRAYS");
}