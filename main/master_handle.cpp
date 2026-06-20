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

void MasterHandle::init_models(int model_1_index, int model_2_index,int model_3_index, int model_4_index,
    bool usePSRAM, char* report_buffer, int size)
{
    this->usePSRAM = usePSRAM;
    this->m_model = new Model*[4];
    init_model(model_1_index, 0 ,report_buffer, size);
    init_model(model_2_index, 1 ,report_buffer, size);
    init_model(model_3_index, 2 ,report_buffer, size);
    init_model(model_4_index, 3 ,report_buffer, size);

    // Build ELU LUTs
    //build_elu_lut(m_model[0]->getInputScale())
}


double MasterHandle::print_output(const float* output_window, int output_size, const float* correct_window)
{
    //printf("OUTPUT DATA ================= \n");

    // Accumulate in double. The per-element squared error can be ~1e-8 or
    // smaller; summing many of those in float32 loses the low-order bits
    // (and the running sum can stall once it dwarfs each new term).
    double sse = 0.0;
    for (int j = 0; j < output_size; j++)
    {
        double diff = (double)output_window[j] - (double)correct_window[j];
        sse += diff * diff;

        // Show enough digits to actually see the residual: scientific notation
        // so small mismatches are never rounded away on screen.
        // printf("(%d) out=% .6e  ref=% .6e  diff=% .3e\n",
        //        j, output_window[j], correct_window[j], diff);
    }
    //printf("==============================\n");

    double mse = sse / (double)output_size;

    // Print with scientific notation so small values are never shown as 0.0000.
    printf("MSE: %.6e  (%.8f)\n", mse, mse);

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

        delete m_model[3];
        m_model[3] = nullptr;
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

        if(this->usePSRAM)
            heap_caps_free(m_psram_model_ptr[3]);
        else
            free(m_psram_model_ptr[3]);

        m_psram_model_ptr[3] = nullptr;

        ESP_LOGI("MASTERhandle", "4 Models Cleared Successfully");

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


void MasterHandle::apply_elu_float(float* in, int n, float alpha)
{
   for (int i = 0; i < n; ++i) {
        in[i] = (in[i] >= 0.0f) ? in[i] : alpha * (expf(in[i]) - 1.0f);
    }
}

// ============================================================
// STAGE 1: First Block A
// ============================================================

// int8 variant: model emits int8, ELU is the int->float dequant path.
void MasterHandle::stage1_first_a_int8(const float* input_ptr, int input_size,
                                       float* out_f32, int first_a_out_size,
                                       WindowStageTimes* wt)
{
    const int* input_lengths_a  = new const int[1]{ input_size };
    int8_t*    first_output_ptr_a = new int8_t[first_a_out_size];
    const int* output_lengths_a = new const int[1]{ first_a_out_size };

    bool success = m_model[0]->predict(input_ptr, input_lengths_a, first_output_ptr_a, output_lengths_a);

    // read profiler ticks BEFORE clearing
    wt->infa_ticks = m_model[0]->getTotalTicks();
    if (!success) {
        ESP_LOGE("MASTERHandle", "First model inference failed");
    }
    m_model[0]->ClearProfiler();

    // ELU via int->float dequant path (alpha = 1.0), writing float result.
    float   first_a_scale = m_model[0]->getOutputScale(0);
    int32_t first_a_shift = m_model[0]->getOutputZeroPoint(0);

    uint64_t startTime_ELU_A = esp_timer_get_time();
    apply_elu_lut_int_float(first_output_ptr_a, first_a_out_size,
                            first_a_scale, first_a_shift, 1.0, out_f32);

    wt->elua_us = esp_timer_get_time() - startTime_ELU_A;

    delete[] input_lengths_a;
    delete[] first_output_ptr_a;
    delete[] output_lengths_a;
}

// float32 variant: model emits float, ELU is the plain float in-place path.
void MasterHandle::stage1_first_a_f32(const float* input_ptr, int input_size,
                                      float* out_f32, int first_a_out_size,
                                      WindowStageTimes* wt)
{
    const int* input_lengths_a  = new const int[1]{ input_size };
    float*     first_output_ptr_a = new float[first_a_out_size];
    const int* output_lengths_a = new const int[1]{ first_a_out_size };

    bool success = m_model[0]->predict(input_ptr, input_lengths_a, first_output_ptr_a, output_lengths_a);

    // read profiler ticks BEFORE clearing
    wt->infa_ticks = m_model[0]->getTotalTicks();
    if (!success) {
        ESP_LOGE("MASTERHandle", "First model inference failed");
    }
    m_model[0]->ClearProfiler();

    // ELU plain float (alpha = 1.0). Copy into out_f32 then activate.
    uint64_t startTime_ELU_A = esp_timer_get_time();
    for (int i = 0; i < first_a_out_size; ++i) out_f32[i] = first_output_ptr_a[i];
    apply_elu_float(out_f32, first_a_out_size, 1.0);
    wt->elua_us = esp_timer_get_time() - startTime_ELU_A;

    delete[] input_lengths_a;
    delete[] first_output_ptr_a;
    delete[] output_lengths_a;
}

// Concat (shared): [ first_a_after_elu (first_a_out_size) | input (input_size) ]
void MasterHandle::concat_with_input(const float* first_a_after_elu, int first_a_out_size,
                                     const float* input_ptr, int input_size,
                                     float* concat_out)
{
    for (int i = 0; i < first_a_out_size; i++)
        concat_out[i] = first_a_after_elu[i];
    for (int i = 0; i < input_size; i++)
        concat_out[i + first_a_out_size] = input_ptr[i];
}

// ============================================================
// STAGE 2: First Block B
// ============================================================

// float32 variant: model emits float, plain float ELU in-place.
void MasterHandle::stage2_first_b_f32(const float* concat_in, int concat_in_size,
                                      float* out_f32, int first_b_out_size,
                                      WindowStageTimes* wt)
{
    const int* input_lengths_b  = new const int[1]{ concat_in_size };
    const int* output_lengths_b = new const int[1]{ first_b_out_size };

    bool success = m_model[1]->predict(concat_in, input_lengths_b, out_f32, output_lengths_b);

    // read profiler ticks BEFORE clearing
    wt->infb_ticks = m_model[1]->getTotalTicks();
    if (!success) {
        ESP_LOGE("MASTERHandle", "First model inference failed");
    }
    m_model[1]->ClearProfiler();

    uint64_t startTime_ELU_B = esp_timer_get_time();
    apply_elu_float(out_f32, first_b_out_size, 1.0);
    wt->elub_us = esp_timer_get_time() - startTime_ELU_B;

    delete[] input_lengths_b;
    delete[] output_lengths_b;
}

// int8 variant: model emits int8, ELU via int->float dequant path into out_f32.
void MasterHandle::stage2_first_b_int8(const float* concat_in, int concat_in_size,
                                       float* out_f32, int first_b_out_size,
                                       WindowStageTimes* wt)
{
    const int* input_lengths_b  = new const int[1]{ concat_in_size };
    int8_t*    first_output_ptr_b = new int8_t[first_b_out_size];
    const int* output_lengths_b = new const int[1]{ first_b_out_size };

    bool success = m_model[1]->predict(concat_in, input_lengths_b, first_output_ptr_b, output_lengths_b);

    // read profiler ticks BEFORE clearing
    wt->infb_ticks = m_model[1]->getTotalTicks();
    if (!success) {
        ESP_LOGE("MASTERHandle", "First model inference failed");
    }
    m_model[1]->ClearProfiler();

    float   first_b_scale = m_model[1]->getOutputScale(0);
    int32_t first_b_shift = m_model[1]->getOutputZeroPoint(0);

    uint64_t startTime_ELU_B = esp_timer_get_time();
    apply_elu_lut_int_float(first_output_ptr_b, first_b_out_size,
                            first_b_scale, first_b_shift, 1.0, out_f32);
    wt->elub_us = esp_timer_get_time() - startTime_ELU_B;

    delete[] input_lengths_b;
    delete[] first_output_ptr_b;
    delete[] output_lengths_b;
}

// ============================================================
// STAGE 3: LSTM step (model[2])
// ============================================================
bool MasterHandle::lstm_step_f32(float* second_input_ptr, const int* second_input_lengths,
                                 float* lstm_out, const int* lstm_out_lengths)
{
    return m_model[2]->predict(second_input_ptr, second_input_lengths,
                               lstm_out, lstm_out_lengths);
}

bool MasterHandle::lstm_step_int8(float* second_input_ptr, const int* second_input_lengths,
                                  float* lstm_out, const int* lstm_out_lengths)
{
    // Same external float interface; the int8 model handles quant internally
    // through predict(). Hook for any int8-specific pre/post-processing.
    return m_model[2]->predict(second_input_ptr, second_input_lengths,
                               lstm_out, lstm_out_lengths);
}

// ============================================================
// STAGE 4: Regressor head step (model[3])
// ============================================================
bool MasterHandle::regressor_step_f32(float* lstm_out, const int* lstm_out_lengths,
                                      float* reg_out, const int* reg_out_lengths)
{
    return m_model[3]->predict(lstm_out, lstm_out_lengths,
                               reg_out, reg_out_lengths);
}

bool MasterHandle::regressor_step_int8(float* lstm_out, const int* lstm_out_lengths,
                                       float* reg_out, const int* reg_out_lengths)
{
    return m_model[3]->predict(lstm_out, lstm_out_lengths,
                               reg_out, reg_out_lengths);
}

// ============================================================
// Recurrent loop orchestrator (dtype-agnostic via member fn ptrs)
//   - pushes per-step ticks (lstm/regressor) into the trial accumulator
// ============================================================
void MasterHandle::recurrent_loop(const float* first_b_out, int first_out_len,
                                  float* output_ptr, int out_len, int step,
                                  float* second_input_ptr, const int* second_input_lengths,
                                  int intermediate_size, int state_size,
                                  StepFn lstm_step, StepFn reg_step,
                                  TrialStats* trial)
{
    const int y_chunk    = CONFIG_OUTPUT_CHANNELS * step;
    const int lstm_chunk = feature_ch * step;

    float* intermediate_A_output_ptr = new float[lstm_chunk + 2 * state_size]{0};
    const int* intermediate_A_output_lengths = new const int[3]{
            lstm_chunk,
            state_size,
            state_size
        };
    float* intermediate_output_ptr = new float[y_chunk * step]{0};
    const int* intermediate_output_lengths = new const int[1]{
        y_chunk * step
    };

    bool success = false;

    //printf("Check Second Model Health: %d", m_model[2]->getInputType(0));
    for (int t = 0; t < out_len; t += step) {

        // x slice: channel-major (56, step) from first model's (56, T) output
        for (int j = 0; j < feature_ch; j++) {
            for (int s = 0; s < step; s++) {
                second_input_ptr[j * step + s] = first_b_out[j * first_out_len + t + s];
            }
        }
        // carry h and c forward from the previous call (zeros on first call)
        if (t > 0) {
            for (int j = 0; j < state_size; j++) {
                second_input_ptr[intermediate_size + j] =
                    intermediate_A_output_ptr[lstm_chunk + j];                      // h
                second_input_ptr[intermediate_size + state_size + j] =
                    intermediate_A_output_ptr[lstm_chunk + state_size + j];         // c
            }
        }

        // do the lstm prediction (selected dtype variant)
        success = (this->*lstm_step)(second_input_ptr, second_input_lengths,
                                     intermediate_A_output_ptr, intermediate_A_output_lengths);
        int32_t lstm_ticks = m_model[2]->getTotalTicks();    // before ClearProfiler

        // regressor head trims h/c internally and deals with the x part
        bool success_2 = (this->*reg_step)(intermediate_A_output_ptr, intermediate_A_output_lengths,
                                           intermediate_output_ptr, intermediate_output_lengths);
        int32_t reg_ticks = m_model[3]->getTotalTicks();     // before ClearProfiler

        // scatter y chunk into the final channel-major output buffer
        for (int j = 0; j < CONFIG_OUTPUT_CHANNELS; j++) {
            for (int s = 0; s < step; s++) {
                output_ptr[j * out_len + t + s] = intermediate_output_ptr[j * step + s];
            }
        }

        if (success && success_2) {
            trial->add_lstm_step(lstm_ticks);
            trial->add_reg_step(reg_ticks);
        } else {
            ESP_LOGE("MASTERHandle", "Second model inference failed @ time_step: %d", t);
        }

        m_model[2]->ClearProfiler();
        m_model[3]->ClearProfiler();
    }

    for (int j = 0; j < state_size; j++) {
        second_input_ptr[intermediate_size + j] =
            intermediate_A_output_ptr[lstm_chunk + j];                      // h
        second_input_ptr[intermediate_size + state_size + j] =
            intermediate_A_output_ptr[lstm_chunk + state_size + j];         // c
    }

    delete[] intermediate_A_output_ptr;
    delete[] intermediate_A_output_lengths;
    delete[] intermediate_output_ptr;
    delete[] intermediate_output_lengths;
}

// ============================================================
// Top-level orchestrator (runtime-configurable stages)
//   - reports timing through wt (per-window) and trial (per-step)
// ============================================================
void MasterHandle::dual_inference(const QuantConfig& qc,
                                  const float* input_ptr, int input_size,
                                  float* output_ptr, int output_size,
                                  float* second_input_ptr, const int* second_input_lengths,
                                  int intermediate_size, int first_out_len,
                                  WindowStageTimes* wt, TrialStats* trial)
{
    assert(m_model[0] != nullptr); // first a model
    assert(m_model[1] != nullptr); // first b model
    assert(m_model[2] != nullptr); // lstm
    assert(m_model[3] != nullptr); // regressor

    const int state_size       = 2 * feature_ch;             // (2, 1, 56) flattened
    const int step             = intermediate_size / feature_ch;
    const int out_len          = output_size / CONFIG_OUTPUT_CHANNELS;   // T_eff
    const int in_len           = input_size / CONFIG_INPUT_CHANNELS;

    const int first_a_out_size        = 48 * in_len;          // 48 * window_len, before CONCAT
    const int concat_first_a_out_size = feature_ch * in_len;
    const int first_b_out_size        = feature_ch * first_out_len;       // 56 * T

    assert(step >= 1 && intermediate_size % feature_ch == 0);
    assert(output_size % CONFIG_OUTPUT_CHANNELS == 0);
    assert(out_len % step == 0);              // second model runs out_len/step times
    assert(out_len <= first_out_len);         // T_eff = (T/step)*step <= T
    assert(first_out_len - out_len < step);   // only a ragged tail may be skipped

    uint64_t win_start = esp_timer_get_time();

    // STAGE 1: First Block A (+ ELU) -> float result. Variant chosen by qc.first_a.
    //          fills wt->infa_ticks, wt->elua_us
    float* first_a_after_elu = new float[first_a_out_size];
    (this->*(qc.first_a))(input_ptr, input_size,
                          first_a_after_elu, first_a_out_size, wt);

    // CONCAT with original input
    float* concat_first_output_ptr_a = new float[concat_first_a_out_size];
    concat_with_input(first_a_after_elu, first_a_out_size,
                      input_ptr, input_size, concat_first_output_ptr_a);

    // STAGE 2: First Block B (+ ELU) -> float result. Variant chosen by qc.first_b.
    //          fills wt->infb_ticks, wt->elub_us
    float* first_output_ptr_b = new float[first_b_out_size];
    (this->*(qc.first_b))(concat_first_output_ptr_a, concat_first_a_out_size,
                          first_output_ptr_b, first_b_out_size, wt);

    // STAGE 3+4: LSTM + Regressor loop. Variants chosen by qc.lstm_step / qc.reg_step.
    //            pushes per-step ticks into trial
    recurrent_loop(first_output_ptr_b, first_out_len,
                   output_ptr, out_len, step,
                   second_input_ptr, second_input_lengths,
                   intermediate_size, state_size,
                   qc.lstm_step, qc.reg_step, trial);

    wt->window_us = esp_timer_get_time() - win_start;

    delete[] first_a_after_elu;
    delete[] concat_first_output_ptr_a;
    delete[] first_output_ptr_b;
    //ESP_LOGI("MASTERHandle", "CLEARED ALL ARRAYS");
}
// void MasterHandle::dual_inference(const float* input_ptr, int input_size,
//                                   float* output_ptr, int output_size,
//                                   float* second_input_ptr, const int* second_input_lengths,
//                                   int intermediate_size, int first_out_len,
//                                   char* report_buffer, int size)
// {
//     assert(m_model[0] != nullptr); // first a model
//     assert(m_model[1] != nullptr); // first b model
//     assert(m_model[2] != nullptr); // lstm
//     assert(m_model[3] != nullptr); // regressor
 
//     const int state_size       = 2 * feature_ch;             // (2, 1, 56) flattened
//     const int step             = intermediate_size / feature_ch;
//     const int out_len          = output_size / CONFIG_OUTPUT_CHANNELS;   // T_eff
//     const int in_len           = input_size / CONFIG_INPUT_CHANNELS;
 
//     const int first_a_out_size        = 48 * in_len;          // 48 * window_len, before CONCAT
//     const int concat_first_a_out_size = feature_ch * in_len;
//     const int first_b_out_size        = feature_ch * first_out_len;       // 56 * T
 
//     assert(step >= 1 && intermediate_size % feature_ch == 0);
//     assert(output_size % CONFIG_OUTPUT_CHANNELS == 0);
//     assert(out_len % step == 0);
//     assert(out_len <= first_out_len);
//     assert(first_out_len - out_len < step);
 
//     // ---- timing wrappers around stages, to reproduce the grand-total line ----
//     uint64_t t0, t1;
 
//     // STAGE 1: First Block A  (+ ELU)  -> float result
//     float* first_a_after_elu = new float[first_a_out_size];
//     t0 = esp_timer_get_time();
//     // >>> SWAP THIS LINE to flip model[0] between int8 / float32 <<<
//     stage1_first_a_int8(input_ptr, input_size, first_a_after_elu, first_a_out_size,
//                         report_buffer, size);
//     t1 = esp_timer_get_time();
//     uint64_t duration_first_a = t1 - t0;
 
//     // CONCAT with original input
//     float* concat_first_output_ptr_a = new float[concat_first_a_out_size];
//     concat_with_input(first_a_after_elu, first_a_out_size,
//                       input_ptr, input_size, concat_first_output_ptr_a);
 
//     // STAGE 2: First Block B  (+ ELU)  -> float result
//     float* first_output_ptr_b = new float[first_b_out_size];
//     t0 = esp_timer_get_time();
//     // >>> SWAP THIS LINE to flip model[1] between int8 / float32 <<<
//     stage2_first_b_int8(concat_first_output_ptr_a, concat_first_a_out_size,
//                        first_output_ptr_b, first_b_out_size,
//                        report_buffer, size);
//     t1 = esp_timer_get_time();
//     uint64_t duration_first_b = t1 - t0;
 
//     // STAGE 3+4: LSTM + Regressor loop
//     t0 = esp_timer_get_time();
//     // >>> SWAP THESE to flip model[2] / model[3] between int8 / float32 <<<
//     recurrent_loop(first_output_ptr_b, first_out_len,
//                    output_ptr, out_len, step,
//                    second_input_ptr, second_input_lengths,
//                    intermediate_size, state_size,
//                    &MasterHandle::lstm_step_f32,
//                    &MasterHandle::regressor_step_int8,
//                    report_buffer, size);
//     t1 = esp_timer_get_time();
//     uint64_t duration_second = t1 - t0;
 
//     // Grand total line (same as before: second + first_a + first_b)
//     if (report_buffer != nullptr) {
//         uint64_t total = duration_second + duration_first_a + duration_first_b;
//         float durationInMs = total / 1000;
//         int report_size = strlen(report_buffer);
//         snprintf(report_buffer + report_size, size - report_size,
//                  "Model Inf: %lld \xCE\xBCs, %0.2f ms\n", total, durationInMs);
//     }
 
//     delete[] first_a_after_elu;
//     delete[] concat_first_output_ptr_a;
//     delete[] first_output_ptr_b;
//     ESP_LOGI("MASTERHandle", "CLEARED ALL ARRAYS");

// }
// void MasterHandle::dual_inference(const float* input_ptr, int input_size,
//                                   float* output_ptr, int output_size,
//                                   float* second_input_ptr, const int* second_input_lengths,
//                                   int intermediate_size, int first_out_len,
//                                   char* report_buffer, int size)
// {

//     int report_size = 0;
//     assert(m_model[0] != nullptr); // Ensure the first a model is initialized
//     assert(m_model[1] != nullptr); // Ensure the first b model is initialized
//     assert(m_model[2] != nullptr); // Ensure the second model is initialized


//     const int state_size       = 2 * feature_ch;             // (2, 1, 56) flattened
//     const int step             = intermediate_size / feature_ch;
//     const int out_len          = output_size / CONFIG_OUTPUT_CHANNELS;   // T_eff
//     const int in_len           = input_size / CONFIG_INPUT_CHANNELS;
    
//     const int first_a_out_size = 48 * in_len;          // 48 * window_len (48, 50) Before CONCAT
//     const int concat_first_a_out_size = feature_ch * in_len;
//     const int first_b_out_size = feature_ch * first_out_len;          // 56 * T

//     assert(step >= 1 && intermediate_size % feature_ch == 0);
//     assert(output_size % CONFIG_OUTPUT_CHANNELS == 0);
//     assert(out_len % step == 0);              // second model runs out_len/step times
//     assert(out_len <= first_out_len);         // T_eff = (T/step)*step <= T
//     assert(first_out_len - out_len < step);   // only a ragged tail may be skipped
 
//     // ESP_LOGI("MASTERHandle", "dual_inference: in=%d, out=%d, inter=%d (step=%d, out_len=%d, T=%d)",
//     //          input_size, output_size, intermediate_size, step, out_len, first_out_len);
    
//     uint64_t startTime_first_a = esp_timer_get_time();
 
//     const int* input_lengths_a = new const int[1]{ input_size }; // 8 * win_len
    
//     //float* first_output_ptr_a = new float[first_a_out_size]; 
//     int8_t* first_output_ptr_a = new int8_t[first_a_out_size]; 
//     const int* output_lengths_a = new const int[1]{ first_a_out_size };
    
//     bool success = m_model[0]->predict(input_ptr, input_lengths_a, first_output_ptr_a, output_lengths_a);
    

//     // for(int i =0; i < input_size; i++)
//     // {
//     //     printf("firstA input(%d): %0.4f\n", i, input_ptr[i]);
//     // }
//     uint64_t duration_first_a = esp_timer_get_time() - startTime_first_a;

//     if (success) {
//         float durationInMs = duration_first_a / 1000;
//         // ESP_LOGI("MASTERHandle", "Inference For First A Model took: %lld micro seconds, %0.4f ms",
//         //     duration_first_a, durationInMs);
            
        
//         if(report_buffer != nullptr)
//         {
//             report_size = strlen(report_buffer);
//             snprintf(report_buffer + report_size, size - report_size,
//                  "0_Model Infa: %lld \xCE\xBCs, %0.2f ms\n", duration_first_a, durationInMs);
//         }
        
//     } else {
//         ESP_LOGE("MASTERHandle", "First model inference failed");
//     }
//     //m_model[0]->getTotalProfileTimePerOp();
//     m_model[0]->ClearProfiler();

//     // Do the ELU LUT layer In-Place
//     float first_a_scale = m_model[0]->getOutputScale(0);
//     int32_t first_a_shift = m_model[0]->getOutputZeroPoint(0);
//     float* first_output_ptr_a_after_elu = new float[first_a_out_size];
    
//     // Do the ELU LUT layer In-Place, We are trying an int8 first conv this time only
//     uint64_t startTime_ELU_A = esp_timer_get_time();
//     apply_elu_lut_int_float(first_output_ptr_a, first_a_out_size,
//     first_a_scale, first_a_shift, 1.0, first_output_ptr_a_after_elu);
//     uint64_t duration_ELU_A = esp_timer_get_time() - startTime_ELU_A;

//     // uint64_t startTime_ELU_A = esp_timer_get_time();
//     // apply_elu_float(first_output_ptr_a, first_a_out_size, 1.0);
//     // uint64_t duration_ELU_A = esp_timer_get_time() - startTime_ELU_A;

//     // DO CONCATENATION WITH ORIGINAL X
//     float* concat_first_output_ptr_a = new float[concat_first_a_out_size];
//     for(int i =0; i < first_a_out_size; i++)
//     {
//         // CONCAT the output of First_BlockA
//         concat_first_output_ptr_a[i] = first_output_ptr_a_after_elu[i];
//     }

//     for(int i =0; i < input_size; i++)
//     {
//         // CONCAT the output of First_BlockA
//         concat_first_output_ptr_a[i + first_a_out_size] = input_ptr[i];
//     }

//     if(report_buffer != nullptr)
//     {
//         report_size = strlen(report_buffer);
//         float durationInMs = duration_ELU_A / 1000;
//         // ESP_LOGI("MASTERHandle", "Inference For first ELU took: %lld micro seconds, %0.4f ms",
//         //     duration_ELU_A, durationInMs);
//         snprintf(report_buffer + report_size, size - report_size,
//                 "0_Model InfELUa: %lld \xCE\xBCs, %0.2f ms\n", duration_ELU_A, durationInMs);
//     }
        
//     uint64_t startTime_first_b = esp_timer_get_time();
 
//     const int* input_lengths_b = new const int[1]{ concat_first_a_out_size };
    
//     float* first_output_ptr_b = new float[first_b_out_size];
//     const int* output_lengths_b = new const int[1]{ first_b_out_size };
 
//     success = m_model[1]->predict(concat_first_output_ptr_a, input_lengths_b, first_output_ptr_b, output_lengths_b);
 
//     uint64_t duration_first_b = esp_timer_get_time() - startTime_first_b;
    
//     if (success) {
//         float durationInMs = duration_first_b / 1000;
//         // ESP_LOGI("MASTERHandle", "Inference For First B Model took: %lld micro seconds, %0.4f ms",
//         //     duration_first_b, durationInMs);
            
        
//         if(report_buffer != nullptr)
//         {
//             report_size = strlen(report_buffer);
//             snprintf(report_buffer + report_size, size - report_size,
//                  "0_Model Infb: %lld \xCE\xBCs, %0.2f ms\n", duration_first_b, durationInMs);
//         }
        
//     } else {
//         ESP_LOGE("MASTERHandle", "First model inference failed");
//     }

//     //m_model[1]->getTotalProfileTimePerOp();
//     m_model[1]->ClearProfiler();
    
//     // Do the ELU LUT layer In-Place
//     uint64_t startTime_ELU_B = esp_timer_get_time();

//     apply_elu_float(first_output_ptr_b, first_b_out_size, 1.0);

//     uint64_t duration_ELU_B = esp_timer_get_time() - startTime_ELU_B;

//     if(report_buffer != nullptr)
//     {
//         report_size = strlen(report_buffer);
//         float durationInMs = duration_ELU_B / 1000;

//         // ESP_LOGI("MASTERHandle", "Inference For second ELU took: %lld micro seconds, %0.4f ms",
//         //     duration_ELU_B, durationInMs);

//         snprintf(report_buffer + report_size, size - report_size,
//                 "0_Model InfELUb: %lld \xCE\xBCs, %0.2f ms\n", duration_ELU_B, durationInMs);

//     }

//     // Second Model A: takes in 3 inputs, emits 3 outputs (x,h,c)
//     // Second Model B: takes in 1 input, emit 1 output x -> y
//     // We need to carry over the h,c data
//     uint64_t startTime_second = esp_timer_get_time();   
 
//     // Outputs: [ y (out_ch*step) | h (2*56) | c (2*56) ]

//     float* intermediate_input_ptr = new float[intermediate_size + 2 * state_size]{0};

    
//     const int y_chunk = CONFIG_OUTPUT_CHANNELS * step;
//     const int lstm_chunk = feature_ch * step;

//     float* intermediate_A_output_ptr = new float[lstm_chunk + 2 * state_size]{0};
//     const int* intermediate_A_output_lengths = new const int[3]{
//             lstm_chunk,
//             state_size,
//             state_size
//         };
//     float* intermediate_output_ptr = new float[y_chunk * step]{0};
//     const int* intermediate_output_lengths = new const int[1]{
//         y_chunk * step
//     };

//     bool reported_single_step = false;
 
//     printf("Check Second Model Health: %d", m_model[2]->getInputType(0));
//     for (int t = 0; t < out_len; t += step) {
//         uint64_t intermediateTime_second = esp_timer_get_time();
 
//         // x slice: channel-major (56, step) taken from first model's
//         // channel-major (56, T) output -- stride by T, walk only T_eff
//         for (int j = 0; j < feature_ch; j++) {
//             for (int s = 0; s < step; s++) {
//                 second_input_ptr[j * step + s] = first_output_ptr_b[j * first_out_len + t + s];
//             }
//         }
//         // for (int j =0; j < second_input_lengths[0]; j++)
//         // {
//         //     printf("second_input_ptr(%d)[%0.4f]\n", j, second_input_ptr[j]);
//         // }
//         // carry h and c forward from the previous call (zeros on first call)
//         if (t > 0) {
//             for (int j = 0; j < state_size; j++) {
//                 second_input_ptr[intermediate_size + j] =
//                     intermediate_A_output_ptr[lstm_chunk + j];                      // h
//                 second_input_ptr[intermediate_size + state_size + j] =
//                     intermediate_A_output_ptr[lstm_chunk + state_size + j];         // c
//             }
//         }
        
//         // do the lstm prediction
//         success = m_model[2]->predict(second_input_ptr, second_input_lengths,
//                                       intermediate_A_output_ptr, intermediate_A_output_lengths);

//         // for (int j =0; j <  intermediate_A_output_lengths[0]; j++)
//         // {
//         //     printf("intermediate_A_output_ptr(%d)[%0.4f]\n", j, intermediate_A_output_ptr[j]);
//         // }
//         // the model predict code will already trim the h and c parts and only deal with the first part containing x
//         bool success_2 = m_model[3]->predict(intermediate_A_output_ptr, intermediate_A_output_lengths,
//             intermediate_output_ptr, intermediate_output_lengths);

//         // for (int j =0; j <  intermediate_output_lengths[0]; j++)
//         // {
//         //     printf("intermediate_output_ptr(%d)[%0.4f]\n", j, intermediate_output_ptr[j]);
//         // }
//         // scatter y chunk into the final channel-major output buffer
//         for (int j = 0; j < CONFIG_OUTPUT_CHANNELS; j++) {
//             for (int s = 0; s < step; s++) {
//                 output_ptr[j * out_len + t + s] = intermediate_output_ptr[j * step + s];
//                 //printf("FIRST DUAL (%d)[%0.4f]\n", j, intermediate_output_ptr[j * step + s]);
//             }
//         }
 
//         uint64_t duration_intermediate = esp_timer_get_time() - intermediateTime_second;
//         if (success && success_2) {
//             // report the per-call latency once, using the 2nd call (steady state)
//             if (report_buffer != nullptr && t > 0 && !reported_single_step) {
//                 float durationInMs = duration_intermediate / 1000;
//                 report_size = strlen(report_buffer);
//                 snprintf(report_buffer + report_size, size - report_size,
//                          "1_Model tInf: %lld \xCE\xBCs, %0.2f ms\n",
//                          duration_intermediate, durationInMs);
//                 reported_single_step = true;
//             }
//         } else {
//             ESP_LOGE("MASTERHandle", "Second model inference failed @ time_step: %d", t);
//         }
        
//         if(t==1)
//         {
//             //m_model[2]->getTotalProfileTimePerOp();
//         }
//         m_model[2]->ClearProfiler();
//         m_model[3]->ClearProfiler();
//     }
    
//     for (int j = 0; j < state_size; j++) {
//         second_input_ptr[intermediate_size + j] =
//             intermediate_A_output_ptr[lstm_chunk + j];                      // h
//         second_input_ptr[intermediate_size + state_size + j] =
//             intermediate_A_output_ptr[lstm_chunk + state_size + j];         // c
//     }
//     uint64_t duration_second = esp_timer_get_time() - startTime_second;
 
//     if (success) {
//         if(report_buffer != nullptr)
//         {
//             float durationInMs = duration_second / 1000;
//             report_size = strlen(report_buffer);
//             snprintf(report_buffer + report_size, size - report_size,
//                  "1_Model Inf: %lld \xCE\xBCs, %0.2f ms\n", duration_second, durationInMs);
//         }
        
//     } else {
//         ESP_LOGE("MASTERHandle", "Second model inference failed");
//     }
 
//     if(report_buffer != nullptr)
//     {
//         float durationInMs = (duration_second + duration_first_a + duration_first_b) / 1000;
//         report_size = strlen(report_buffer);
//         snprintf(report_buffer + report_size, size - report_size,
//              "Model Inf: %lld \xCE\xBCs, %0.2f ms\n",
//              duration_second + duration_first_a + duration_first_b, durationInMs);
//     }
    

//     delete[] input_lengths_a;
//     delete[] first_output_ptr_a;
//     delete[] output_lengths_a;

//     delete[] input_lengths_b;
//     delete[] first_output_ptr_b;
//     delete[] output_lengths_b;

//     delete[] intermediate_output_ptr;
//     delete[] intermediate_output_lengths;
//     delete[] concat_first_output_ptr_a;
//     ESP_LOGI("MASTERHandle", "CLEARED ALL ARRAYS");
// }