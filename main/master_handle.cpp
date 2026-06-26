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
    build_elu_lut(m_model[0]->getOutputScale(0), m_model[0]->getOutputZeroPoint(0), 
    m_model[1]->getInputScale(0), m_model[1]->getInputZeroPoint(0), 1.0);
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
        printf("(%d) out=% .6lf  ref=% .6lf  diff=% .3lf\n",
               j, output_window[j], correct_window[j], diff);
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

void MasterHandle::init_optim_lstm(const char* partition)
{
    this->m_optim_lstm = new OptimizedNativeLSTM();

    // Allocate data for weights and biases (hidden == x_features == 56)
    m_model_optim_data.m_x_l0_w = new int8_t [LSTM_Q_HIDDEN * LSTM_Q_HIDDEN * 4];
    m_model_optim_data.m_x_l0_b = new int32_t[LSTM_Q_HIDDEN * 4];
    m_model_optim_data.m_h_l0_w = new int8_t [LSTM_Q_HIDDEN * LSTM_Q_HIDDEN * 4];
    m_model_optim_data.m_h_l0_b = new int32_t[LSTM_Q_HIDDEN * 4];
    m_model_optim_data.m_x_l1_w = new int8_t [LSTM_Q_HIDDEN * LSTM_Q_HIDDEN * 4];
    m_model_optim_data.m_x_l1_b = new int32_t[LSTM_Q_HIDDEN * 4];
    m_model_optim_data.m_h_l1_w = new int8_t [LSTM_Q_HIDDEN * LSTM_Q_HIDDEN * 4];
    m_model_optim_data.m_h_l1_b = new int32_t[LSTM_Q_HIDDEN * 4];

    // Map the flash partition and get zero-copy pointers to each block.
    bool ok = this->m_model_flash->allocatePointerOnFlashOptimLSTMWeightsBiases(
        partition,
        &m_model_optim_data_flash.m_x_l0_w,
        &m_model_optim_data_flash.m_x_l0_b,
        &m_model_optim_data_flash.m_h_l0_w,
        &m_model_optim_data_flash.m_h_l0_b,
        &m_model_optim_data_flash.m_x_l1_w,
        &m_model_optim_data_flash.m_x_l1_b,
        &m_model_optim_data_flash.m_h_l1_w,
        &m_model_optim_data_flash.m_h_l1_b
    );

    if (!ok) {
        ESP_LOGE("MASTERHandle", "Failed to map optim LSTM weights from flash");
        return;
    }

    // Copy flash -> owned buffers. Lengths come from the header _COUNT macros
    // (weights are int8 -> COUNT bytes; biases are int32 -> COUNT * 4 bytes).
    memcpy(m_model_optim_data.m_x_l0_w, m_model_optim_data_flash.m_x_l0_w, LSTM_Q_L0_x_W_COUNT);
    memcpy(m_model_optim_data.m_x_l0_b, m_model_optim_data_flash.m_x_l0_b, LSTM_Q_L0_x_B_COUNT * sizeof(int32_t));
    memcpy(m_model_optim_data.m_h_l0_w, m_model_optim_data_flash.m_h_l0_w, LSTM_Q_L0_h_W_COUNT);
    memcpy(m_model_optim_data.m_h_l0_b, m_model_optim_data_flash.m_h_l0_b, LSTM_Q_L0_h_B_COUNT * sizeof(int32_t));
    memcpy(m_model_optim_data.m_x_l1_w, m_model_optim_data_flash.m_x_l1_w, LSTM_Q_L1_x_W_COUNT);
    memcpy(m_model_optim_data.m_x_l1_b, m_model_optim_data_flash.m_x_l1_b, LSTM_Q_L1_x_B_COUNT * sizeof(int32_t));
    memcpy(m_model_optim_data.m_h_l1_w, m_model_optim_data_flash.m_h_l1_w, LSTM_Q_L1_h_W_COUNT);
    memcpy(m_model_optim_data.m_h_l1_b, m_model_optim_data_flash.m_h_l1_b, LSTM_Q_L1_h_B_COUNT * sizeof(int32_t));

    // Hand the owned pointers to the LSTM (it only stores them; MasterHandle owns/frees).
    m_optim_lstm->set_weights(
        feature_ch, LSTM_Q_HIDDEN,
        m_model_optim_data.m_x_l0_w, m_model_optim_data.m_x_l0_b,
        m_model_optim_data.m_h_l0_w, m_model_optim_data.m_h_l0_b,
        m_model_optim_data.m_x_l1_w, m_model_optim_data.m_x_l1_b,
        m_model_optim_data.m_h_l1_w, m_model_optim_data.m_h_l1_b
    );

    // Build LUTs (sigmoid/tanh) and elementwise requant multipliers.
    m_optim_lstm->prepare();

    m_lstm_hq = new int8_t[2 * LSTM_Q_HIDDEN];
    m_lstm_cq = new int8_t[2 * LSTM_Q_HIDDEN];

    m_lstm_cq16 = new int16_t[2 * LSTM_Q_HIDDEN];
    reset_optim_lstm_state();

}
void MasterHandle::reset_optim_lstm_state()
{
    // float h=0 -> int8 IN_H zero-point;  float c=0 -> int8 IN_C zero-point
    for (int i = 0; i < 2 * LSTM_Q_HIDDEN; i++) {
        m_lstm_hq[i] = (int8_t)LSTM_Q_IN_H_ZP;   // 7
        m_lstm_cq[i] = (int8_t)LSTM_Q_IN_C_ZP;   // (c8 path zero-point)
    }

    if (m_lstm_cq16) {
        // float c=0 -> int16 IN_C zero-point (-12336)
        int16_t c0 = (int16_t)LSTM_Q_IN_C_ZP;
        for (int i = 0; i < 2 * LSTM_Q_HIDDEN; i++) m_lstm_cq16[i] = c0;
    }
}
void MasterHandle::clear_optim_lstm()
{
    delete m_optim_lstm;
    m_optim_lstm = nullptr;

    delete[] m_model_optim_data.m_x_l0_w;  m_model_optim_data.m_x_l0_w = nullptr;
    delete[] m_model_optim_data.m_x_l0_b;  m_model_optim_data.m_x_l0_b = nullptr;
    delete[] m_model_optim_data.m_h_l0_w;  m_model_optim_data.m_h_l0_w = nullptr;
    delete[] m_model_optim_data.m_h_l0_b;  m_model_optim_data.m_h_l0_b = nullptr;
    delete[] m_model_optim_data.m_x_l1_w;  m_model_optim_data.m_x_l1_w = nullptr;
    delete[] m_model_optim_data.m_x_l1_b;  m_model_optim_data.m_x_l1_b = nullptr;
    delete[] m_model_optim_data.m_h_l1_w;  m_model_optim_data.m_h_l1_w = nullptr;
    delete[] m_model_optim_data.m_h_l1_b;  m_model_optim_data.m_h_l1_b = nullptr;

    delete[] m_lstm_hq;   m_lstm_hq = nullptr;
    delete[] m_lstm_cq;   m_lstm_cq = nullptr;
    delete[] m_lstm_cq16; m_lstm_cq16 = nullptr;
}


uint64_t MasterHandle::optim_lstm_step(const float* x_slice_f32, int x_len,
                                    float* y_out_f32, bool use_c16)
{
    const int H = LSTM_Q_HIDDEN;
    int8_t x_q[LSTM_Q_HIDDEN];
    for (int i = 0; i < x_len; i++) {
        float q = roundf(x_slice_f32[i] / LSTM_Q_IN_X_SCALE) + LSTM_Q_IN_X_ZP;
        x_q[i] = (q < -128.f) ? -128 : (q > 127.f) ? 127 : (int8_t)q;
    }
    int8_t y_q[LSTM_Q_HIDDEN];
    uint64_t t0 = esp_timer_get_time();
    if (use_c16) m_optim_lstm->run_timestep_q_c16(x_q, m_lstm_hq, m_lstm_cq16, y_q);
    else         m_optim_lstm->run_timestep_q   (x_q, m_lstm_hq, m_lstm_cq,   y_q);
    uint64_t elapsed = esp_timer_get_time() - t0;
    for (int i = 0; i < H; i++)
        y_out_f32[i] = ((int32_t)y_q[i] - LSTM_Q_L1_H_NEW_ZP) * LSTM_Q_L1_H_NEW_SCALE;
    return elapsed;
}


void MasterHandle::apply_elu_float(float* in, int n, float alpha)
{
   for (int i = 0; i < n; ++i) {
        in[i] = (in[i] >= 0.0f) ? in[i] : alpha * (expf(in[i]) - 1.0f);
    }
}


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

// ---- stage 1, int8 out (your version, kept) ----
void MasterHandle::stage1_first_a_int8(const float* input_ptr, int input_size,
                                       int8_t* out_int8, int first_a_out_size,
                                       WindowStageTimes* wt)
{
    const int* input_lengths_a  = new const int[1]{ input_size };
    const int* output_lengths_a = new const int[1]{ first_a_out_size };
    bool success = m_model[0]->predict(input_ptr, input_lengths_a, out_int8, output_lengths_a);
    // read profiler ticks BEFORE clearing
    wt->infa_ticks = m_model[0]->getTotalTicks();
    if (!success) {
        ESP_LOGE("MASTERHandle", "First model inference failed");
    }
    m_model[0]->ClearProfiler();


    // ELU on int8 in place. LUT maps model[0] output space -> model[1] input space.
    uint64_t startTime_ELU_A = esp_timer_get_time();
    apply_elu_lut(out_int8, first_a_out_size);
    wt->elua_us = esp_timer_get_time() - startTime_ELU_A;

    delete[] input_lengths_a;
    delete[] output_lengths_a;
}

// ---- concat in int8 (model[1] input space): [ elu'd_A | quant(x) ] ----
void MasterHandle::concat_with_input_int8(const int8_t* first_a_after_elu, int first_a_out_size,
                                          const float* input_ptr, int input_size,
                                          int8_t* concat_out)
{
    // first part: A's ELU output is already in model[1] input space
    for (int i = 0; i < first_a_out_size; i++)
        concat_out[i] = first_a_after_elu[i];

    // tail: quantize original x into model[1] input space
    const float   in_scale = m_model[1]->getInputScale(0);
    const int32_t in_zp    = m_model[1]->getInputZeroPoint(0);
    for (int i = 0; i < input_size; i++) {
        const float q = roundf(input_ptr[i] / in_scale) + in_zp;
        if      (q < -128.0f) concat_out[i + first_a_out_size] = -128;
        else if (q >  127.0f) concat_out[i + first_a_out_size] =  127;
        else                  concat_out[i + first_a_out_size] = (int8_t)q;
    }
}

// ---- stage 2, int8 in -> float out ----
void MasterHandle::stage2_first_b_int8_in(const int8_t* concat_in, int concat_in_size,
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

    // ELU stays plain float (matches the float path); model[1] output is dequantized float
    uint64_t startTime_ELU_B = esp_timer_get_time();
    apply_elu_float(out_f32, first_b_out_size, 1.0);
    wt->elub_us = esp_timer_get_time() - startTime_ELU_B;

    delete[] input_lengths_b;
    delete[] output_lengths_b;
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
    //m_model[0]->getTotalProfileTimePerOp();
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
//   - use_optim_lstm: route LSTM through the native ESP-NN implementation
//   - use_c16:        within the native path, use the int16 cell-state variant
// ============================================================
void MasterHandle::recurrent_loop(const float* first_b_out, int first_out_len,
                                  float* output_ptr, int out_len, int step,
                                  float* second_input_ptr, const int* second_input_lengths,
                                  int intermediate_size, int state_size,
                                  StepFn lstm_step, StepFn reg_step,
                                  TrialStats* trial,
                                  bool use_optim_lstm,
                                  bool use_c16)
{
    const int y_chunk    = CONFIG_OUTPUT_CHANNELS * step;
    const int lstm_chunk = feature_ch * step;

    float* intermediate_A_output_ptr = new float[lstm_chunk + 2 * state_size]{0};
    const int* intermediate_A_output_lengths = new const int[3]{
            lstm_chunk, state_size, state_size };
    float* intermediate_output_ptr = new float[y_chunk * step]{0};
    const int* intermediate_output_lengths = new const int[1]{ y_chunk * step };

    bool success = false;

    for (int t = 0; t < out_len; t += step) {

        // x slice: channel-major (56, step) from first model's (56, T) output
        for (int j = 0; j < feature_ch; j++)
            for (int s = 0; s < step; s++)
                second_input_ptr[j * step + s] = first_b_out[j * first_out_len + t + s];

        if (use_optim_lstm) {
            // ---- native ESP-NN LSTM path (step==1). h/c carried in m_lstm_hq/cq(16). ----
            assert(step == 1);   // native export is step==1 only

            // x slice is the first lstm_chunk (== H for step==1) of second_input_ptr.
            // use_c16 selects the int16 cell-state variant inside optim_lstm_step.
            uint64_t lstm_us = optim_lstm_step(second_input_ptr, lstm_chunk,
                                               intermediate_A_output_ptr, use_c16);
            success = true;

            bool success_2 = (this->*reg_step)(intermediate_A_output_ptr, intermediate_A_output_lengths,
                                               intermediate_output_ptr, intermediate_output_lengths);
            int32_t reg_ticks = m_model[3]->getTotalTicks();

            for (int j = 0; j < CONFIG_OUTPUT_CHANNELS; j++)
                for (int s = 0; s < step; s++)
                    output_ptr[j * out_len + t + s] = intermediate_output_ptr[j * step + s];

            if (success && success_2) {
                trial->add_lstm_native_step((double)lstm_us);   // native us
                trial->add_reg_step(reg_ticks);
                // note: trial->add_lstm_step(...) intentionally NOT called here,
                // so lstm_step_ticks stays 0 for native runs
            } else {
                ESP_LOGE("MASTERHandle", "Second model inference failed @ time_step: %d", t);
            }
            m_model[3]->ClearProfiler();

        } else {
            // ---- original tflite LSTM path (unchanged) ----
            if (t > 0) {
                for (int j = 0; j < state_size; j++) {
                    second_input_ptr[intermediate_size + j] =
                        intermediate_A_output_ptr[lstm_chunk + j];                      // h
                    second_input_ptr[intermediate_size + state_size + j] =
                        intermediate_A_output_ptr[lstm_chunk + state_size + j];         // c
                }
            }

            success = (this->*lstm_step)(second_input_ptr, second_input_lengths,
                                         intermediate_A_output_ptr, intermediate_A_output_lengths);
            int32_t lstm_ticks = m_model[2]->getTotalTicks();

            bool success_2 = (this->*reg_step)(intermediate_A_output_ptr, intermediate_A_output_lengths,
                                               intermediate_output_ptr, intermediate_output_lengths);
            int32_t reg_ticks = m_model[3]->getTotalTicks();

            for (int j = 0; j < CONFIG_OUTPUT_CHANNELS; j++)
                for (int s = 0; s < step; s++)
                    output_ptr[j * out_len + t + s] = intermediate_output_ptr[j * step + s];

            if (success && success_2) {
                trial->add_lstm_step(lstm_ticks);
                trial->add_reg_step(reg_ticks);
            } else {
                ESP_LOGE("MASTERHandle", "Second model inference failed @ time_step: %d", t);
            }
            m_model[2]->ClearProfiler();
            m_model[3]->ClearProfiler();
        }
    }

    // post-loop state stash (only meaningful for tflite path; harmless otherwise)
    if (!use_optim_lstm) {
        for (int j = 0; j < state_size; j++) {
            second_input_ptr[intermediate_size + j] =
                intermediate_A_output_ptr[lstm_chunk + j];
            second_input_ptr[intermediate_size + state_size + j] =
                intermediate_A_output_ptr[lstm_chunk + state_size + j];
        }
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

    float* first_output_ptr_b = new float[first_b_out_size];

    if (qc.first_a_is_int8) {
        // -------- int8-first workflow: concat happens in model[1] input space --------
        // STAGE 1: model[0] emits int8 (already ELU'd into model[1] input space)
        int8_t* first_a_after_elu = new int8_t[first_a_out_size];
        stage1_first_a_int8(input_ptr, input_size,
                            first_a_after_elu, first_a_out_size, wt);

        // CONCAT in int8: [ elu'd_A | quant(x) into model[1] input space ]
        int8_t* concat_int8 = new int8_t[concat_first_a_out_size];
        concat_with_input_int8(first_a_after_elu, first_a_out_size,
                               input_ptr, input_size, concat_int8);

        // STAGE 2: model[1] int8 in -> float out (+ ELU float)
        stage2_first_b_int8_in(concat_int8, concat_first_a_out_size,
                               first_output_ptr_b, first_b_out_size, wt);

        delete[] first_a_after_elu;
        delete[] concat_int8;
    } else {
        // -------- original float workflow (unchanged) --------
        // STAGE 1: First Block A (+ ELU) -> float result. Variant chosen by qc.first_a.
        float* first_a_after_elu = new float[first_a_out_size];
        (this->*(qc.first_a))(input_ptr, input_size,
                              first_a_after_elu, first_a_out_size, wt);

        // CONCAT with original input
        float* concat_first_output_ptr_a = new float[concat_first_a_out_size];
        concat_with_input(first_a_after_elu, first_a_out_size,
                          input_ptr, input_size, concat_first_output_ptr_a);

        // STAGE 2: First Block B (+ ELU) -> float result. Variant chosen by qc.first_b.
        (this->*(qc.first_b))(concat_first_output_ptr_a, concat_first_a_out_size,
                              first_output_ptr_b, first_b_out_size, wt);

        delete[] first_a_after_elu;
        delete[] concat_first_output_ptr_a;
    }

    // STAGE 3+4: LSTM + Regressor loop. Variants chosen by qc.lstm_step / qc.reg_step.
    //            use_optim_lstm routes to native LSTM; use_c16 selects int16 cell state.
    //            pushes per-step ticks into trial
    recurrent_loop(first_output_ptr_b, first_out_len,
                   output_ptr, out_len, step,
                   second_input_ptr, second_input_lengths,
                   intermediate_size, state_size,
                   qc.lstm_step, qc.reg_step, trial,
                   qc.use_optim_lstm, qc.use_c16);

    wt->window_us = esp_timer_get_time() - win_start;

    delete[] first_output_ptr_b;
    //ESP_LOGI("MASTERHandle", "CLEARED ALL ARRAYS");
}