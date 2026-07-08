#ifndef MASTER_HANDLE_H
#define MASTER_HANDLE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "MicroInterface/Model.h"
#include "OptimizedNativeLSTM.h"
#include "lsl_handle.h"

#include "trial_report.h"   // AvgMax, WindowStageTimes, TrialStats, CSV helpers

class MasterHandle {
    public:
        struct optim_lstm_weights_biases_t
            {
                int8_t* m_x_l0_w;
                int32_t* m_x_l0_b;

                int8_t* m_h_l0_w;
                int32_t* m_h_l0_b;

                int8_t* m_x_l1_w;
                int32_t* m_x_l1_b;

                int8_t* m_h_l1_w;
                int32_t* m_h_l1_b;
            };

    struct optim_lstm_weights_biases_const_t {
        const int8_t*  m_x_l0_w;  const int32_t* m_x_l0_b;
        const int8_t*  m_h_l0_w;  const int32_t* m_h_l0_b;
        const int8_t*  m_x_l1_w;  const int32_t* m_x_l1_b;
        const int8_t*  m_h_l1_w;  const int32_t* m_h_l1_b;
    };
    private:
        Model** m_model;
        uint8_t* m_psram_model_ptr[4];
        ModelFlash* m_model_flash;

        uint32_t* input_sizes;
        uint32_t* output_sizes;

        int models_count;
        const uint8_t** flash_models_ptrs;
        const uint32_t* model_input_sizes;
        const uint32_t* model_output_sizes;
        const uint32_t* model_sizes;

        int8_t elu_lut[256];
        const int feature_ch = 56;

        bool usePSRAM = true;
        OptimizedNativeLSTM* m_optim_lstm;
        optim_lstm_weights_biases_t m_model_optim_data;
        optim_lstm_weights_biases_const_t m_model_optim_data_flash;

        int8_t* m_lstm_hq = nullptr;   // [2*H] persistent hidden state (int8, IN_H space)
        int8_t* m_lstm_cq = nullptr;   // [2*H] persistent cell state   (int8, IN_C space)

        int16_t* m_lstm_cq16 = nullptr;   // [2*H] int16 cell state for c16 path
    public:
        MasterHandle(const int models_count,
            const uint8_t** flash_models_ptrs,
            const uint32_t* model_input_sizes,
            const uint32_t* model_output_sizes,
            const uint32_t* model_sizes
        );

        ~MasterHandle(){};

        void init_model(int model_index, int internal_index,
        char* report_buffer, int size);
        void init_models(int model_1_index, int model_2_index,int model_3_index,
             bool usePSRAM, char* report_buffer, int size);

        void init_models(int model_1_index, int model_2_index,int model_3_index, int model_4_index,
        bool usePSRAM, char* report_buffer, int size);


        void init_optim_lstm(const char* partition);
        void reset_optim_lstm_state();
        void clear_optim_lstm();

        uint64_t optim_lstm_step(const float* x_slice_f32, int x_len, float* y_out_f32, bool use_c16);

        double print_output(const float* output_window, int output_size, const float* correct_window);
        
        double print_output(const float* output_window, int output_size, const float* correct_window,
                            double& out_mse, double& out_mae, double& out_nmse);
        void display_output(const float* output_window, int output_sizes);
        void clear_models();

        // Size (bytes) of the model at the given flash index (same index passed to init_models).
        uint32_t getModelSizeBytes(int flash_index) const { return model_sizes[flash_index]; }

        // Arena bytes actually used by the model in internal slot 0..3 (recorded by
        // the Model at AllocateTensors() time). NOTE: indexed by internal slot, not
        // flash index, because it reads from the live m_model[] objects.
        size_t getArenaUsedBytes(int internal_index) const {
            return (m_model && m_model[internal_index])
                 ? m_model[internal_index]->getArenaUsedBytes()
                 : 0;
        }

        // ============================================================
        // STAGE 1: First Block A  (conv front-end)
        //   - emits float32 result of size first_a_out_size into out_f32
        //   - already includes the ELU activation
        //   - fills wt->infa_ticks (profiler) and wt->elua_us (esp_timer)
        // float / int8 variants differ only in predict() dtype + ELU path
        // ============================================================
        void stage1_first_a_f32(const float* input_ptr, int input_size,
                                float* out_f32, int first_a_out_size,
                                WindowStageTimes* wt);
        void stage1_first_a_int8(const float* input_ptr, int input_size,
                                 float* out_f32, int first_a_out_size,
                                 WindowStageTimes* wt);

        // int8-out variant of stage 1 (model[0] emits int8 in model[1] input space, ELU via LUT in place)
        void stage1_first_a_int8(const float* input_ptr, int input_size,
                                int8_t* out_int8, int first_a_out_size,
                                WindowStageTimes* wt);

        

        // Concatenate stage1 output (already ELU'd) with the original input.
        // (identical for both variants -> single function)
        void concat_with_input(const float* first_a_after_elu, int first_a_out_size,
                               const float* input_ptr, int input_size,
                               float* concat_out);

        // concat in int8 (model[1] input space): [ elu'd_A | quant(x) ]
        void concat_with_input_int8(const int8_t* first_a_after_elu, int first_a_out_size,
                                    const float* input_ptr, int input_size,
                                    int8_t* concat_out);

        // stage 2 with int8 input, float output (model[1] = int8 model, dequantized result)
        void stage2_first_b_int8_in(const int8_t* concat_in, int concat_in_size,
                                    float* out_f32, int first_b_out_size,
                                    WindowStageTimes* wt);
        // ============================================================
        // STAGE 2: First Block B  (conv/dense)
        //   - emits float32 result of size first_b_out_size into out_f32
        //   - already includes the ELU activation
        //   - fills wt->infb_ticks (profiler) and wt->elub_us (esp_timer)
        // ============================================================
        void stage2_first_b_f32(const float* concat_in, int concat_in_size,
                                float* out_f32, int first_b_out_size,
                                WindowStageTimes* wt);
        void stage2_first_b_int8(const float* concat_in, int concat_in_size,
                                 float* out_f32, int first_b_out_size,
                                 WindowStageTimes* wt);

        // ============================================================
        // STAGE 3+4: LSTM (model[2]) + Regressor head (model[3]) loop
        //   - the loop, the h/c carry, and the scatter are shared
        //   - only the per-step predict() dtype differs, so each model
        //     gets a tiny "step" function selected once before the loop
        // ============================================================
        bool lstm_step_f32(float* second_input_ptr, const int* second_input_lengths,
                           float* lstm_out, const int* lstm_out_lengths);
        bool lstm_step_int8(float* second_input_ptr, const int* second_input_lengths,
                            float* lstm_out, const int* lstm_out_lengths);

        bool regressor_step_f32(float* lstm_out, const int* lstm_out_lengths,
                                float* reg_out, const int* reg_out_lengths);
        bool regressor_step_int8(float* lstm_out, const int* lstm_out_lengths,
                                 float* reg_out, const int* reg_out_lengths);

        // --- function-pointer typedefs for runtime-swappable stages ---
        // first_a and first_b share the same shape, so one typedef covers both.
        using StageFn = void (MasterHandle::*)(const float* /*in*/, int /*in_size*/,
                                               float* /*out_f32*/, int /*out_size*/,
                                               WindowStageTimes* /*wt*/);

        // lstm step and regressor step share the recurrent per-step shape.
        using StepFn  = bool (MasterHandle::*)(float* /*in*/, const int* /*in_len*/,
                                               float* /*out*/, const int* /*out_len*/);

        // Bundle the four stage choices. Build one of these per config (matching
        // whatever init_models() you called) and pass it to dual_inference.
        struct QuantConfig {
            StageFn first_a;
            StageFn first_b;
            StepFn  lstm_step;
            StepFn  reg_step;
            const char* label_a    = "f32";
            const char* label_b    = "f32";
            const char* label_lstm = "f32";
            const char* label_reg  = "f32";
            bool first_a_is_int8 = false;
            bool use_optim_lstm  = false;   // true -> native ESP-NN LSTM (step==1 only)
            bool use_c16 = false;
        };
        // The recurrent loop orchestrator (dtype-agnostic; takes member fn ptrs).
        // Pushes per-step ticks (lstm/regressor) into the trial accumulator.
        void recurrent_loop(const float* first_b_out, int first_out_len,
                                  float* output_ptr, int out_len, int step,
                                  float* second_input_ptr, const int* second_input_lengths,
                                  int intermediate_size, int state_size,
                                  StepFn lstm_step, StepFn reg_step,
                                  TrialStats* trial,
                                  bool use_optim_lstm,
                                  bool use_c16);   

        // Top-level orchestrator. Stages chosen at runtime via QuantConfig.
        // Reports timing through wt (per-window) and trial (per-step).
        void dual_inference(const QuantConfig& qc,
                            const float* input_ptr, int input_size,
                            float* output_ptr, int output_size,
                            float* second_input_ptr, const int* second_input_lengths,
                            int intermediate_size, int first_out_len,
                            WindowStageTimes* wt, TrialStats* trial);

    
        void apply_elu_float(float* in, int n, float alpha);
        void build_elu_lut(float in_scale, int32_t in_zp,
                        float out_scale, int32_t out_zp, float alpha);
        // between blocks, over the whole tensor:
        void apply_elu_lut(int8_t *t, int n);

        void apply_elu_lut_int_float(
        int8_t* in, int n, float in_scale, int32_t in_zp,
        float alpha, float* out);
};

#endif // MASTER_HANDLE_H