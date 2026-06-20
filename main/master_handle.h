#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "MicroInterface/Model.h"
#include "trial_report.h"
#include "lsl_handle.h"

class MasterHandle {
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

        void append_report(char* report_buffer, int size,
                           const char* tag, uint64_t duration_us);
    public:
        MasterHandle(const int models_count,
            const uint8_t** flash_models_ptrs,
            const uint32_t* model_input_sizes,
            const uint32_t* model_output_sizes,
            const uint32_t* model_sizes
        );

        using StageFn = void (MasterHandle::*)(const float* /*in*/, int /*in_size*/,
                                       float* /*out_f32*/, int /*out_size*/,
                                       WindowStageTimes* /*wt*/);
 
        using StepFn  = bool (MasterHandle::*)(float* /*in*/, const int* /*in_len*/,
                                       float* /*out*/, const int* /*out_len*/);
        
        struct QuantConfig {
            StageFn first_a;
            StageFn first_b;
            StepFn  lstm_step;
            StepFn  reg_step;
            // quant labels carried alongside the function choice, so the CSV can
            // print them without a second source of truth.
            const char* label_a    = "f32";
            const char* label_b    = "f32";
            const char* label_lstm = "f32";
            const char* label_reg  = "f32";
        };

        ~MasterHandle(){};
        void init_model(int model_index, int internal_index,
        char* report_buffer, int size);
        void init_models(int model_1_index, int model_2_index,int model_3_index,
             bool usePSRAM, char* report_buffer, int size);
        
        void init_models(int model_1_index, int model_2_index,int model_3_index, int model_4_index,
        bool usePSRAM, char* report_buffer, int size);

        void init_optimized_lstm();

        double print_output(const float* output_window, int output_size, const float* correct_window);
        
        void display_output(const float* output_window, int output_sizes);
        void clear_models();
        
        void stage1_first_a_f32(const float* input_ptr, int input_size,
                                float* out_f32, int first_a_out_size,
                                char* report_buffer, int size);

        void stage1_first_a_int8(const float* input_ptr, int input_size,
                                       float* out_f32, int first_a_out_size,
                                       WindowStageTimes* wt);


        void concat_with_input(const float* first_a_after_elu, int first_a_out_size,
                               const float* input_ptr, int input_size,
                               float* concat_out);

        void stage2_first_b_f32(const float* concat_in, int concat_in_size,
                                float* out_f32, int first_b_out_size,
                                char* report_buffer, int size);

        void stage2_first_b_int8(const float* concat_in, int concat_in_size,
                                 float* out_f32, int first_b_out_size,
                                 char* report_buffer, int size);
        
        bool lstm_step_f32(float* second_input_ptr, const int* second_input_lengths,
                           float* lstm_out, const int* lstm_out_lengths);
        bool lstm_step_int8(float* second_input_ptr, const int* second_input_lengths,
                            float* lstm_out, const int* lstm_out_lengths);
 
        bool regressor_step_f32(float* lstm_out, const int* lstm_out_lengths,
                                float* reg_out, const int* reg_out_lengths);
        bool regressor_step_int8(float* lstm_out, const int* lstm_out_lengths,
                                 float* reg_out, const int* reg_out_lengths);
        
                                 
        void recurrent_loop(const float* first_b_out, int first_out_len,
                            float* output_ptr, int out_len, int step,
                            float* second_input_ptr, const int* second_input_lengths,
                            int intermediate_size, int state_size,
                            bool (MasterHandle::*lstm_step)(float*, const int*, float*, const int*),
                            bool (MasterHandle::*reg_step)(float*, const int*, float*, const int*),
                            char* report_buffer, int size);
            
        // void dual_inference(const float* input_ptr, int input_size,
        //                     float* output_ptr, int output_size,
        //                     float* second_input_ptr, const int* second_input_lengths,
        //                     int intermediate_size, int first_out_len,
        //                     char* report_buffer, int size);

        // dual_inference now reports through the accumulator instead of a char*.
        void dual_inference(const QuantConfig& qc,
                            const float* input_ptr, int input_size,
                            float* output_ptr, int output_size,
                            float* second_input_ptr, const int* second_input_lengths,
                            int intermediate_size, int first_out_len,
                            WindowStageTimes* wt, TrialStats* trial);
        
        // recurrent_loop gains the TrialStats* so it can push per-step ticks:
        void recurrent_loop(const float* first_b_out, int first_out_len,
                            float* output_ptr, int out_len, int step,
                            float* second_input_ptr, const int* second_input_lengths,
                            int intermediate_size, int state_size,
                            StepFn lstm_step, StepFn reg_step,
                            TrialStats* trial);

        void apply_elu_float(float* in, int n, float alpha);
        void build_elu_lut(float in_scale, int32_t in_zp,
                        float out_scale, int32_t out_zp, float alpha);
        // between blocks, over the whole tensor:
        void apply_elu_lut(int8_t *t, int n);

        void apply_elu_lut_int_float(
        int8_t* in, int n, float in_scale, int32_t in_zp,
        float alpha, float* out);


};