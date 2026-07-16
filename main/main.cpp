#include "master_handle.h"
#include "benchmark_handle.h"

#include "lsl_handle.h"

#define EMG_MODEL_BENCHMARK
//#define RANDOM_WEIGHTS_BENCHMARK
//#define QUANTIZATION_FIDELITY
//#define RANDOM_WEIGHTS_BENCHMARK
#ifdef EMG_MODEL_BENCHMARK
    #include "binary_manifests/refactored_models_verification/manifest.h"
    #include "binary_manifests/refactored_models_verification/models_manifest_0.h"
#endif

#ifdef RANDOM_WEIGHTS_BENCHMARK
    // #include "binary_manifests/lstm-timestep-verification/manifest_0.h"
    // #include "binary_manifests/lstm-timestep-verification/models_manifest_0.h"

    // #include "binary_manifests/dense_layers_seed1/manifest_0.h"
    // #include "binary_manifests/dense_layers_seed1/models_manifest_0.h"

    // #include "binary_manifests/lstm-layers-verification/manifest_0.h"
    // #include "binary_manifests/lstm-layers-verification/models_manifest_0.h"

    // #include "binary_manifests/lstm-hiddensize-verification2/manifest_0.h"
    // #include "binary_manifests/lstm-hiddensize-verification2/models_manifest_0.h"

    #include "binary_manifests/Conv2D-InputSize-Verification/manifest_0.h"
    #include "binary_manifests/Conv2D-InputSize-Verification/models_manifest_0.h"

#endif

#ifdef QUANTIZATION_FIDELITY
    // #include "binary_manifests/multiple_train_trials/manifest_0.h"
    // #include "binary_manifests/calibrated_better/models_manifest_0.h"
    #include "binary_manifests/quant_fidelity_test_verification/manifest.h"
    #include "binary_manifests/quant_fidelity_test_verification/models_manifest_0.h"
#endif

#include "tensorflow/lite/micro/micro_time.h"

#include "benchmark_metrics.h"
#include "sram_fitness.h"
#include "tensorflow/lite/micro/micro_time.h"   // tflite::ticks_per_second()
#define REPORT_MAX 20000


#define LSTM_FEATURES (CONFIG_INPUT_CHANNELS + 48)


static double run_inference_window(
    MasterHandle* master_handle,
    const MasterHandle::QuantConfig& qc,
    int win_count,
    const uint8_t* const* x_data,
    const uint8_t* const* y_data,
    int config_index,
    int trial_len_samples,
    int out_trial_len,
    int input_win_len, int input_size,
    int output_win_len, int output_size,
    int intermediate_size, int first_out_len,
    float* input_trial, float* output_trial, float* correct_trial,
    float* second_input_ptr, const int* second_input_lengths,
    WindowStageTimes* wt, TrialStats* trial)
{
    const int win_start_t = win_count * input_win_len;   // start time of this window (samples)
    const int out_start_t = win_count * output_win_len;  // same window counter as input
    //ESP_LOGI("MAIN", "Win: %d", win_count);
    //memory copy from flash to sram arrays
    // memcpy(&input_trial[in_win * CONFIG_INPUT_CHANNELS],
    // &x_data[0][in_win * CONFIG_INPUT_CHANNELS * sizeof(float)], input_size * sizeof(float));
    // memcpy(&correct_trial[out_win * CONFIG_OUTPUT_CHANNELS],
    // &y_data[i][out_win * CONFIG_OUTPUT_CHANNELS * sizeof(float)], output_size * sizeof(float));

    for (int ch = 0; ch < CONFIG_INPUT_CHANNELS; ch++) {
        const float* src =
            (const float*)x_data[0] + (size_t)ch * trial_len_samples + win_start_t;
        memcpy(&input_trial[ch * input_win_len],
            src,
            input_win_len * sizeof(float));
    }

    for (int ch = 0; ch < CONFIG_OUTPUT_CHANNELS; ch++) {
        const float* src =
            (const float*)y_data[config_index] + (size_t)ch * out_trial_len + out_start_t;
        memcpy(&correct_trial[ch * output_win_len],
            src,
            output_win_len * sizeof(float));
    }

    master_handle->dual_inference(
        qc,
        &input_trial[0],
        input_size,
        &output_trial[0],
        output_size,
        second_input_ptr,
        second_input_lengths,
        intermediate_size, first_out_len,
        wt, trial);

    // double current_mse = master_handle->print_output(
    //     &output_trial[0],
    //     output_size,
    //     &correct_trial[0]);
    double current_mse;double current_mae;double current_nmse;
    master_handle->print_output(
        &output_trial[0],
        output_size,
        &correct_trial[0],
        current_mse,
        current_mae,
        current_nmse);

    wt->mse = current_mse;
    wt->mae = current_mae;
    wt->nmse = current_nmse;
    return current_mse;
}

#ifdef EMG_MODEL_BENCHMARK
    void run_app()
    {

        // We need to now do a for loop for all models
        // Step 1: Pull the data from lsl into the windowed buffer
        // Step 2: Run inference on the windowed buffer which should put the appropriate output into the output buffer
        // Step 3: Push the output data buffer to the LSL outlet

        char final_report[REPORT_MAX] = {0};
        ModelFlash* mf = new ModelFlash();
        const uint8_t** x_data = new const uint8_t*[1];
        const uint8_t** y_data = new const uint8_t*[DATA_CONFIG_COUNT];


        mf->allocatePointerOnFlashXY(
            "benchmark_data",
            X_REGION_BYTE_OFFSET,
            x_data,
            Y_REGION_BYTE_OFFSET,
            y_data,
            DATA_CONFIG_COUNT,
            MAIN_X_OFFSETS,
            MAIN_Y_OFFSETS,
            OFFSET_TYPE::FLOAT32
        );



        uint32_t max_output_size = 0;
        for (int i = 0; i < DATA_CONFIG_COUNT; i++) {
            if (MAIN_Y_SIZES[i] > max_output_size) max_output_size = MAIN_Y_SIZES[i];
        }
        float* output_trial = new float[max_output_size]; //biggest output_trial = input_trial / downsample_rate
        float* correct_trial = new float[max_output_size];

        const uint32_t max_input_size = Y_REGION_BYTE_OFFSET / sizeof(float);
        const uint32_t input_trial_len = max_input_size / 8;

        float* input_trial = new float[max_input_size]; // max_input_size = samples / trial (about 9000 * 8 channels)
        const uint8_t** models_on_flash = new const uint8_t*[MODEL_COUNT];

        bool success = mf->allocatePointerOnFlash("benchmark_models", models_on_flash,
        MODEL_COUNT, MODEL_OFFSETS, OFFSET_TYPE::INT8);

        if (!success) {
            ESP_LOGE("MAIN", "Could not initialize mmaped pointers");
            return;
        }

        MasterHandle* master_handle = new MasterHandle(MODEL_COUNT,
            models_on_flash,
            INPUT_SIZES,
            OUTPUT_SIZES,
            MODEL_SIZES);

        // CSV header, written ONCE at the top of the report.
        trial_report_header(final_report, REPORT_MAX);

        for(int i = 0; i < CONFIG_COUNT; i++)
        {
            uint64_t startTimeTrial = esp_timer_get_time();
            const int input_win_len     = (int)MAIN_WINDOW_LENS[i];
            const int input_size        = (int)MAIN_WINDOW_LENS[i] * CONFIG_INPUT_CHANNELS;

            const int output_win_len    = (int)MAIN_OUT_LENS[i];
            const int output_size       = (int)MAIN_OUT_LENS[i] * CONFIG_OUTPUT_CHANNELS;

            // Calculates the effective trial length which is not always = INPUT_TRIAL_LENGTH / DOWNSAMPLE_RATE
            // Due to rounding
            const int output_trial_len  = (int)(MAIN_Y_SIZES[i] / CONFIG_OUTPUT_CHANNELS);

            const int intermediate_size = (int)MAIN_INTERMEDIATE_SIZE[i];
            const int first_out_len     = (int)MAIN_OUT_LENS[i];
            const int out_len           = output_size / CONFIG_OUTPUT_CHANNELS;

            const int state_size        = 2 * LSTM_FEATURES;
            const int lstm_step         = intermediate_size / LSTM_FEATURES;   // step used by the LSTM loop

            printf(
            "input_win_len:%d\ninput_size:%d\noutput_win_len:%d\noutput_size:%d\noutput_trial_len:%d\nintermediate_size:%d\nfirst_out_len:%d\nstate_size%d\n", input_win_len,
            input_size, output_win_len, output_size, output_trial_len, intermediate_size, first_out_len, state_size);
            // LOOP over the entire trial window by window
            // loope for j, j< trial_length, j+=window_len
            int in_win =0;
            int out_win =0;

            //Need to maintain h and c states accross trials
            // Inputs:  [ x (56*step) | h (2*56) | c (2*56) ]
            float* second_input_ptr = new float[intermediate_size + 2 * state_size]{0};

            const int* second_input_lengths = new const int[3]{
                intermediate_size,
                state_size,
                state_size
            };

            ESP_LOGI("MAIN", "Infenencing Config (%d)", i);

            // 0,1,2,3. 4,5,6,7
            //trying with only the first one being int8
            const int idx_a    = i*8 + 4;
            const int idx_b    = i*8 + 5;
            const int idx_lstm = i*8 + 2;
            const int idx_reg  = i*8 + 3;

            // const int idx_a    = i*8 + 0;
            // const int idx_b    = i*8 + 1;
            // const int idx_lstm = i*8 + 2;
            // const int idx_reg  = i*8 + 3;
            master_handle->init_models(idx_a, idx_b, idx_lstm, idx_reg,
                                    true, final_report, REPORT_MAX);
            master_handle->init_optim_lstm("lstm_weights");
            // Pick the per-stage quant variants to match the init_models indices above.
            // The labels carried here are what the CSV prints for this trial.
            MasterHandle::QuantConfig qc;
            qc.first_a_is_int8 = true;
            qc.use_optim_lstm = false;
            qc.use_c16 = false;
            qc.first_a    = &MasterHandle::stage1_first_a_int8;
            qc.label_a    = "int8";
            qc.first_b    = &MasterHandle::stage2_first_b_int8; 
            qc.label_b    = "int8";
            qc.lstm_step  = &MasterHandle::lstm_step_f32;      qc.label_lstm = "f32";
            qc.reg_step   = &MasterHandle::regressor_step_f32; qc.label_reg  = "f32";

            // MasterHandle::QuantConfig qc;
            // qc.first_a_is_int8 = false;
            // qc.use_optim_lstm = false;
            // qc.use_c16 = false;
            // qc.first_a    = &MasterHandle::stage1_first_a_f32;
            // qc.label_a    = "f32";
            // qc.first_b    = &MasterHandle::stage2_first_b_f32; 
            // qc.label_b    = "f32";
            // qc.lstm_step  = &MasterHandle::lstm_step_f32;      qc.label_lstm = "f32";
            // qc.reg_step   = &MasterHandle::regressor_step_f32; qc.label_reg  = "f32";
            
            // ---- per-trial accumulator ----
            TrialStats trial;
            trial.config       = i;
            trial.window_len   = (long)MAIN_WINDOW_LENS[i];
            trial.step         = lstm_step;
            trial.quant_a      = qc.label_a;
            trial.quant_b      = qc.label_b;
            trial.quant_lstm   = qc.label_lstm;
            trial.quant_reg    = qc.label_reg;
            trial.size_a_kb    = master_handle->getModelSizeBytes(idx_a)    / 1024.0;
            trial.size_b_kb    = master_handle->getModelSizeBytes(idx_b)    / 1024.0;
            trial.size_lstm_kb = master_handle->getModelSizeBytes(idx_lstm) / 1024.0;
            trial.size_reg_kb  = master_handle->getModelSizeBytes(idx_reg)  / 1024.0;
            // arena used is keyed by INTERNAL slot (0..3), not flash index, and is
            // valid now because init_models() has run AllocateTensors() for each.
            trial.arena_a_kb    = master_handle->getArenaUsedBytes(0) / 1024.0;
            trial.arena_b_kb    = master_handle->getArenaUsedBytes(1) / 1024.0;
            trial.arena_lstm_kb = master_handle->getArenaUsedBytes(2) / 1024.0;
            trial.arena_reg_kb  = master_handle->getArenaUsedBytes(3) / 1024.0;

            if (qc.use_optim_lstm) {
                master_handle->reset_optim_lstm_state();
            }
            int win_count = 0;
            while(in_win < input_trial_len && out_win < output_trial_len)
            {
                const int trial_len_samples = (int)input_trial_len;   // = max_input_size / 8, per-channel length
                const int out_trial_len = (int)(MAIN_Y_SIZES[i] / CONFIG_OUTPUT_CHANNELS);

                WindowStageTimes wt;   // fresh per window
                run_inference_window(
                    master_handle, qc, win_count,
                    x_data, y_data, i,
                    trial_len_samples, out_trial_len,
                    input_win_len, input_size,
                    output_win_len, output_size,
                    intermediate_size, first_out_len,
                    input_trial, output_trial, correct_trial,
                    second_input_ptr, second_input_lengths,
                    &wt, &trial);

                trial.add_window(wt);   // fold this window into avg/max

                // UPDATE
                in_win +=  input_win_len;
                out_win += output_win_len;
                win_count++;
            }

            master_handle->clear_models();

            delete[] second_input_ptr;
            delete[] second_input_lengths;

            trial.trial_total_us = esp_timer_get_time() - startTimeTrial;

            // ---- emit one CSV row for this trial ----
            trial_report_row(final_report, REPORT_MAX, trial);

            ESP_LOGI("TRIAL", "TRIAL %d ENDED, TOTAL TIME: %llu \xCE\xBCs, AVG MSE: %.6e\n\n",
                    i, (unsigned long long)trial.trial_total_us, trial.mse.avg());

            vTaskDelay(10 / portTICK_PERIOD_MS); // Adjust delay as needed for timing
            
            master_handle->clear_optim_lstm();
        }

        // The whole report is now a valid CSV: one header line + one row per trial.
        ESP_LOGI("FINAL REPORT", "%s", final_report);

    }
#endif

#ifdef QUANTIZATION_FIDELITY
    // TRIAL SWEEP TEST
    void run_app_trial_sweep()
    {
        char final_report[REPORT_MAX] = {0};
        ModelFlash* mf = new ModelFlash();

        // One X and one Y pointer per trial now (was per-config).
        const uint8_t** x_data = new const uint8_t*[DATA_TRIAL_COUNT];
        const uint8_t** y_data = new const uint8_t*[DATA_TRIAL_COUNT];

        mf->allocatePointerOnFlashXY_MultiTrial(   // was allocatePointerOnFlashXY
            "benchmark_data",
            X_REGION_BYTE_OFFSET, x_data,
            Y_REGION_BYTE_OFFSET, y_data,
            DATA_TRIAL_COUNT,
            MAIN_X_OFFSETS, MAIN_Y_OFFSETS,
            OFFSET_TYPE::FLOAT32);

        uint32_t max_output_size = 0;
        for (int t = 0; t < DATA_TRIAL_COUNT; t++)
            if (MAIN_Y_SIZES[t] > max_output_size) max_output_size = MAIN_Y_SIZES[t];

        uint32_t max_input_size = 0;
        for (int t = 0; t < DATA_TRIAL_COUNT; t++)
            if (MAIN_X_SIZES[t] > max_input_size) max_input_size = MAIN_X_SIZES[t];

        float* output_trial  = new float[max_output_size];
        float* correct_trial = new float[max_output_size];
        float* input_trial   = new float[max_input_size];

        const uint8_t** models_on_flash = new const uint8_t*[MODEL_COUNT];
        bool success = mf->allocatePointerOnFlash("benchmark_models", models_on_flash,
            MODEL_COUNT, MODEL_OFFSETS, OFFSET_TYPE::INT8);
        if (!success) { ESP_LOGE("MAIN", "Could not initialize mmaped pointers"); return; }

        MasterHandle* master_handle = new MasterHandle(
            MODEL_COUNT, models_on_flash, INPUT_SIZES, OUTPUT_SIZES, MODEL_SIZES);

        trial_report_header(final_report, REPORT_MAX);

        // ---- Config is FIXED (baked into the manifest). Derive its params once. ----
        const int cfg               = DATA_CONFIG_INDEX;
        const int input_win_len     = (int)MAIN_WINDOW_LEN;
        const int input_size        = (int)MAIN_WINDOW_LEN * CONFIG_INPUT_CHANNELS;
        const int output_win_len    = (int)MAIN_OUT_LEN;
        const int output_size       = (int)MAIN_OUT_LEN * CONFIG_OUTPUT_CHANNELS;
        const int intermediate_size = (int)MAIN_INTERMEDIATE_SIZE;
        const int first_out_len     = (int)MAIN_OUT_LEN;
        const int state_size        = 2 * LSTM_FEATURES;
        const int lstm_step         = intermediate_size / LSTM_FEATURES;

        const int idx_a    = cfg*8 + 4;
        const int idx_b    = cfg*8 + 5;
        const int idx_lstm = cfg*8 + 2;
        const int idx_reg  = cfg*8 + 3;

        master_handle->init_models(idx_a, idx_b, idx_lstm, idx_reg,
                                true, final_report, REPORT_MAX);
        master_handle->init_optim_lstm("lstm_weights");

        MasterHandle::QuantConfig qc;
        qc.first_a_is_int8 = true;
        qc.use_optim_lstm  = false;
        qc.use_c16         = false;
        qc.first_a   = &MasterHandle::stage1_first_a_int8; qc.label_a   = "int8";
        qc.first_b   = &MasterHandle::stage2_first_b_int8; qc.label_b   = "int8";
        qc.lstm_step = &MasterHandle::lstm_step_f32;       qc.label_lstm = "f32";
        qc.reg_step  = &MasterHandle::regressor_step_f32;  qc.label_reg  = "f32";

        // Fixed length => derive per-trial sample/out lengths from trial 0 (all equal).
        const int trial_len_samples = (int)(MAIN_X_SIZES[0] / CONFIG_INPUT_CHANNELS);
        const int out_trial_len     = (int)(MAIN_Y_SIZES[0] / CONFIG_OUTPUT_CHANNELS);

        for (int t = 0; t < DATA_TRIAL_COUNT; t++)
        {
            uint64_t startTimeTrial = esp_timer_get_time();

            // Per-trial LSTM state buffer, zeroed (state must not leak across trials).
            float* second_input_ptr = new float[intermediate_size + 2 * state_size]{0};
            const int* second_input_lengths = new const int[3]{
                intermediate_size, state_size, state_size };

            if (qc.use_optim_lstm) master_handle->reset_optim_lstm_state();

            TrialStats trial;
            trial.config       = t;            // report the TRIAL index in this sweep
            trial.window_len   = (long)MAIN_WINDOW_LEN;
            trial.step         = lstm_step;
            trial.quant_a      = qc.label_a;
            trial.quant_b      = qc.label_b;
            trial.quant_lstm   = qc.label_lstm;
            trial.quant_reg    = qc.label_reg;
            trial.size_a_kb    = master_handle->getModelSizeBytes(idx_a)    / 1024.0;
            trial.size_b_kb    = master_handle->getModelSizeBytes(idx_b)    / 1024.0;
            trial.size_lstm_kb = master_handle->getModelSizeBytes(idx_lstm) / 1024.0;
            trial.size_reg_kb  = master_handle->getModelSizeBytes(idx_reg)  / 1024.0;
            trial.arena_a_kb    = master_handle->getArenaUsedBytes(0) / 1024.0;
            trial.arena_b_kb    = master_handle->getArenaUsedBytes(1) / 1024.0;
            trial.arena_lstm_kb = master_handle->getArenaUsedBytes(2) / 1024.0;
            trial.arena_reg_kb  = master_handle->getArenaUsedBytes(3) / 1024.0;

            int in_win = 0, out_win = 0, win_count = 0;
            while (in_win < trial_len_samples && out_win < out_trial_len)
            {
                WindowStageTimes wt;
                run_inference_window(
                    master_handle, qc, win_count,
                    &x_data[t], y_data, t,        // x: this trial; y indexed by t
                    trial_len_samples, out_trial_len,
                    input_win_len, input_size,
                    output_win_len, output_size,
                    intermediate_size, first_out_len,
                    input_trial, output_trial, correct_trial,
                    second_input_ptr, second_input_lengths,
                    &wt, &trial);

                trial.add_window(wt);
                in_win  += input_win_len;
                out_win += output_win_len;
                win_count++;
                vTaskDelay(1 / portTICK_PERIOD_MS);   // feed TWDT / let idle task run
            }

            delete[] second_input_ptr;
            delete[] second_input_lengths;

            trial.trial_total_us = esp_timer_get_time() - startTimeTrial;
            trial_report_row(final_report, REPORT_MAX, trial);

            ESP_LOGI("TRIAL", "SWEEP TRIAL %d/%d ENDED, %llu \xCE\xBCs, AVG NMSE: %.6e",
                    t, DATA_TRIAL_COUNT, (unsigned long long)trial.trial_total_us,
                    trial.nmse.avg());

            vTaskDelay(10 / portTICK_PERIOD_MS);
        }

        master_handle->clear_models();
        master_handle->clear_optim_lstm();

        ESP_LOGI("FINAL REPORT", "%s", final_report);
    }

#endif

#ifdef RANDOM_WEIGHTS_BENCHMARK
    // change input sizes and output size to be double pointers
    void allocateInputandOutputbuffers(float** input_buffer, float** output_buffer, float** correct_buffer,
    int** input_sizes, int** output_sizes, const uint8_t** data,int* total_input_size, int* total_output_size,int index
    , bool usePSRAM)
    {
    
        // Get the size of the total input for the buffer
        for(int j =0; j < BENCHMARK_INPUT_SIZES[index]; j++)
        {
            *total_input_size += SIZES[ MODEL_INPUT_INDEX[index] + j];
        }
    
        // Allocate the entire buffer
        if(usePSRAM)
            *input_buffer = (float*)heap_caps_malloc((*total_input_size) * sizeof(float), MALLOC_CAP_SPIRAM);
        else
            *input_buffer = (float*)malloc((*total_input_size) * sizeof(float));
    
        *input_sizes = new int[BENCHMARK_INPUT_SIZES[index]];
    
        //transfer the data
        int off = 0;
        for(int j =0; j < BENCHMARK_INPUT_SIZES[index]; j++)
        {
            //Allocate data on PSRAM
            (*input_sizes)[j] = SIZES[MODEL_INPUT_INDEX[index] + j];
    
            memcpy( (*input_buffer) + off, data[MODEL_INPUT_INDEX[index] + j],
            SIZES[MODEL_INPUT_INDEX[index] + j] * sizeof(float));
    
            off += SIZES[MODEL_INPUT_INDEX[index] + j];
        }
    
        //Get size of output
        for(int j =0; j < BENCHMARK_OUTPUT_SIZES[index]; j++)
        {
            *total_output_size += SIZES[X_TENSOR_COUNT + MODEL_OUTPUT_INDEX[index] + j];
        }
    
        if(usePSRAM)
        {
            *output_buffer = (float*)heap_caps_malloc(*total_output_size * sizeof(float), MALLOC_CAP_SPIRAM);
            *correct_buffer = (float*)heap_caps_malloc(*total_output_size * sizeof(float), MALLOC_CAP_SPIRAM);
        }else{
            *output_buffer = (float*)malloc(*total_output_size * sizeof(float));
            *correct_buffer = (float*)malloc(*total_output_size * sizeof(float));
        }
    
    
        *output_sizes = new int[BENCHMARK_OUTPUT_SIZES[index]];
    
        off = 0;
        for(int j =0; j < BENCHMARK_OUTPUT_SIZES[index]; j++)
        {
            //Allocate data on PSRAM
            (*output_sizes)[j] = SIZES[X_TENSOR_COUNT + MODEL_OUTPUT_INDEX[index] + j];
            memcpy(*correct_buffer + off, data[X_TENSOR_COUNT + MODEL_OUTPUT_INDEX[index] + j],
                SIZES[X_TENSOR_COUNT + MODEL_OUTPUT_INDEX[index] + j] * sizeof(float));
            // if(j < 5)
            //     printf("ALLOC: (%d)[%0.4f] \n", j, (*correct_buffer + off)[j]);
            off += SIZES[X_TENSOR_COUNT + MODEL_OUTPUT_INDEX[index] + j];
        }
    
    
    }
    static void benchmark_one_quant(BenchmarkHandle* bh,
                                    const uint8_t** data,
                                    int config_index,
                                    bool quant_is_int8,
                                    uint32_t tps,
                                    char* csv, int cap, int* csv_len) {
    
        const int slot = config_index * 2 + (quant_is_int8 ? 1 : 0);
    
        BenchmarkMetrics row;
        bm_init_row(&row);
        row.model_config_index = config_index;
        row.quant_type         = quant_is_int8 ? "int8" : "float32";
        row.model_size_bytes   = BENCHMARK_MODEL_SIZES[slot]; // manifest lives here in main now
        row.ticks_per_second   = tps;
    
    
        float *input_buffer = nullptr, *output_buffer = nullptr, *correct_buffer = nullptr;
        int   *input_sizes  = nullptr, *output_sizes  = nullptr;
        int    total_input = 0, total_output = 0;
    
        allocateInputandOutputbuffers(&input_buffer, &output_buffer, &correct_buffer,
                                    &input_sizes, &output_sizes, data,
                                    &total_input, &total_output, config_index,
                                    /*usePSRAM=*/true);
    
        // No arena_bytes arg -> uses the generous fixed CONFIG_ARENA_SIZE so TFLite
        // can allocate and we can read arena_used() afterwards.
        bh->init_model(slot, /*usePSRAM=*/true,
                    BENCHMARK_INPUT_SIZES[config_index],
                    BENCHMARK_OUTPUT_SIZES[config_index],
                    csv, cap);
    
        size_t measured_arena = 0;
        if (bh->model_ok()) {
            measured_arena       = bh->arena_used();
            row.arena_used_bytes = measured_arena;
    
            row.psram_inf_ticks = bh->run_inference_ticks(input_buffer, input_sizes,
                                                        output_buffer, output_sizes);
            row.psram_inf_us    = bm_ticks_to_us(row.psram_inf_ticks, tps);
            row.psram_mse       = BenchmarkHandle::compute_mse(output_buffer, total_output,
                                                            correct_buffer);
            row.psram_sqnr_db   = BenchmarkHandle::compute_sqnr_db(output_buffer, total_output,
                                                                correct_buffer);
        } else {
            row.sram_skip_reason = "psram_init_failed";
        }
        bh->clear_models();
    
        // SramFitness fit = sram_fitness_check(/*model_copy=*/row.model_size_bytes,
        //                                      /*arena=*/measured_arena);
        // row.sram_eligible = fit.eligible;
        // if (!fit.eligible && row.sram_skip_reason[0] == '\0') {
        //     row.sram_skip_reason = fit.reason;
        // }
    
        SramFitness fit;
        fit.eligible = 1;
        heap_caps_free(input_buffer);   input_buffer   = nullptr;
        heap_caps_free(output_buffer);  output_buffer  = nullptr;
        heap_caps_free(correct_buffer); correct_buffer = nullptr;
        delete[] input_sizes;  input_sizes  = nullptr;
        delete[] output_sizes; output_sizes = nullptr;
        total_input = total_output = 0;
    
        // ----- PASS 2: SRAM, only if eligible. Right-sized arena. -----
        if (fit.eligible) {
            allocateInputandOutputbuffers(&input_buffer, &output_buffer, &correct_buffer,
                                        &input_sizes, &output_sizes, data,
                                        &total_input, &total_output, config_index,
                                        /*usePSRAM=*/false);
    
            // Right-size the SRAM arena from the PSRAM-measured requirement + margin,
            // instead of a fixed 180 KB. This is what stops repeated oversized
            // requests from fragmenting internal RAM and starving later models.
            const int sram_arena = (int)measured_arena + SRAM_FITNESS_MARGIN_BYTES;
    
            bh->init_model(slot, /*usePSRAM=*/false,
                        BENCHMARK_INPUT_SIZES[config_index],
                        BENCHMARK_OUTPUT_SIZES[config_index],
                        csv, cap, /*arena_bytes=*/sram_arena);
    
            if (bh->model_ok()) {
                row.sram_ran       = 1;
                row.sram_inf_ticks = bh->run_inference_ticks(input_buffer, input_sizes,
                                                            output_buffer, output_sizes);
                row.sram_inf_us    = bm_ticks_to_us(row.sram_inf_ticks, tps);
                row.sram_mse       = BenchmarkHandle::compute_mse(output_buffer, total_output,
                                                                correct_buffer);
                row.sram_sqnr_db   = BenchmarkHandle::compute_sqnr_db(output_buffer, total_output,
                                                                    correct_buffer);
            } else {
                row.sram_ran         = 0;
                row.sram_skip_reason = "sram_alloc_failed_after_eligible";
            }
            bh->clear_models();
    
            free(input_buffer);   input_buffer   = nullptr;
            free(output_buffer);  output_buffer  = nullptr;
            free(correct_buffer); correct_buffer = nullptr;
            delete[] input_sizes;  input_sizes  = nullptr;
            delete[] output_sizes; output_sizes = nullptr;
        }
    
        // ----- emit row -----
        *csv_len = (int)strlen(csv);
        *csv_len += bm_csv_row(csv + *csv_len, cap - *csv_len, &row);
    }
    
    void run_testing_benchmark_csv() {
        // The manifest data is injected here — BenchmarkHandle itself never
        // touches BENCHMARK_MODEL_COUNT / OFFSETS / SIZES directly anymore.
        BenchmarkHandle* bh = new BenchmarkHandle("benchmark_models",
                                                BENCHMARK_MODEL_COUNT,
                                                BENCHMARK_MODEL_OFFSETS,
                                                BENCHMARK_MODEL_SIZES);
    
        ModelFlash* mf = new ModelFlash();
        const uint8_t** data = new const uint8_t*[TENSOR_COUNT];
        mf->allocatePointerOnFlash("benchmark_data", data, TENSOR_COUNT, OFFSETS,
                                OFFSET_TYPE::FLOAT32);
    
        const uint32_t tps = tflite::ticks_per_second();
    
        static char csv[REPORT_MAX] = {0};
        int len = bm_csv_header(csv, REPORT_MAX);
    
        for (int i = 0; i < (int)(BENCHMARK_MODEL_COUNT / 2); i++) {
            benchmark_one_quant(bh, data, i, /*int8=*/false, tps, csv, REPORT_MAX, &len);
            vTaskDelay(10 / portTICK_PERIOD_MS);
            benchmark_one_quant(bh, data, i, /*int8=*/true,  tps, csv, REPORT_MAX, &len);
            vTaskDelay(10 / portTICK_PERIOD_MS);
        }
    
        printf("\n===BENCHMARK_CSV_BEGIN===\n%s===BENCHMARK_CSV_END===\n", csv);
    
        delete[] data;
        delete mf;
        delete bh;
    
    }
#endif
extern "C" void app_main(void) {
    printf("ESP32_READY\n");
    ESP_LOGE("MAIN", "FREE_HEAP_CONT_START,%u",
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
 
    //printf("ticks_per_second = %lu\n", (unsigned long)tflite::ticks_per_second());

    #ifdef EMG_MODEL_BENCHMARK
        run_app();
    #endif

    #ifdef RANDOM_WEIGHTS_BENCHMARK
        run_testing_benchmark_csv();
    #endif

    #ifdef QUANTIZATION_FIDELITY
        run_app_trial_sweep();
    #endif
}
 