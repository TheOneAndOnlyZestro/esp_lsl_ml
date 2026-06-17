#include "master_handle.h"
#include "benchmark_handle.h"
//#include "window_data.h"
#include "binary_manifests/dense_for_espnn/manifest_0.h"
#include "OptimizedNativeLSTM.h"
#include "weights.h"

#include "lsl_handle.h"
#include "binary_manifests/tri_models/manifest_0.h"
#include "binary_manifests/tri_models/models_manifest_0.h"
#define REPORT_MAX 20000

void run_one_app()
{
    char final_report[REPORT_MAX] = {0};
    int final_report_size = strlen(final_report);
    ModelFlash* mf = new ModelFlash();

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
    
    int model_index = 0;
    //Initialize model
    const int input_size        = (int)MAIN_X_SIZES[model_index];
    const int output_size       = (int)MAIN_Y_SIZES[model_index];
    const int intermediate_size = (int)MAIN_INTERMEDIATE_SIZE[model_index];
    const int first_out_len     = (int)MAIN_OUT_LENS[model_index];
    const int out_len           = output_size / CONFIG_OUTPUT_CHANNELS;
    
    LSLHandle* main_handle = new LSLHandle((int)MAIN_WINDOW_LENS[0]);
    float* output_window = new float[output_size];
    float* input_window = main_handle->expose_window();

    final_report_size = strlen(final_report);
    snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Window %ld\nQuant Type: Float32\n", MAIN_WINDOW_LENS[model_index]);

    ESP_LOGI("MAIN", "Infenencing Config (%d) Float32", model_index);
    master_handle->init_models(model_index*6 + 3, 
    model_index*6 + 4, 
    model_index*6 + 5,
    true, final_report, final_report_size);
    //float first, float second, int first,int second

    while (true)
    {
        //allocate data from flash onto sram
        while(main_handle->add_to_window()){}
        

        uint64_t startTime_first = esp_timer_get_time();
        master_handle->dual_inference(&input_window[0], input_size,
                                        &output_window[0], output_size,
                                        intermediate_size, first_out_len,
                                        nullptr, REPORT_MAX);

        uint64_t duration_first = esp_timer_get_time() - startTime_first;

        ESP_LOGW("INFERENCE", "infernce: %lld \xCE\xBCs\n", duration_first);

        master_handle->display_output(output_window, output_size);    

    }


}
void run_app()
{
    
    // We need to now do a for loop for all models 
    
    // Step 1: Pull the data from lsl into the windowed buffer
    // Step 2: Run inference on the windowed buffer which should put the appropriate output into the output buffer
    // Step 3: Push the output data buffer to the LSL outlet
    
    char final_report[REPORT_MAX] = {0};
    int final_report_size = strlen(final_report);
    ModelFlash* mf = new ModelFlash();
    const uint8_t** x_data = new const uint8_t*[DATA_CONFIG_COUNT];
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
    
    float mse = 0;
    
    uint32_t max_output_size = 0;
    for (int i = 0; i < DATA_CONFIG_COUNT; i++) {
        if (MAIN_Y_SIZES[i] > max_output_size) max_output_size = MAIN_Y_SIZES[i];
    }

    float* output_window = new float[max_output_size];
    float* correct_window = new float[max_output_size];
    uint32_t max_input_size = 0;
    for (int i = 0; i < DATA_CONFIG_COUNT; i++) {
        if (MAIN_X_SIZES[i] > max_input_size) max_input_size = MAIN_X_SIZES[i];
    }

    float* input_window = new float[max_input_size];
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

    for(int i =0; i < CONFIG_COUNT; i++)
    {
        const int input_size        = (int)MAIN_X_SIZES[i];
        const int output_size       = (int)MAIN_Y_SIZES[i];
        const int intermediate_size = (int)MAIN_INTERMEDIATE_SIZE[i];
        const int first_out_len     = (int)MAIN_OUT_LENS[i];
        const int out_len           = output_size / CONFIG_OUTPUT_CHANNELS;
        
        memcpy(input_window, x_data[i], input_size * sizeof(float));
        memcpy(correct_window, y_data[i], output_size * sizeof(float));

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Window %ld\nQuant Type: Float32\n", MAIN_WINDOW_LENS[i]);
    
        ESP_LOGI("MAIN", "Infenencing Config (%d) INT First 2 models and Float32 last", i);
        master_handle->init_models(
        i*6 + 3,
        i*6 + 4,
        i*6 + 5,
        true, final_report, final_report_size);

        //allocate data from flash onto sram
        master_handle->dual_inference(&input_window[0], input_size,
                                      &output_window[0], output_size,
                                      intermediate_size, first_out_len,
                                      &final_report[0], REPORT_MAX);
        
        mse = master_handle->print_output(output_window, output_size, &correct_window[0]);
        
        master_handle->clear_models();
        
        final_report_size = strlen(final_report);

        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: %0.4f\n\n", mse);

        vTaskDelay(10 / portTICK_PERIOD_MS); // Adjust delay as needed for timing
        
    }
    
    ESP_LOGI("FINAL REPORT", "%s", final_report);
    
}

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


void run_testing_benchmark()
{
    BenchmarkHandle* benchmark_handle = new BenchmarkHandle("benchmark_models");
    
    // We need to now do a for loop for all models 

    // Step 1: Pull the data from lsl into the windowed buffer
    // Step 2: Run inference on the windowed buffer which should put the appropriate output into the output buffer
    // Step 3: Push the output data buffer to the LSL outlet    
    char final_report[REPORT_MAX] = {0};
    int final_report_size = strlen(final_report);

    ModelFlash* mf = new ModelFlash();

    const uint8_t** data = new const uint8_t*[TENSOR_COUNT];
    // need to convert OFFSET from float count to actual byte size
    bool success = mf->allocatePointerOnFlash("benchmark_data", data, TENSOR_COUNT, OFFSETS, OFFSET_TYPE::FLOAT32);
    
    float mse = 0;

    float* input_buffer;
    float* output_buffer;
    float* correct_buffer;
    int* input_sizes;
    int* output_sizes;

    int total_input_size = 0;
    int total_output_size = 0;

    for(int i =0; i < (int)(BENCHMARK_MODEL_COUNT / 2); i++)
    {
        int index = (i * 2);
        allocateInputandOutputbuffers(&input_buffer, &output_buffer, &correct_buffer, &input_sizes, &output_sizes,
             data, &total_input_size, &total_output_size, i, true);

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Index %d\nQuant Type: Float32\n", i);
        
        benchmark_handle->init_model(index, true, BENCHMARK_INPUT_SIZES[i], BENCHMARK_OUTPUT_SIZES[i], final_report, final_report_size);

        benchmark_handle->run_inference(input_buffer, input_sizes, output_buffer, output_sizes, final_report, final_report_size);
       
        mse = benchmark_handle->print_output(output_buffer, total_output_size, correct_buffer);
        
        benchmark_handle->clear_models();

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: %0.4f\n\n", mse);
        
        // Now for int8
        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Index %d\nQuant Type: Int8\n", i);
        
        benchmark_handle->init_model(index + 1, true, BENCHMARK_INPUT_SIZES[i], BENCHMARK_OUTPUT_SIZES[i], final_report, final_report_size);

        benchmark_handle->run_inference(input_buffer, input_sizes, output_buffer, output_sizes, final_report, final_report_size);
       
        mse = benchmark_handle->print_output(output_buffer, total_output_size, correct_buffer);

        benchmark_handle->clear_models();

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: %0.4f\n\n", mse);

        heap_caps_free(input_buffer);
        input_buffer = nullptr;
        heap_caps_free(output_buffer);
        output_buffer = nullptr;
        heap_caps_free(correct_buffer);
        correct_buffer = nullptr;

        delete[] input_sizes;
        input_sizes = nullptr;
        delete[] output_sizes;
        output_sizes = nullptr;

        total_input_size = 0;
        total_output_size = 0;
        
        if (BENCHMARK_MODEL_SIZES[index] > 150000 || BENCHMARK_MODEL_SIZES[index + 1] > 150000)
        {
            // Don't allocate on SRAM
            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Index %d\nQuant Type: Float32\n", i);
            
            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model Model_Size: TOO_LARGE B\n");

            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model Arena_Size: TOO_LARGE B\n");

            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model init: TOO_LARGE \xCE\xBCs, TOO_LARGE ms\n");

            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model Inf: TOO_LARGE \xCE\xBCs, TOO_LARGE ms\n");
            
            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: TOO_LARGE \n\n");

            // Don't allocate on SRAM
            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Index %d\nQuant Type: Int8\n", i);
            
            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model Model_Size: TOO_LARGE B\n");

            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model Arena_Size: TOO_LARGE B\n");

            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model init: TOO_LARGE \xCE\xBCs, TOO_LARGE ms\n");

            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "0_Model Inf: TOO_LARGE \xCE\xBCs, TOO_LARGE ms\n");
            
            final_report_size = strlen(final_report);
            snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: TOO_LARGE \n\n");
            
            continue;
        }
        // TEST with normal ram
        allocateInputandOutputbuffers(&input_buffer, &output_buffer, &correct_buffer, &input_sizes, &output_sizes,
             data, &total_input_size, &total_output_size, i, false);

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Index %d\nQuant Type: Float32\n", i);
        
        benchmark_handle->init_model(index, false, BENCHMARK_INPUT_SIZES[i], BENCHMARK_OUTPUT_SIZES[i], final_report, final_report_size);

        benchmark_handle->run_inference(input_buffer, input_sizes, output_buffer, output_sizes, final_report, final_report_size);
       
        mse = benchmark_handle->print_output(output_buffer, total_output_size, correct_buffer);
        
        benchmark_handle->clear_models();

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: %0.4f\n\n", mse);
        vTaskDelay(10 / portTICK_PERIOD_MS); // Adjust delay as needed for timing

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Index %d\nQuant Type: Int8\n", i);
        
        benchmark_handle->init_model(index + 1, false, BENCHMARK_INPUT_SIZES[i], BENCHMARK_OUTPUT_SIZES[i], final_report, final_report_size);

        benchmark_handle->run_inference(input_buffer, input_sizes, output_buffer, output_sizes, final_report, final_report_size);
       
        mse = benchmark_handle->print_output(output_buffer, total_output_size, correct_buffer);

        benchmark_handle->clear_models();

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: %0.4f\n\n", mse);

        free(input_buffer);
        input_buffer = nullptr;
        free(output_buffer);
        output_buffer = nullptr;
        free(correct_buffer);
        correct_buffer = nullptr;

        delete[] input_sizes;
        input_sizes = nullptr;
        delete[] output_sizes;
        output_sizes = nullptr;

        total_input_size = 0;
        total_output_size = 0;
    }

    ESP_LOGI("FINAL REPORT", "%s", final_report);
}

void run_optimized()
{
    OptimizedNativeLSTM* lstm = new OptimizedNativeLSTM();

    int x_features = 25;
    int32_t x_zeropoint = 14;
    float x_scale = 0.010052992030978203;


    int h_features = 25;
    const int8_t* x = new int8_t[x_features]{
        56,  118,   86,  -82, -128,   91,   26,   21,   80,   94,  127,
             8, -113,  -21,  -67,  113,   94,    6,  113,  -37,   94,  -55,
            25,  -31,   86
    };

    int8_t* y = new int8_t[h_features * 4];
    int32_t y_zeropoint = 8;
    float y_scale = 0.021355733275413513;

    lstm->set_x_h_weights_and_bias(&weights[0], nullptr,
         x_features, weights_scale,
          nullptr, &weights[0], nullptr,
           h_features, weights_scale, nullptr);  
    
    lstm->calculate_per_ch_M(x_scale, y_scale);
    uint64_t startTime_first = esp_timer_get_time();
    lstm->run_inference(x, x_zeropoint, y, y_zeropoint);
    uint64_t duration_first = esp_timer_get_time() - startTime_first;

    ESP_LOGI("OPTIMIZED", "took  %lld \xCE\xBCs", duration_first);
    // Print Y
    // for(int i =0; i < h_features * 4; i++)
    // {
    //     ESP_LOGI("OPTIMIZED", "(%d)\n" ,y[i]);
    // }

}

void test_lsl()
{
    LSLHandle* main_handle = new LSLHandle((int)MAIN_WINDOW_LENS[0]);

    while (true)
    {
        main_handle->add_to_window();
        main_handle->print_current_window();
    }
    
}
extern "C" void app_main(void) {
    //Disable watchdog
    printf("ESP32_READY\n");
    ESP_LOGE("MAIN", "FREE_HEAP_CONT_START,%u",
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    
    
    //run_testing_benchmark();
    //run_optimized();

    run_app();

    //run_one_app();
}