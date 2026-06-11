#include "master_handle.h"
#include "benchmark_handle.h"
#include "window_data.h"
#include "binary_manifests/conv_all/manifest_0.h"
#include "OptimizedNativeLSTM.h"
#include "weights.h"
#define REPORT_MAX 20000
void run_app()
{
    MasterHandle* master_handle = new MasterHandle("models");
    
    // We need to now do a for loop for all models 

    // Step 1: Pull the data from lsl into the windowed buffer
    // Step 2: Run inference on the windowed buffer which should put the appropriate output into the output buffer
    // Step 3: Push the output data buffer to the LSL outlet

    char final_report[REPORT_MAX] = {0};
    int final_report_size = strlen(final_report);
    float* output_window = new float[CONFIG_OUTPUT_CHANNELS * (int)(WINDOW_LENS[9]/2) ];

    float mse = 0;
    for(int i =0; i < (int)(MODEL_COUNT / 4); i++)
    {
        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Window %ld\nQuant Type: Float32\n", WINDOW_LENS[i]);
        
        ESP_LOGI("MAIN", "=============WINDOW-(%d)===========", i);
        ESP_LOGI("MAIN", "Infenencing Win (%d) Float32", i);
        master_handle->init_models((i*4), (i*4) + 1, final_report, final_report_size);
        // TODO: put a condition so that we only run inference when the input window is full 
        //master_handle->update_input_window();
        master_handle->dual_inference(&(X_DATA[X_OFFSETS[i]]), WINDOW_LENS[i], output_window, (int)(WINDOW_LENS[i] /2), &final_report[0], REPORT_MAX);
        //master_handle->push_output_window();
        // Now it only works when the actual output window is filled, which is after the first inference runs. We can change this later to push partial windows if we want.
        //master_handle->reset_for_next_window();
        mse = master_handle->print_output(output_window, (int)(WINDOW_LENS[i] /2), &(Y_DATA[Y_OFFSETS[i]]));
        master_handle->clear_models();
        
        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: %0.4f\n", mse);

        vTaskDelay(10 / portTICK_PERIOD_MS); // Adjust delay as needed for timing
        
        
        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "Model Window %ld\nQuant Type: Int8\n", WINDOW_LENS[i]);
        
        // For INT8
        ESP_LOGI("MAIN", "Infenencing Win (%d) Int8", i);
        master_handle->init_models((i*4) + 2, (i*4) + 3, final_report, final_report_size);
        // TODO: put a condition so that we only run inference when the input window is full 
        //master_handle->update_input_window();
        master_handle->dual_inference(&(X_DATA[X_OFFSETS[i]]), WINDOW_LENS[i], output_window, (int)(WINDOW_LENS[i] /2), &final_report[0], REPORT_MAX);
        //master_handle->push_output_window();
        // Now it only works when the actual output window is filled, which is after the first inference runs. We can change this later to push partial windows if we want.
        //master_handle->reset_for_next_window();
        mse = master_handle->print_output(output_window, (int)(WINDOW_LENS[i] /2), &(Y_DATA[Y_OFFSETS[i]]));
        master_handle->clear_models();

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "MSE: %0.4f\n", mse);

        final_report_size = strlen(final_report);
        snprintf(final_report + final_report_size, REPORT_MAX - final_report_size, "=================================\n");
        
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
    bool success = mf->allocatePointerOnFlash("benchmark_data", data, TENSOR_COUNT, OFFSETS, SIZES, OFFSET_TYPE::FLOAT32);
    
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
    
    uint64_t startTime_first = esp_timer_get_time();
    lstm->run_inference(x, x_zeropoint, x_scale, y, y_zeropoint, y_scale);
    uint64_t duration_first = esp_timer_get_time() - startTime_first;

    ESP_LOGI("OPTIMIZED", "took  %lld \xCE\xBCs", duration_first);
    // Print Y
    // for(int i =0; i < h_features * 4; i++)
    // {
    //     ESP_LOGI("OPTIMIZED", "(%d)\n" ,y[i]);
    // }

}
extern "C" void app_main(void) {
    //Disable watchdog
    printf("ESP32_READY\n");
    ESP_LOGE("MAIN", "FREE_HEAP_CONT_START,%u",
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    
    //run_optimized();
    run_testing_benchmark();
}