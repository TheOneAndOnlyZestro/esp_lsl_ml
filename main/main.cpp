#include "master_handle.h"
#include "window_data.h"
void run_app()
{
   
    MasterHandle* master_handle = new MasterHandle("models");
    
    // We need to now do a for loop for all models 

    // Step 1: Pull the data from lsl into the windowed buffer
    // Step 2: Run inference on the windowed buffer which should put the appropriate output into the output buffer
    // Step 3: Push the output data buffer to the LSL outlet

    float* output_window = new float[CONFIG_OUTPUT_CHANNELS * (int)(WINDOW_LENS[9]/2) ];
    for(int i =0; i < (int)(MODEL_COUNT / 4); i++)
    {
        ESP_LOGI("MAIN", "=============WINDOW-(%d)===========", i);
        ESP_LOGI("MAIN", "Inferencing Win (%d) Float32", i);
        master_handle->init_models((i*4), (i*4) + 1);
        // TODO: put a condition so that we only run inference when the input window is full 
        //master_handle->update_input_window();
        master_handle->dual_inference(&(X_DATA[X_OFFSETS[i]]), WINDOW_LENS[i], output_window, (int)(WINDOW_LENS[i] /2));
        //master_handle->push_output_window();
        // Now it only works when the actual output window is filled, which is after the first inference runs. We can change this later to push partial windows if we want.
        //master_handle->reset_for_next_window();
        master_handle->print_output(output_window, (int)(WINDOW_LENS[i] /2), &(Y_DATA[Y_OFFSETS[i]]));

        master_handle->clear_models();

        vTaskDelay(10 / portTICK_PERIOD_MS); // Adjust delay as needed for timing


        // For INT8
        ESP_LOGI("MAIN", "Inferencing Win (%d) Int8", i);
        master_handle->init_models((i*4) + 2, (i*4) + 3);
        // TODO: put a condition so that we only run inference when the input window is full 
        //master_handle->update_input_window();
        master_handle->dual_inference(&(X_DATA[X_OFFSETS[i]]), WINDOW_LENS[i], output_window, (int)(WINDOW_LENS[i] /2));
        //master_handle->push_output_window();
        // Now it only works when the actual output window is filled, which is after the first inference runs. We can change this later to push partial windows if we want.
        //master_handle->reset_for_next_window();
        master_handle->print_output(output_window, (int)(WINDOW_LENS[i] /2), &(Y_DATA[Y_OFFSETS[i]]));

        master_handle->clear_models();

        vTaskDelay(10 / portTICK_PERIOD_MS); // Adjust delay as needed for timing
    }
        
    
}
extern "C" void app_main(void) {
    //Disable watchdog
    printf("ESP32_READY\n");
    ESP_LOGE("MAIN", "FREE_HEAP_CONT_START,%u",
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    
    run_app();
}