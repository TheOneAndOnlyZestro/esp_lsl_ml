#include "master_handle.h"
void run_app()
{
    char** model_partitions = new char*[2];
    
    model_partitions[0] = new char[8];
    model_partitions[1] = new char[8];

    strcpy(model_partitions[0], "model_1");
    strcpy(model_partitions[1], "model_2");

    const uint32_t model_partition_sizes[] = {
        26376, // Size of first model partition in bytes
        223916  // Size of second model partition in bytes
    };

    const uint32_t input_sizes[] = {
       1,
       3
    };

    const uint32_t output_sizes[] = {
        1,
        3
    };

    MasterHandle* master_handle = new MasterHandle(model_partitions, model_partition_sizes, input_sizes, output_sizes, 2);

    // We need to now do a while loop
    // Step 1: Pull the data from lsl into the windowed buffer
    // Step 2: Run inference on the windowed buffer which should put the appropriate output into the output buffer
    // Step 3: Push the output data buffer to the LSL outlet

    while (true) {

        // TODO: put a condition so that we only run inference when the input window is full 
        master_handle->update_input_window();
        master_handle->dual_inference();
        master_handle->push_output_window();
        // Now it only works when the actual output window is filled, which is after the first inference runs. We can change this later to push partial windows if we want.
        master_handle->reset_for_next_window();
        
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