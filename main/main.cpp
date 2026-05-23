#include "master_handle.h"
void run_app()
{
    // Initialize the master handler
    MasterHandle* master_handle = new MasterHandle();

    // We need to now do a while loop
    // Step 1: Pull the data from lsl into the windowed buffer
    // Step 2: Run inference on the windowed buffer which should put the appropriate output into the output buffer
    // Step 3: Push the output data buffer to the LSL outlet

    while (true) {
        master_handle->update_input_window();
        master_handle->run_inference();
        master_handle->push_output_window();
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