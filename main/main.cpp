#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "MicroInterface/Model.h"
#include "model_flash.h"

#define UART_NUM       UART_NUM_0
#define BUF_SIZE       2048

#define FEATURE_COUNT  9
#define OUTPUT_COUNT   7

// Bumped from 150 KB. The model no longer lives in heap, so the arena
// gets the whole heap to itself. 240 KB leaves comfortable margin for
// WiFi-less FreeRTOS + your stack + UART buffers.
#define ARENA_SIZE     (150 * 1024)

// Chunk size for streaming UART -> flash. 4 KB matches the flash sector
// size and keeps RAM use minimal.
#define UART_CHUNK     4096

Model*       benchmark_model = nullptr;
ModelFlash   model_flash;

int uart_read_fully(uint8_t* buf, int len) {
    int total_read = 0;
    while (total_read < len) {
        int rx = uart_read_bytes(UART_NUM, buf + total_read,
                                 len - total_read,
                                 20 / portTICK_PERIOD_MS);
        if (rx > 0) total_read += rx;
    }
    return total_read;
}

void uart_read_line(char* buf, int max_len) {
    int idx = 0;
    while (idx < max_len - 1) {
        uint8_t c;
        if (uart_read_bytes(UART_NUM, &c, 1, portMAX_DELAY) == 1) {
            if (c == '\n') break;
            if (c != '\r') buf[idx++] = (char)c;
        }
    }
    buf[idx] = '\0';
}

void setup_uart() {
    uart_config_t uart_config = {};
    uart_config.baud_rate           = 115200;
    uart_config.data_bits           = UART_DATA_8_BITS;
    uart_config.parity              = UART_PARITY_DISABLE;
    uart_config.stop_bits           = UART_STOP_BITS_1;
    uart_config.flow_ctrl           = UART_HW_FLOWCTRL_DISABLE;
    uart_config.rx_flow_ctrl_thresh = 122;
    uart_config.source_clk          = UART_SCLK_DEFAULT;

    uart_param_config(UART_NUM, &uart_config);
    uart_driver_install(UART_NUM, BUF_SIZE, 0, 0, NULL, 0);
}

// ----------------------------------------------------------------------
// Stream a model of `model_size` bytes from UART straight into flash.
// Only `UART_CHUNK` bytes live in RAM at any moment.
// ----------------------------------------------------------------------
static bool receive_model_to_flash(int model_size) {
    // 1. Erase enough sectors of the partition.
    if (model_flash.beginWrite(model_size) != ESP_OK) {
        printf("ERR_FLASH_INIT\n");
        return false;
    }

    // 2. Tell PC we are ready for the binary.
    printf("ACK_SIZE\n");

    // 3. Stream chunks UART -> flash.
    static uint8_t chunk[UART_CHUNK]; // static, not stack — large
    int remaining = model_size;
    printf("going to write chunks: %d \n", remaining);
    while (remaining > 0) {
        printf("writing: %d \n", remaining);
        int n = remaining > UART_CHUNK ? UART_CHUNK : remaining;
        uart_read_fully(chunk, n);
        printf("read chunk \n");
        if (model_flash.writeChunk(chunk, n) != ESP_OK) {
            printf("ERR_FLASH_WRITE\n");
            return false;
        }
        remaining -= n;
        printf("CHUNK_OK\n");
    }
    return true;
}

void process_commands() {
    char cmd_buf[32];

    while (1) {
        uart_read_line(cmd_buf, sizeof(cmd_buf));

        // ==========================================================
        // 1. RECEIVE NEW MODEL — Command format: "M<size>"
        //    Now streams to flash instead of buffering in RAM.
        // ==========================================================
        if (cmd_buf[0] == 'M') {
            printf("Model M receive \n");
            int model_size = atoi(&cmd_buf[1]);

            // Tear down any existing model FIRST so its arena is freed
            // before we try to allocate a new one.
            if (benchmark_model != nullptr) {
                delete benchmark_model;
                benchmark_model = nullptr;
            }
            // model_flash.beginWrite() will release the previous mmap.


            if (!receive_model_to_flash(model_size)) {
                continue; // error already printed
            }

            printf("finished writing to chunks to flash");
            // mmap the partition — gives us a CPU-readable pointer.
            const uint8_t* model_ptr = model_flash.finalize();
            if (model_ptr == nullptr) {
                printf("ERR_FLASH_MMAP\n");
                continue;
            }


            // Optional: report free heap so you can see headroom for the arena.
            printf("FREE_HEAP,%u\n",
                   (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));

            benchmark_model = new Model(model_ptr, ARENA_SIZE);
            
            printf("Model Created \n");
            if (benchmark_model->isInitialized()) {
                printf("ACK_M\n");
            } else {

                printf("ERR_INIT\n");
                delete benchmark_model;
                benchmark_model = nullptr;
                model_flash.release();
            }
        }

        // ==========================================================
        // 2. RECEIVE DATA & INFER — Command format: "D<num_floats>"
        //    Unchanged.
        // ==========================================================
        else if (cmd_buf[0] == 'D') {
            int num_floats = atoi(&cmd_buf[1]);
            int bytes_to_read = num_floats * sizeof(float);

            if (benchmark_model == nullptr || !benchmark_model->isInitialized()) {
                printf("ERR_NO_MODEL\n");
                uint8_t dummy[4];
                for (int i = 0; i < num_floats; i++) uart_read_fully(dummy, 4);
                continue;
            }

            float* input_data = (float*)malloc(bytes_to_read);
            if (input_data == nullptr) {
                printf("ERR_DATA_MALLOC\n");
                uint8_t dummy[4];
                for (int i = 0; i < num_floats; i++) uart_read_fully(dummy, 4);
                continue;
            }

            uart_read_fully((uint8_t*)input_data, bytes_to_read);

            float results[OUTPUT_COUNT] = {0};

            uint64_t startTime = esp_timer_get_time();
            bool success = benchmark_model->predict(
                input_data, num_floats, results, OUTPUT_COUNT);
            uint64_t duration = esp_timer_get_time() - startTime;

            if (success) {
                printf("RES,%lld,%d", duration,
                       benchmark_model->getArenaUsedBytes());
                for (int i = 0; i < OUTPUT_COUNT; i++) {
                    printf(",%0.4f", results[i]);
                }
                printf("\n");
            } else {
                printf("ERR_INFERENCE\n");
            }

            free(input_data);
        }
    }
}

extern "C" void app_main(void) {
    setup_uart();
    printf("ESP32_READY\n");
    process_commands();
}
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "esp_timer.h"
// #include "driver/uart.h"

// #include "MicroInterface/Model.h"

// #define UART_NUM UART_NUM_0
// #define BUF_SIZE 2048

// #define FEATURE_COUNT 9
// #define OUTPUT_COUNT 7

// Model* benchmark_model = nullptr;
// uint8_t* model_buffer = nullptr; 

// int uart_read_fully(uint8_t* buf, int len) {
//     int total_read = 0;
//     while (total_read < len) {
//         int rx = uart_read_bytes(UART_NUM, buf + total_read, len - total_read, 20 / portTICK_PERIOD_MS);
//         if (rx > 0) {
//             total_read += rx;
//         }
//     }
//     return total_read;
// }

// void uart_read_line(char* buf, int max_len) {
//     int idx = 0;
//     while (idx < max_len - 1) {
//         uint8_t c;
//         if (uart_read_bytes(UART_NUM, &c, 1, portMAX_DELAY) == 1) {
//             if (c == '\n') break;
//             if (c != '\r') buf[idx++] = (char)c;
//         }
//     }
//     buf[idx] = '\0';
// }

// void setup_uart() {
//     uart_config_t uart_config = {}; 
//     uart_config.baud_rate = 115200; 
//     uart_config.data_bits = UART_DATA_8_BITS;
//     uart_config.parity    = UART_PARITY_DISABLE;
//     uart_config.stop_bits = UART_STOP_BITS_1;
//     uart_config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
//     uart_config.rx_flow_ctrl_thresh = 122;
//     uart_config.source_clk = UART_SCLK_DEFAULT;

//     uart_param_config(UART_NUM, &uart_config);
//     uart_driver_install(UART_NUM, BUF_SIZE, 0, 0, NULL, 0);
// }

// void process_commands() {
//     char cmd_buf[32];
    
//     while (1) {
//         uart_read_line(cmd_buf, sizeof(cmd_buf));

//         // ==========================================
//         // 1. RECEIVE NEW MODEL (Command format: "M<size>")
//         // ==========================================
//         if (cmd_buf[0] == 'M') {
//             int model_size = atoi(&cmd_buf[1]);
            
//             if (benchmark_model != nullptr) {
//                 delete benchmark_model;
//                 benchmark_model = nullptr;
//             }
//             if (model_buffer != nullptr) {
//                 free(model_buffer);
//                 model_buffer = nullptr;
//             }

//             model_buffer = (uint8_t*)malloc(model_size);
//             if (model_buffer == nullptr) {
//                 // Tell PC we DO NOT have memory, so PC aborts sending binary
//                 printf("ERR_SIZE,%d\n", model_size);
//                 continue; 
//             }

//             // Tell PC we are ready to receive the binary
//             printf("ACK_SIZE\n");

//             // Read the binary stream safely
//             uart_read_fully(model_buffer, model_size);

//             benchmark_model = new Model(model_buffer, 150 * 1024);

//             if (benchmark_model->isInitialized()) {
//                 printf("ACK_M\n"); 
//             } else {
//                 printf("ERR_INIT\n");
//             }
//         }
        
//         // ==========================================
//         // 2. RECEIVE DATA & INFER (Command format: "D<num_floats>")
//         // ==========================================
//         else if (cmd_buf[0] == 'D') {
//             int num_floats = atoi(&cmd_buf[1]);
//             int bytes_to_read = num_floats * sizeof(float);
            
//             if (benchmark_model == nullptr || !benchmark_model->isInitialized()) {
//                 printf("ERR_NO_MODEL\n");
//                 uint8_t dummy[4];
//                 for(int i=0; i<num_floats; i++) uart_read_fully(dummy, 4);
//                 continue;
//             }

//             float* input_data = (float*)malloc(bytes_to_read);
//             if (input_data == nullptr) {
//                 printf("ERR_DATA_MALLOC\n");
//                 uint8_t dummy[4];
//                 for(int i=0; i<num_floats; i++) uart_read_fully(dummy, 4);
//                 continue;
//             }

//             uart_read_fully((uint8_t*)input_data, bytes_to_read);

//             float results[OUTPUT_COUNT] = {0};

//             uint64_t startTime = esp_timer_get_time();
//             bool success = benchmark_model->predict(input_data, num_floats, results, OUTPUT_COUNT);
//             uint64_t duration = esp_timer_get_time() - startTime;

//             if (success) {
//                 printf("RES,%lld,%d", duration, benchmark_model->getArenaUsedBytes());
//                 for(int i = 0; i < OUTPUT_COUNT; i++) {
//                     printf(",%0.4f", results[i]);
//                 }
//                 printf("\n");
//             } else {
//                 printf("ERR_INFERENCE\n");
//             }

//             free(input_data);
//         }
//     }
// }

// extern "C" void app_main(void) {
//     setup_uart();
//     printf("ESP32_READY\n");
//     process_commands();
// }