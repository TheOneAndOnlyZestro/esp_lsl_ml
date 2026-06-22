#ifndef MODEL_FLASH_H
#define MODEL_FLASH_H

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include "lstm_nn_manifests/c16/lstm_q.h"
enum OFFSET_TYPE{
    FLOAT32, INT8
};
class ModelFlash {
public:    
    ModelFlash();
    ~ModelFlash() {}

    bool allocatePointerOnFlash(const char* partition, const uint8_t** out_ptrs, int count,
         const uint32_t* model_offsets, OFFSET_TYPE offset_type = OFFSET_TYPE::INT8);
    
    bool allocatePointerOnFlashXY(const char* partition,
        const uint32_t base_x, const uint8_t** out_ptr_x, 
        const uint32_t base_y, const uint8_t** out_ptrs_y,
        int count,
        const uint32_t* x_offsets,
        const uint32_t* y_offsets,
        OFFSET_TYPE offset_type = OFFSET_TYPE::FLOAT32);
    
    
    bool allocatePointerOnFlashOptimLSTMWeightsBiases(const char* partition,
        const int8_t** x0w, const int32_t** x0b,
        const int8_t** h0w, const int32_t** h0b,
        const int8_t** x1w, const int32_t** x1b,
        const int8_t** h1w, const int32_t** h1b);

    uint8_t* allocatePointerOnPSRAM(const int size);
};

#endif // MODEL_FLASH_H