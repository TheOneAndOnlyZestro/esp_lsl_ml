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
        const uint32_t base_x, const uint8_t** out_ptrs_x, 
        const uint32_t base_y, const uint8_t** out_ptrs_y,
        int count,
        const uint32_t* x_offsets,
        const uint32_t* y_offsets,
        OFFSET_TYPE offset_type = OFFSET_TYPE::FLOAT32);

    uint8_t* allocatePointerOnPSRAM(const int size);
};

#endif // MODEL_FLASH_H