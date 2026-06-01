#ifndef MODEL_FLASH_H
#define MODEL_FLASH_H

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "models_manifest.h"
#include "esp_err.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

class ModelFlash {
public:    
    ModelFlash();
    ~ModelFlash() {}

    bool allocatePointerOnFlash(const char* partition, const uint8_t** out_ptrs, int count);
    uint8_t* allocatePointerOnPSRAM(const int size);
};

#endif // MODEL_FLASH_H