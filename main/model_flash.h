#ifndef MODEL_FLASH_H
#define MODEL_FLASH_H

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

class ModelFlash {
public:    
    ModelFlash() = default;
    ~ModelFlash() {}

    const uint8_t* allocatePointerOnFlash(const char* partition_label);
    uint8_t* allocatePointerOnPSRAM(const int size);

    uint32_t GetModelSize() const;
private:
    const esp_partition_t* partition_ = nullptr;
    uint32_t model_size = 2821448;
    esp_partition_mmap_handle_t mmap_handle_ = 0;

    const uint8_t* mapped_ptr_ = nullptr;
    bool mapped_ = false;
};

#endif // MODEL_FLASH_H