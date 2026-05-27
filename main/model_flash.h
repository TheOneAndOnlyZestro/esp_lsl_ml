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
    ModelFlash(const char** partitions, const uint32_t* sizes, size_t count);
    ~ModelFlash() {}

    const uint8_t* allocatePointerOnFlash(int partition_index = 0);
    uint8_t* allocatePointerOnPSRAM(const int size);

    uint32_t GetModelSize(int partition_index = 0) const;
private:
    char** partition_labels_;
    uint32_t* partition_sizes_;
    size_t partition_count_;
};

#endif // MODEL_FLASH_H