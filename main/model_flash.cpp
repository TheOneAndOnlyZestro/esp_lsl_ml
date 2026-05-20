#include "model_flash.h"
#include <stdio.h>
#include <string.h>

// Flash is erased in sector-sized chunks (4 KB on ESP32).
static constexpr size_t SECTOR_SIZE = 4096;

const uint8_t* ModelFlash::allocatePointerOnFlash(const char* partition_label) {
    partition_ = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        ESP_PARTITION_SUBTYPE_ANY,
        partition_label);

    if (partition_ == nullptr) {
        printf("ModelFlash: partition '%s' not found in partition table\n",
               partition_label);
        return nullptr;
    }

    const void* ptr = nullptr;
    
    esp_err_t err = esp_partition_mmap(
        partition_, 0, model_size,
        ESP_PARTITION_MMAP_DATA,
        &ptr, &mmap_handle_);

    if (err != ESP_OK) {
        printf("ModelFlash: mmap failed: %d\n", err);
        return nullptr;
    }
    mapped_ = true;
    mapped_ptr_ = static_cast<const uint8_t*>(ptr);
    return mapped_ptr_;

}

uint8_t* ModelFlash::allocatePointerOnPSRAM(const int size)
{
    if(!esp_psram_is_initialized())
    {
        ESP_LOGE("PSRAM", "PSRAM NOT INTIALIZED");
        return nullptr;
    }
    size_t psram_size = esp_psram_get_size();

    ESP_LOGI("PSRAM", "We want to allocate %d", size);
    ESP_LOGI("PSRAM", "We have %d", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    
    if(size > psram_size)
    {
        ESP_LOGE("PSRAM", "Not enough psram, required: %d, available: %d",
        size, psram_size);
        
        return nullptr;
    }

    void* ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);

    if(ptr == nullptr)
    {
        ESP_LOGE("PSRAM", "MALLOC on Psram Failed");
        
        return nullptr;
    }

    return static_cast<uint8_t*>(ptr);
}

uint32_t ModelFlash::GetModelSize() const 
{
    return model_size;
}


