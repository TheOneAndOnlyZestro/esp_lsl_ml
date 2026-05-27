#include "model_flash.h"
#include <stdio.h>
#include <string.h>

// Flash is erased in sector-sized chunks (4 KB on ESP32).
static constexpr size_t SECTOR_SIZE = 4096;

ModelFlash::ModelFlash(char** partitions, const uint32_t* sizes, size_t count) {
    partition_labels_ = new char*[count];
    partition_sizes_ = new uint32_t[count];
    partition_count_ = count;

    for (size_t i = 0; i < count; i++) {
        size_t label_len = strlen(partitions[i]) + 1;
        partition_labels_[i] = new char[label_len];
        strncpy(partition_labels_[i], partitions[i], label_len);
        partition_sizes_[i] = sizes[i];
    }
}
const uint8_t* ModelFlash::allocatePointerOnFlash(int partition_index) {
    if (partition_index < 0 || partition_index >= partition_count_) {
        ESP_LOGE("ModelFlash", "Invalid partition index: %d", partition_index);
        return nullptr;
    }
    const esp_partition_t* partition_ = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        ESP_PARTITION_SUBTYPE_ANY,
        static_cast<const char*>(partition_labels_[partition_index]));

    if (partition_ == nullptr) {
        printf("ModelFlash: partition '%s' not found in partition table\n",
               partition_labels_[partition_index]);
        return nullptr;
    }

    const void* ptr = nullptr;
    
    esp_partition_mmap_handle_t mmap_handle_ = 0;
    esp_err_t err = esp_partition_mmap(
        partition_, 0, partition_sizes_[partition_index],
        ESP_PARTITION_MMAP_DATA,
        &ptr, &mmap_handle_);

    if (err != ESP_OK) {
        printf("ModelFlash: mmap failed: %d\n", err);
        return nullptr;
    }

    return static_cast<const uint8_t*>(ptr);

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

uint32_t ModelFlash::GetModelSize(int partition_index) const 
{
    if (partition_index < 0 || partition_index >= partition_count_) {
        ESP_LOGE("ModelFlash", "Invalid partition index: %d", partition_index);
        return 0;
    }
    return partition_sizes_[partition_index];
}


