#include "model_flash.h"


// Flash is erased in sector-sized chunks (4 KB on ESP32).
static constexpr size_t SECTOR_SIZE = 4096;

ModelFlash::ModelFlash() {}
bool ModelFlash::allocatePointerOnFlash(const char* partition, const uint8_t** out_ptrs, int count,
     const uint32_t* model_offsets, const uint32_t* model_sizes, OFFSET_TYPE offset_type) {
    const esp_partition_t* partition_ = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        ESP_PARTITION_SUBTYPE_ANY,
        partition);

    if (partition_ == nullptr) {
        printf("ModelFlash: partition '%s' not found in partition table\n",partition);
        return false;
    }

    const void* base = nullptr;
    
    esp_partition_mmap_handle_t mmap_handle_ = 0;
    esp_err_t err = esp_partition_mmap(
        partition_, 0, partition_->size,
        ESP_PARTITION_MMAP_DATA,
        &base, &mmap_handle_);

    
    if (err != ESP_OK) {
        printf("ModelFlash: mmap failed: %d\n", err);
        return false;
    }

    const uint8_t* base8 = static_cast<const uint8_t*>(base);
    for (int i = 0; i < count; i++) {
        uint32_t current_offset = offset_type == OFFSET_TYPE::INT8 ? model_offsets[i] : model_offsets[i] * sizeof(float);
        ESP_LOGW("FLASH", "Current Offset %llu", (unsigned long long)current_offset);
        out_ptrs[i] = base8 + current_offset;
    }

    for (int i = 0; i < count; i++) {
        uint32_t current_offset = offset_type == OFFSET_TYPE::INT8 ? model_offsets[i] : model_offsets[i] * sizeof(float);
        uint32_t current_size = offset_type == OFFSET_TYPE::INT8 ? model_sizes[i] : model_sizes[i] * sizeof(float);
       
        const uint8_t* p = base8 + current_offset;
        // TFLite flatbuffer has "TFL3" at byte offset 4
        char ident[5] = {0};
        memcpy(ident, p + 4, 4);
        ESP_LOGI("BLOB", "model %d @ off=%u size=%u ident='%s' first4=%02x%02x%02x%02x align=%d",
        i, current_offset, current_size, ident,
        p[0], p[1], p[2], p[3],
        (int)((uintptr_t)p & 0x3));
    }
    return true;

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

