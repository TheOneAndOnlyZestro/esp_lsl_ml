#include "model_flash.h"
#include <stdio.h>
#include <string.h>

// Flash is erased in sector-sized chunks (4 KB on ESP32).
static constexpr size_t SECTOR_SIZE = 4096;

esp_err_t ModelFlash::beginWrite(size_t total_size) {
    // If anything is currently mapped, drop it first.
    release();

    partition_ = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        ESP_PARTITION_SUBTYPE_ANY,
        PARTITION_LABEL);

    if (partition_ == nullptr) {
        printf("ModelFlash: partition '%s' not found in partition table\n",
               PARTITION_LABEL);
        return ESP_ERR_NOT_FOUND;
    }

    if (total_size > partition_->size) {
        printf("ModelFlash: model size %u > partition size %u\n",
               (unsigned)total_size, (unsigned)partition_->size);
        return ESP_ERR_INVALID_SIZE;
    }

    // Round erase size UP to a sector boundary.
    const size_t erase_size =
        (total_size + SECTOR_SIZE - 1) & ~(SECTOR_SIZE - 1);

    esp_err_t err = esp_partition_erase_range(partition_, 0, erase_size);
    if (err != ESP_OK) {
        printf("ModelFlash: erase failed: %d\n", err);
        return err;
    }

    total_size_ = total_size;
    written_ = 0;
    return ESP_OK;
}

esp_err_t ModelFlash::writeChunk(const uint8_t* data, size_t len) {
    if (partition_ == nullptr) return ESP_ERR_INVALID_STATE;
    if (written_ + len > total_size_) {
        printf("ModelFlash: chunk overflow (%u + %u > %u)\n",
               (unsigned)written_, (unsigned)len, (unsigned)total_size_);
        return ESP_ERR_INVALID_SIZE;
    }

    esp_err_t err = esp_partition_write(partition_, written_, data, len);
    if (err != ESP_OK) {
        printf("ModelFlash: write at offset %u failed: %d\n",
               (unsigned)written_, err);
        return err;
    }
    written_ += len;
    return ESP_OK;
}

const uint8_t* ModelFlash::finalize() {
    if (partition_ == nullptr) return nullptr;
    if (written_ != total_size_) {
        printf("ModelFlash: finalize called with %u/%u bytes written\n",
               (unsigned)written_, (unsigned)total_size_);
        return nullptr;
    }

    const void* ptr = nullptr;
    esp_err_t err = esp_partition_mmap(
        partition_, 0, total_size_,
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

void ModelFlash::release() {
    if (mapped_) {
        esp_partition_munmap(mmap_handle_);
        mapped_ = false;
        mapped_ptr_ = nullptr;
        mmap_handle_ = 0;
    }
    partition_ = nullptr;
    total_size_ = 0;
    written_ = 0;
}