#ifndef MODEL_FLASH_H
#define MODEL_FLASH_H

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "esp_partition.h"

/**
 * Streaming writer for a TFLite model into a dedicated flash partition.
 *
 * Usage:
 *   ModelFlash mf;
 *   mf.beginWrite(total_size);            // erases just enough sectors
 *   while (more_data) mf.writeChunk(p, n);// called for each UART chunk
 *   const uint8_t* ptr = mf.finalize();   // mmaps and returns XIP pointer
 *   // hand `ptr` to TFLite. Keep ModelFlash alive while the model is in use.
 *   mf.release();                         // when done with this model
 */
class ModelFlash {
public:
    static constexpr const char* PARTITION_LABEL = "model";

    ModelFlash() = default;
    ~ModelFlash() { release(); }

    // Prepare the partition for a new model of `total_size` bytes.
    // Erases the minimum number of 4 KB sectors needed.
    esp_err_t beginWrite(size_t total_size);

    // Append a chunk of bytes. Must be called in order, with the total
    // length across all calls equal to total_size passed to beginWrite().
    esp_err_t writeChunk(const uint8_t* data, size_t len);

    // After all chunks are written, mmap the partition and return a
    // pointer the CPU can read from. nullptr on failure.
    const uint8_t* finalize();

    // Drop the mmap handle. Safe to call multiple times.
    void release();

    size_t size() const { return total_size_; }

private:
    const esp_partition_t* partition_ = nullptr;
    esp_partition_mmap_handle_t mmap_handle_ = 0;
    const uint8_t* mapped_ptr_ = nullptr;
    size_t total_size_ = 0;
    size_t written_ = 0;
    bool mapped_ = false;
};

#endif // MODEL_FLASH_H