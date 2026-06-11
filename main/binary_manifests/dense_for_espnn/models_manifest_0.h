#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-Dense-Against-ESP-NN/batch_0
// Blob: models_0.bin (15680 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 2

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_25_p25_100_2_None_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_25_p25_100_2_None_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // dense_aaaa_x1_1_25_p25_100_2_None_float32.tflite (10920 bytes)
    10928, // dense_aaaa_x1_1_25_p25_100_2_None_int8.tflite (4752 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    10920, // dense_aaaa_x1_1_25_p25_100_2_None_float32.tflite
    4752, // dense_aaaa_x1_1_25_p25_100_2_None_int8.tflite
};
