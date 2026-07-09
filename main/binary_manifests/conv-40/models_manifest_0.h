#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Conv2D-InputSize-40-Verification/batch_0
// Blob: models_0.bin (4944 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 2

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // conv_aaaa_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    2320, // conv_aaaa_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    2308, // conv_aaaa_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaaa_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
};
