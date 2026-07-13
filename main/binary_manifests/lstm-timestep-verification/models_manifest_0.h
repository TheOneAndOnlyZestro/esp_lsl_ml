#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/lstm-timestep-verification/batch_0
// Blob: models_0.bin (725400 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 10

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    3, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite
    3, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite
    3, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    3, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite
    3, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite
    3, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite (26396 bytes)
    26400, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite (16352 bytes)
    42752, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite (42432 bytes)
    85184, // lstm_aaab_x5_1_25_p25_25_1_int8.tflite (44560 bytes)
    129744, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite (62312 bytes)
    192064, // lstm_aaac_x10_1_25_p25_25_1_int8.tflite (79512 bytes)
    271584, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite (83408 bytes)
    354992, // lstm_aaad_x15_1_25_p25_25_1_int8.tflite (115912 bytes)
    470912, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite (103396 bytes)
    574320, // lstm_aaae_x20_1_25_p25_25_1_int8.tflite (151080 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    26396, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    16352, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite
    42432, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    44560, // lstm_aaab_x5_1_25_p25_25_1_int8.tflite
    62312, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite
    79512, // lstm_aaac_x10_1_25_p25_25_1_int8.tflite
    83408, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite
    115912, // lstm_aaad_x15_1_25_p25_25_1_int8.tflite
    103396, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite
    151080, // lstm_aaae_x20_1_25_p25_25_1_int8.tflite
};
