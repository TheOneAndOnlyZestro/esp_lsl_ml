#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/lstm-hiddensize-verification/batch_0
// Blob: models_0.bin (3152192 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 10

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x1_1_50_p50_50_1_float32.tflite
    3, // lstm_aaac_x1_1_100_p100_100_1_float32.tflite
    3, // lstm_aaad_x1_1_150_p150_150_1_float32.tflite
    3, // lstm_aaae_x1_1_200_p200_200_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x1_1_50_p50_50_1_float32.tflite
    3, // lstm_aaac_x1_1_100_p100_100_1_float32.tflite
    3, // lstm_aaad_x1_1_150_p150_150_1_float32.tflite
    3, // lstm_aaae_x1_1_200_p200_200_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite (26396 bytes)
    26400, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite (16352 bytes)
    42752, // lstm_aaab_x1_1_50_p50_50_1_float32.tflite (86396 bytes)
    129152, // lstm_aaab_x1_1_50_p50_50_1_int8.tflite (33760 bytes)
    162912, // lstm_aaac_x1_1_100_p100_100_1_float32.tflite (326396 bytes)
    489312, // lstm_aaac_x1_1_100_p100_100_1_int8.tflite (98560 bytes)
    587872, // lstm_aaad_x1_1_150_p150_150_1_float32.tflite (726396 bytes)
    1314272, // lstm_aaad_x1_1_150_p150_150_1_int8.tflite (203360 bytes)
    1517632, // lstm_aaae_x1_1_200_p200_200_1_float32.tflite (1286396 bytes)
    2804032, // lstm_aaae_x1_1_200_p200_200_1_int8.tflite (348160 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    26396, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    16352, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite
    86396, // lstm_aaab_x1_1_50_p50_50_1_float32.tflite
    33760, // lstm_aaab_x1_1_50_p50_50_1_int8.tflite
    326396, // lstm_aaac_x1_1_100_p100_100_1_float32.tflite
    98560, // lstm_aaac_x1_1_100_p100_100_1_int8.tflite
    726396, // lstm_aaad_x1_1_150_p150_150_1_float32.tflite
    203360, // lstm_aaad_x1_1_150_p150_150_1_int8.tflite
    1286396, // lstm_aaae_x1_1_200_p200_200_1_float32.tflite
    348160, // lstm_aaae_x1_1_200_p200_200_1_int8.tflite
};
