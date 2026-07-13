#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/lstm-hiddensize-verification2/batch_0
// Blob: models_0.bin (3109440 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 8

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_50_p50_50_1_float32.tflite
    3, // lstm_aaab_x1_1_100_p100_100_1_float32.tflite
    3, // lstm_aaac_x1_1_150_p150_150_1_float32.tflite
    3, // lstm_aaad_x1_1_200_p200_200_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_50_p50_50_1_float32.tflite
    3, // lstm_aaab_x1_1_100_p100_100_1_float32.tflite
    3, // lstm_aaac_x1_1_150_p150_150_1_float32.tflite
    3, // lstm_aaad_x1_1_200_p200_200_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // lstm_aaaa_x1_1_50_p50_50_1_float32.tflite (86396 bytes)
    86400, // lstm_aaaa_x1_1_50_p50_50_1_int8.tflite (33760 bytes)
    120160, // lstm_aaab_x1_1_100_p100_100_1_float32.tflite (326396 bytes)
    446560, // lstm_aaab_x1_1_100_p100_100_1_int8.tflite (98560 bytes)
    545120, // lstm_aaac_x1_1_150_p150_150_1_float32.tflite (726396 bytes)
    1271520, // lstm_aaac_x1_1_150_p150_150_1_int8.tflite (203360 bytes)
    1474880, // lstm_aaad_x1_1_200_p200_200_1_float32.tflite (1286396 bytes)
    2761280, // lstm_aaad_x1_1_200_p200_200_1_int8.tflite (348160 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    86396, // lstm_aaaa_x1_1_50_p50_50_1_float32.tflite
    33760, // lstm_aaaa_x1_1_50_p50_50_1_int8.tflite
    326396, // lstm_aaab_x1_1_100_p100_100_1_float32.tflite
    98560, // lstm_aaab_x1_1_100_p100_100_1_int8.tflite
    726396, // lstm_aaac_x1_1_150_p150_150_1_float32.tflite
    203360, // lstm_aaac_x1_1_150_p150_150_1_int8.tflite
    1286396, // lstm_aaad_x1_1_200_p200_200_1_float32.tflite
    348160, // lstm_aaad_x1_1_200_p200_200_1_int8.tflite
};
