#pragma once
#include <stdint.h>

#define BENCHMARK_MODEL_COUNT 6

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // conv2d_float32.tflite (6304 bytes)
    6304, // conv2d_int8.tflite (3984 bytes)
    10288, // dense_float32.tflite (11056 bytes)
    21344, // dense_int8.tflite (4280 bytes)
    25632, // lstm(lin)_1_float32.tflite (106712 bytes)
    132352, // lstm(lin)_1_int8.tflite (44664 bytes)
};
static const int BENCHMARK_INPUT_SIZES[3] = {
    1, 1, 3
};
static const int BENCHMARK_OUTPUT_SIZES[3] = {
    1, 1, 2
};
static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    6304, // conv2d_float32.tflite
    3984, // conv2d_int8.tflite
    11056, // dense_float32.tflite
    4280, // dense_int8.tflite
    106712, // lstm(lin)_1_float32.tflite
    44664, // lstm(lin)_1_int8.tflite
};

