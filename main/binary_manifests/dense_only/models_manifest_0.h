#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-Dense/batch_0
// Blob: models_0.bin (4961000 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 18

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_50_p50_2_1_A_float32.tflite
    1, // dense_aaab_x1_1_100_p100_2_1_A_float32.tflite
    1, // dense_aaac_x1_1_1280_p1280_2_1_A_float32.tflite
    1, // dense_aaad_x1_1_50_p50_2_4_A_float32.tflite
    1, // dense_aaae_x1_1_100_p100_2_4_A_float32.tflite
    1, // dense_aaaf_x1_1_1280_p1280_2_4_A_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_8_A_float32.tflite
    1, // dense_aaah_x1_1_100_p100_2_8_A_float32.tflite
    1, // dense_aaai_x1_1_1280_p1280_2_8_A_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_50_p50_2_1_A_float32.tflite
    1, // dense_aaab_x1_1_100_p100_2_1_A_float32.tflite
    1, // dense_aaac_x1_1_1280_p1280_2_1_A_float32.tflite
    1, // dense_aaad_x1_1_50_p50_2_4_A_float32.tflite
    1, // dense_aaae_x1_1_100_p100_2_4_A_float32.tflite
    1, // dense_aaaf_x1_1_1280_p1280_2_4_A_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_8_A_float32.tflite
    1, // dense_aaah_x1_1_100_p100_2_8_A_float32.tflite
    1, // dense_aaai_x1_1_1280_p1280_2_8_A_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // dense_aaaa_x1_1_50_p50_2_1_A_float32.tflite (560 bytes)
    560, // dense_aaaa_x1_1_50_p50_2_1_A_int8.tflite (560 bytes)
    1120, // dense_aaab_x1_1_100_p100_2_1_A_float32.tflite (560 bytes)
    1680, // dense_aaab_x1_1_100_p100_2_1_A_int8.tflite (560 bytes)
    2240, // dense_aaac_x1_1_1280_p1280_2_1_A_float32.tflite (560 bytes)
    2800, // dense_aaac_x1_1_1280_p1280_2_1_A_int8.tflite (560 bytes)
    3360, // dense_aaad_x1_1_50_p50_2_4_A_float32.tflite (5552 bytes)
    8912, // dense_aaad_x1_1_50_p50_2_4_A_int8.tflite (3304 bytes)
    12224, // dense_aaae_x1_1_100_p100_2_4_A_float32.tflite (13308 bytes)
    25536, // dense_aaae_x1_1_100_p100_2_4_A_int8.tflite (5376 bytes)
    30912, // dense_aaaf_x1_1_1280_p1280_2_4_A_float32.tflite (774844 bytes)
    805760, // dense_aaaf_x1_1_1280_p1280_2_4_A_int8.tflite (197344 bytes)
    1003104, // dense_aaag_x1_1_50_p50_2_8_A_float32.tflite (13876 bytes)
    1016992, // dense_aaag_x1_1_50_p50_2_8_A_int8.tflite (7632 bytes)
    1024624, // dense_aaah_x1_1_100_p100_2_8_A_float32.tflite (37244 bytes)
    1061872, // dense_aaah_x1_1_100_p100_2_8_A_int8.tflite (14064 bytes)
    1075936, // dense_aaai_x1_1_1280_p1280_2_8_A_float32.tflite (3097412 bytes)
    4173360, // dense_aaai_x1_1_1280_p1280_2_8_A_int8.tflite (787640 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    560, // dense_aaaa_x1_1_50_p50_2_1_A_float32.tflite
    560, // dense_aaaa_x1_1_50_p50_2_1_A_int8.tflite
    560, // dense_aaab_x1_1_100_p100_2_1_A_float32.tflite
    560, // dense_aaab_x1_1_100_p100_2_1_A_int8.tflite
    560, // dense_aaac_x1_1_1280_p1280_2_1_A_float32.tflite
    560, // dense_aaac_x1_1_1280_p1280_2_1_A_int8.tflite
    5552, // dense_aaad_x1_1_50_p50_2_4_A_float32.tflite
    3304, // dense_aaad_x1_1_50_p50_2_4_A_int8.tflite
    13308, // dense_aaae_x1_1_100_p100_2_4_A_float32.tflite
    5376, // dense_aaae_x1_1_100_p100_2_4_A_int8.tflite
    774844, // dense_aaaf_x1_1_1280_p1280_2_4_A_float32.tflite
    197344, // dense_aaaf_x1_1_1280_p1280_2_4_A_int8.tflite
    13876, // dense_aaag_x1_1_50_p50_2_8_A_float32.tflite
    7632, // dense_aaag_x1_1_50_p50_2_8_A_int8.tflite
    37244, // dense_aaah_x1_1_100_p100_2_8_A_float32.tflite
    14064, // dense_aaah_x1_1_100_p100_2_8_A_int8.tflite
    3097412, // dense_aaai_x1_1_1280_p1280_2_8_A_float32.tflite
    787640, // dense_aaai_x1_1_1280_p1280_2_8_A_int8.tflite
};
