#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-Dense-Improved/batch_0
// Blob: models_0.bin (126344 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 24

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_25_p25_2_2_None_float32.tflite
    1, // dense_aaab_x1_1_25_p25_2_3_None_float32.tflite
    1, // dense_aaac_x1_1_25_p25_2_4_None_float32.tflite
    1, // dense_aaad_x1_1_25_p25_2_5_None_float32.tflite
    1, // dense_aaae_x1_1_25_p25_2_6_None_float32.tflite
    1, // dense_aaaf_x1_1_25_p25_2_2_None_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaah_x1_1_100_p100_2_2_None_float32.tflite
    1, // dense_aaai_x1_1_150_p150_2_2_None_float32.tflite
    1, // dense_aaaj_x1_1_200_p200_2_2_None_float32.tflite
    1, // dense_aaak_x1_1_25_p25_100_2_None_float32.tflite
    1, // dense_aaal_x1_1_56_p56_224_2_None_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_25_p25_2_2_None_float32.tflite
    1, // dense_aaab_x1_1_25_p25_2_3_None_float32.tflite
    1, // dense_aaac_x1_1_25_p25_2_4_None_float32.tflite
    1, // dense_aaad_x1_1_25_p25_2_5_None_float32.tflite
    1, // dense_aaae_x1_1_25_p25_2_6_None_float32.tflite
    1, // dense_aaaf_x1_1_25_p25_2_2_None_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaah_x1_1_100_p100_2_2_None_float32.tflite
    1, // dense_aaai_x1_1_150_p150_2_2_None_float32.tflite
    1, // dense_aaaj_x1_1_200_p200_2_2_None_float32.tflite
    1, // dense_aaak_x1_1_25_p25_100_2_None_float32.tflite
    1, // dense_aaal_x1_1_56_p56_224_2_None_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // dense_aaaa_x1_1_25_p25_2_2_None_float32.tflite (1120 bytes)
    1120, // dense_aaaa_x1_1_25_p25_2_2_None_int8.tflite (1120 bytes)
    2240, // dense_aaab_x1_1_25_p25_2_3_None_float32.tflite (2384 bytes)
    4624, // dense_aaab_x1_1_25_p25_2_3_None_int8.tflite (1784 bytes)
    6416, // dense_aaac_x1_1_25_p25_2_4_None_float32.tflite (3312 bytes)
    9728, // dense_aaac_x1_1_25_p25_2_4_None_int8.tflite (2456 bytes)
    12192, // dense_aaad_x1_1_25_p25_2_5_None_float32.tflite (4084 bytes)
    16288, // dense_aaad_x1_1_25_p25_2_5_None_int8.tflite (3080 bytes)
    19376, // dense_aaae_x1_1_25_p25_2_6_None_float32.tflite (4904 bytes)
    24288, // dense_aaae_x1_1_25_p25_2_6_None_int8.tflite (3728 bytes)
    28016, // dense_aaaf_x1_1_25_p25_2_2_None_float32.tflite (1120 bytes)
    29136, // dense_aaaf_x1_1_25_p25_2_2_None_int8.tflite (1120 bytes)
    30256, // dense_aaag_x1_1_50_p50_2_2_None_float32.tflite (1320 bytes)
    31584, // dense_aaag_x1_1_50_p50_2_2_None_int8.tflite (1176 bytes)
    32768, // dense_aaah_x1_1_100_p100_2_2_None_float32.tflite (1720 bytes)
    34496, // dense_aaah_x1_1_100_p100_2_2_None_int8.tflite (1272 bytes)
    35776, // dense_aaai_x1_1_150_p150_2_2_None_float32.tflite (2120 bytes)
    37904, // dense_aaai_x1_1_150_p150_2_2_None_int8.tflite (1376 bytes)
    39280, // dense_aaaj_x1_1_200_p200_2_2_None_float32.tflite (2520 bytes)
    41808, // dense_aaaj_x1_1_200_p200_2_2_None_int8.tflite (1472 bytes)
    43280, // dense_aaak_x1_1_25_p25_100_2_None_float32.tflite (10920 bytes)
    54208, // dense_aaak_x1_1_25_p25_100_2_None_int8.tflite (4752 bytes)
    58960, // dense_aaal_x1_1_56_p56_224_2_None_float32.tflite (51096 bytes)
    110064, // dense_aaal_x1_1_56_p56_224_2_None_int8.tflite (16280 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    1120, // dense_aaaa_x1_1_25_p25_2_2_None_float32.tflite
    1120, // dense_aaaa_x1_1_25_p25_2_2_None_int8.tflite
    2384, // dense_aaab_x1_1_25_p25_2_3_None_float32.tflite
    1784, // dense_aaab_x1_1_25_p25_2_3_None_int8.tflite
    3312, // dense_aaac_x1_1_25_p25_2_4_None_float32.tflite
    2456, // dense_aaac_x1_1_25_p25_2_4_None_int8.tflite
    4084, // dense_aaad_x1_1_25_p25_2_5_None_float32.tflite
    3080, // dense_aaad_x1_1_25_p25_2_5_None_int8.tflite
    4904, // dense_aaae_x1_1_25_p25_2_6_None_float32.tflite
    3728, // dense_aaae_x1_1_25_p25_2_6_None_int8.tflite
    1120, // dense_aaaf_x1_1_25_p25_2_2_None_float32.tflite
    1120, // dense_aaaf_x1_1_25_p25_2_2_None_int8.tflite
    1320, // dense_aaag_x1_1_50_p50_2_2_None_float32.tflite
    1176, // dense_aaag_x1_1_50_p50_2_2_None_int8.tflite
    1720, // dense_aaah_x1_1_100_p100_2_2_None_float32.tflite
    1272, // dense_aaah_x1_1_100_p100_2_2_None_int8.tflite
    2120, // dense_aaai_x1_1_150_p150_2_2_None_float32.tflite
    1376, // dense_aaai_x1_1_150_p150_2_2_None_int8.tflite
    2520, // dense_aaaj_x1_1_200_p200_2_2_None_float32.tflite
    1472, // dense_aaaj_x1_1_200_p200_2_2_None_int8.tflite
    10920, // dense_aaak_x1_1_25_p25_100_2_None_float32.tflite
    4752, // dense_aaak_x1_1_25_p25_100_2_None_int8.tflite
    51096, // dense_aaal_x1_1_56_p56_224_2_None_float32.tflite
    16280, // dense_aaal_x1_1_56_p56_224_2_None_int8.tflite
};
