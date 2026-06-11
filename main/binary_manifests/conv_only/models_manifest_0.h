#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-Conv/batch_0
// Blob: models_0.bin (925184 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 20

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_2_50_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaab_x1_2_10_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaac_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaad_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaae_x1_8_100_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaaf_x1_8_200_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaag_x1_50_50_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    1, // conv_aaah_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    1, // conv_aaai_x1_50_50_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite
    1, // conv_aaaj_x1_224_224_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_2_50_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaab_x1_2_10_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaac_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaad_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaae_x1_8_100_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaaf_x1_8_200_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaag_x1_50_50_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    1, // conv_aaah_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    1, // conv_aaai_x1_50_50_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite
    1, // conv_aaaj_x1_224_224_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // conv_aaaa_x1_2_50_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite (2284 bytes)
    2288, // conv_aaaa_x1_2_50_p1_10_1_A_1_5_pool_1_2_1_1_int8.tflite (2568 bytes)
    4864, // conv_aaab_x1_2_10_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite (2284 bytes)
    7152, // conv_aaab_x1_2_10_p1_10_1_A_1_5_pool_1_2_1_1_int8.tflite (2568 bytes)
    9728, // conv_aaac_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite (2484 bytes)
    12224, // conv_aaac_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite (2624 bytes)
    14848, // conv_aaad_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite (2484 bytes)
    17344, // conv_aaad_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite (2624 bytes)
    19968, // conv_aaae_x1_8_100_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite (35836 bytes)
    55808, // conv_aaae_x1_8_100_p1_40_4_A_1_10_pool_1_2_1_1_int8.tflite (15320 bytes)
    71136, // conv_aaaf_x1_8_200_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite (35836 bytes)
    106976, // conv_aaaf_x1_8_200_p1_40_4_A_1_10_pool_1_2_1_1_int8.tflite (15320 bytes)
    122304, // conv_aaag_x1_50_50_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite (6084 bytes)
    128400, // conv_aaag_x1_50_50_p1_10_1_A_10_10_pool_2_2_1_1_int8.tflite (3520 bytes)
    131920, // conv_aaah_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite (6084 bytes)
    138016, // conv_aaah_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_int8.tflite (3520 bytes)
    141536, // conv_aaai_x1_50_50_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite (308356 bytes)
    449904, // conv_aaai_x1_50_50_p1_40_4_A_10_10_pool_2_2_1_1_int8.tflite (83456 bytes)
    533360, // conv_aaaj_x1_224_224_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite (308356 bytes)
    841728, // conv_aaaj_x1_224_224_p1_40_4_A_10_10_pool_2_2_1_1_int8.tflite (83456 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    2284, // conv_aaaa_x1_2_50_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite
    2568, // conv_aaaa_x1_2_50_p1_10_1_A_1_5_pool_1_2_1_1_int8.tflite
    2284, // conv_aaab_x1_2_10_p1_10_1_A_1_5_pool_1_2_1_1_float32.tflite
    2568, // conv_aaab_x1_2_10_p1_10_1_A_1_5_pool_1_2_1_1_int8.tflite
    2484, // conv_aaac_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    2624, // conv_aaac_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite
    2484, // conv_aaad_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    2624, // conv_aaad_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite
    35836, // conv_aaae_x1_8_100_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite
    15320, // conv_aaae_x1_8_100_p1_40_4_A_1_10_pool_1_2_1_1_int8.tflite
    35836, // conv_aaaf_x1_8_200_p1_40_4_A_1_10_pool_1_2_1_1_float32.tflite
    15320, // conv_aaaf_x1_8_200_p1_40_4_A_1_10_pool_1_2_1_1_int8.tflite
    6084, // conv_aaag_x1_50_50_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    3520, // conv_aaag_x1_50_50_p1_10_1_A_10_10_pool_2_2_1_1_int8.tflite
    6084, // conv_aaah_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    3520, // conv_aaah_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_int8.tflite
    308356, // conv_aaai_x1_50_50_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite
    83456, // conv_aaai_x1_50_50_p1_40_4_A_10_10_pool_2_2_1_1_int8.tflite
    308356, // conv_aaaj_x1_224_224_p1_40_4_A_10_10_pool_2_2_1_1_float32.tflite
    83456, // conv_aaaj_x1_224_224_p1_40_4_A_10_10_pool_2_2_1_1_int8.tflite
};
