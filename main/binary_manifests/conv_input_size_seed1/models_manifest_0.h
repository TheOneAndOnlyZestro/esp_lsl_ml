#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Conv-Input-Size-Seed1/batch_0
// Blob: models_0.bin (39552 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 16

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_5_5_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaab_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaac_x1_15_15_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaad_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaae_x1_25_25_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaaf_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaag_x1_35_35_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_5_5_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaab_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaac_x1_15_15_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaad_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaae_x1_25_25_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaaf_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaag_x1_35_35_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // conv_aaaa_x1_5_5_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    2320, // conv_aaaa_x1_5_5_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    4944, // conv_aaab_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    7264, // conv_aaab_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    9888, // conv_aaac_x1_15_15_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    12208, // conv_aaac_x1_15_15_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    14832, // conv_aaad_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    17152, // conv_aaad_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    19776, // conv_aaae_x1_25_25_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    22096, // conv_aaae_x1_25_25_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    24720, // conv_aaaf_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    27040, // conv_aaaf_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    29664, // conv_aaag_x1_35_35_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    31984, // conv_aaag_x1_35_35_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    34608, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    36928, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    2308, // conv_aaaa_x1_5_5_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaaa_x1_5_5_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaab_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaab_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaac_x1_15_15_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaac_x1_15_15_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaad_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaad_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaae_x1_25_25_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaae_x1_25_25_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaaf_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaaf_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaag_x1_35_35_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaag_x1_35_35_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
};
