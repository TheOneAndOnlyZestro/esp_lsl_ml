#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-Conv-All/batch_0
// Blob: models_0.bin (97104 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 32

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_2_50_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaab_x1_2_50_p1_10_2_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaac_x1_2_50_p1_10_3_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaad_x1_2_50_p1_10_4_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaae_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaaf_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaag_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaai_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaaj_x1_1_100_p1_10_1_None_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaak_x1_1_100_p1_10_1_None_1_20_pool_1_2_1_1_float32.tflite
    1, // conv_aaal_x1_1_100_p1_10_1_None_1_30_pool_1_2_1_1_float32.tflite
    1, // conv_aaam_x1_1_100_p1_1_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaan_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaao_x1_1_100_p1_20_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaap_x1_1_100_p1_30_1_None_1_5_pool_1_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_2_50_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaab_x1_2_50_p1_10_2_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaac_x1_2_50_p1_10_3_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaad_x1_2_50_p1_10_4_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaae_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaaf_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaag_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    1, // conv_aaai_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaaj_x1_1_100_p1_10_1_None_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaak_x1_1_100_p1_10_1_None_1_20_pool_1_2_1_1_float32.tflite
    1, // conv_aaal_x1_1_100_p1_10_1_None_1_30_pool_1_2_1_1_float32.tflite
    1, // conv_aaam_x1_1_100_p1_1_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaan_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaao_x1_1_100_p1_20_1_None_1_5_pool_1_2_1_1_float32.tflite
    1, // conv_aaap_x1_1_100_p1_30_1_None_1_5_pool_1_2_1_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // conv_aaaa_x1_2_50_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite (2348 bytes)
    2352, // conv_aaaa_x1_2_50_p1_10_1_None_1_5_pool_1_2_1_1_int8.tflite (2632 bytes)
    4992, // conv_aaab_x1_2_50_p1_10_2_None_1_5_pool_1_2_1_1_float32.tflite (3736 bytes)
    8736, // conv_aaab_x1_2_50_p1_10_2_None_1_5_pool_1_2_1_1_int8.tflite (3880 bytes)
    12624, // conv_aaac_x1_2_50_p1_10_3_None_1_5_pool_1_2_1_1_float32.tflite (5012 bytes)
    17648, // conv_aaac_x1_2_50_p1_10_3_None_1_5_pool_1_2_1_1_int8.tflite (4912 bytes)
    22560, // conv_aaad_x1_2_50_p1_10_4_None_1_5_pool_1_2_1_1_float32.tflite (6188 bytes)
    28752, // conv_aaad_x1_2_50_p1_10_4_None_1_5_pool_1_2_1_1_int8.tflite (5920 bytes)
    34672, // conv_aaae_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    36992, // conv_aaae_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    39616, // conv_aaaf_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    41936, // conv_aaaf_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    44560, // conv_aaag_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    46880, // conv_aaag_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    49504, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite (2308 bytes)
    51824, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite (2624 bytes)
    54448, // conv_aaai_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite (2348 bytes)
    56800, // conv_aaai_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_int8.tflite (2632 bytes)
    59440, // conv_aaaj_x1_1_100_p1_10_1_None_1_10_pool_1_2_1_1_float32.tflite (2548 bytes)
    62000, // conv_aaaj_x1_1_100_p1_10_1_None_1_10_pool_1_2_1_1_int8.tflite (2688 bytes)
    64688, // conv_aaak_x1_1_100_p1_10_1_None_1_20_pool_1_2_1_1_float32.tflite (2948 bytes)
    67648, // conv_aaak_x1_1_100_p1_10_1_None_1_20_pool_1_2_1_1_int8.tflite (2784 bytes)
    70432, // conv_aaal_x1_1_100_p1_10_1_None_1_30_pool_1_2_1_1_float32.tflite (3348 bytes)
    73792, // conv_aaal_x1_1_100_p1_10_1_None_1_30_pool_1_2_1_1_int8.tflite (2888 bytes)
    76688, // conv_aaam_x1_1_100_p1_1_1_None_1_5_pool_1_2_1_1_float32.tflite (1780 bytes)
    78480, // conv_aaam_x1_1_100_p1_1_1_None_1_5_pool_1_2_1_1_int8.tflite (1928 bytes)
    80416, // conv_aaan_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite (2348 bytes)
    82768, // conv_aaan_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_int8.tflite (2632 bytes)
    85408, // conv_aaao_x1_1_100_p1_20_1_None_1_5_pool_1_2_1_1_float32.tflite (2588 bytes)
    88000, // conv_aaao_x1_1_100_p1_20_1_None_1_5_pool_1_2_1_1_int8.tflite (2968 bytes)
    90976, // conv_aaap_x1_1_100_p1_30_1_None_1_5_pool_1_2_1_1_float32.tflite (2828 bytes)
    93808, // conv_aaap_x1_1_100_p1_30_1_None_1_5_pool_1_2_1_1_int8.tflite (3296 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    2348, // conv_aaaa_x1_2_50_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    2632, // conv_aaaa_x1_2_50_p1_10_1_None_1_5_pool_1_2_1_1_int8.tflite
    3736, // conv_aaab_x1_2_50_p1_10_2_None_1_5_pool_1_2_1_1_float32.tflite
    3880, // conv_aaab_x1_2_50_p1_10_2_None_1_5_pool_1_2_1_1_int8.tflite
    5012, // conv_aaac_x1_2_50_p1_10_3_None_1_5_pool_1_2_1_1_float32.tflite
    4912, // conv_aaac_x1_2_50_p1_10_3_None_1_5_pool_1_2_1_1_int8.tflite
    6188, // conv_aaad_x1_2_50_p1_10_4_None_1_5_pool_1_2_1_1_float32.tflite
    5920, // conv_aaad_x1_2_50_p1_10_4_None_1_5_pool_1_2_1_1_int8.tflite
    2308, // conv_aaae_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaae_x1_10_10_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaaf_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaaf_x1_20_20_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaag_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaag_x1_30_30_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2308, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_float32.tflite
    2624, // conv_aaah_x1_40_40_p1_10_1_None_2_2_pool_2_2_1_1_int8.tflite
    2348, // conv_aaai_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    2632, // conv_aaai_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_int8.tflite
    2548, // conv_aaaj_x1_1_100_p1_10_1_None_1_10_pool_1_2_1_1_float32.tflite
    2688, // conv_aaaj_x1_1_100_p1_10_1_None_1_10_pool_1_2_1_1_int8.tflite
    2948, // conv_aaak_x1_1_100_p1_10_1_None_1_20_pool_1_2_1_1_float32.tflite
    2784, // conv_aaak_x1_1_100_p1_10_1_None_1_20_pool_1_2_1_1_int8.tflite
    3348, // conv_aaal_x1_1_100_p1_10_1_None_1_30_pool_1_2_1_1_float32.tflite
    2888, // conv_aaal_x1_1_100_p1_10_1_None_1_30_pool_1_2_1_1_int8.tflite
    1780, // conv_aaam_x1_1_100_p1_1_1_None_1_5_pool_1_2_1_1_float32.tflite
    1928, // conv_aaam_x1_1_100_p1_1_1_None_1_5_pool_1_2_1_1_int8.tflite
    2348, // conv_aaan_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_float32.tflite
    2632, // conv_aaan_x1_1_100_p1_10_1_None_1_5_pool_1_2_1_1_int8.tflite
    2588, // conv_aaao_x1_1_100_p1_20_1_None_1_5_pool_1_2_1_1_float32.tflite
    2968, // conv_aaao_x1_1_100_p1_20_1_None_1_5_pool_1_2_1_1_int8.tflite
    2828, // conv_aaap_x1_1_100_p1_30_1_None_1_5_pool_1_2_1_1_float32.tflite
    3296, // conv_aaap_x1_1_100_p1_30_1_None_1_5_pool_1_2_1_1_int8.tflite
};
