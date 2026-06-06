#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute/batch_0
// Blob: models_0.bin (6454256 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 36

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaab_x1_8_100_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaac_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaad_x1_8_100_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite
    1, // conv_aaae_x1_8_200_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite
    1, // conv_aaaf_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_1_A_float32.tflite
    1, // dense_aaah_x1_1_1280_p1280_2_1_A_float32.tflite
    1, // dense_aaai_x1_1_50_p50_8_4_A_float32.tflite
    1, // dense_aaaj_x1_1_448_p448_8_4_A_float32.tflite
    3, // lstm_aaak_x1_1_10_p10_10_1_float32.tflite
    3, // lstm_aaal_x1_1_60_p60_60_1_float32.tflite
    3, // lstm_aaam_x1_1_200_p200_200_1_float32.tflite
    3, // lstm_aaan_x1_1_10_p10_10_2_float32.tflite
    3, // lstm_aaao_x1_1_60_p60_60_2_float32.tflite
    3, // lstm_aaap_x1_1_200_p200_200_2_float32.tflite
    3, // lstm_aaaq_x5_1_10_p10_10_2_float32.tflite
    3, // lstm_aaar_x5_1_60_p60_60_2_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // conv_aaaa_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaab_x1_8_100_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaac_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    1, // conv_aaad_x1_8_100_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite
    1, // conv_aaae_x1_8_200_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite
    1, // conv_aaaf_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_1_A_float32.tflite
    1, // dense_aaah_x1_1_1280_p1280_2_1_A_float32.tflite
    1, // dense_aaai_x1_1_50_p50_8_4_A_float32.tflite
    1, // dense_aaaj_x1_1_448_p448_8_4_A_float32.tflite
    3, // lstm_aaak_x1_1_10_p10_10_1_float32.tflite
    3, // lstm_aaal_x1_1_60_p60_60_1_float32.tflite
    3, // lstm_aaam_x1_1_200_p200_200_1_float32.tflite
    3, // lstm_aaan_x1_1_10_p10_10_2_float32.tflite
    3, // lstm_aaao_x1_1_60_p60_60_2_float32.tflite
    3, // lstm_aaap_x1_1_200_p200_200_2_float32.tflite
    3, // lstm_aaaq_x5_1_10_p10_10_2_float32.tflite
    3, // lstm_aaar_x5_1_60_p60_60_2_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // conv_aaaa_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite (2484 bytes)
    2496, // conv_aaaa_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite (2624 bytes)
    5120, // conv_aaab_x1_8_100_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite (2484 bytes)
    7616, // conv_aaab_x1_8_100_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite (2624 bytes)
    10240, // conv_aaac_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite (2484 bytes)
    12736, // conv_aaac_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite (2624 bytes)
    15360, // conv_aaad_x1_8_100_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite (16032 bytes)
    31392, // conv_aaad_x1_8_100_p1_30_2_A_1_20_pool_1_2_1_1_int8.tflite (7960 bytes)
    39360, // conv_aaae_x1_8_200_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite (16032 bytes)
    55392, // conv_aaae_x1_8_200_p1_30_2_A_1_20_pool_1_2_1_1_int8.tflite (7960 bytes)
    63360, // conv_aaaf_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite (6084 bytes)
    69456, // conv_aaaf_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_int8.tflite (3520 bytes)
    72976, // dense_aaag_x1_1_50_p50_2_1_A_float32.tflite (560 bytes)
    73536, // dense_aaag_x1_1_50_p50_2_1_A_int8.tflite (560 bytes)
    74096, // dense_aaah_x1_1_1280_p1280_2_1_A_float32.tflite (560 bytes)
    74656, // dense_aaah_x1_1_1280_p1280_2_1_A_int8.tflite (560 bytes)
    75216, // dense_aaai_x1_1_50_p50_8_4_A_float32.tflite (9196 bytes)
    84416, // dense_aaai_x1_1_50_p50_8_4_A_int8.tflite (4512 bytes)
    88928, // dense_aaaj_x1_1_448_p448_8_4_A_float32.tflite (226860 bytes)
    315792, // dense_aaaj_x1_1_448_p448_8_4_A_int8.tflite (60208 bytes)
    376000, // lstm_aaak_x1_1_10_p10_10_1_float32.tflite (9068 bytes)
    385072, // lstm_aaak_x1_1_10_p10_10_1_int8.tflite (10208 bytes)
    395280, // lstm_aaal_x1_1_60_p60_60_1_float32.tflite (121068 bytes)
    516352, // lstm_aaal_x1_1_60_p60_60_1_int8.tflite (43008 bytes)
    559360, // lstm_aaam_x1_1_200_p200_200_1_float32.tflite (1285868 bytes)
    1845232, // lstm_aaam_x1_1_200_p200_200_1_int8.tflite (347648 bytes)
    2192880, // lstm_aaan_x1_1_10_p10_10_2_float32.tflite (20588 bytes)
    2213472, // lstm_aaan_x1_1_10_p10_10_2_int8.tflite (23856 bytes)
    2237328, // lstm_aaao_x1_1_60_p60_60_2_float32.tflite (244588 bytes)
    2481920, // lstm_aaao_x1_1_60_p60_60_2_int8.tflite (89456 bytes)
    2571376, // lstm_aaap_x1_1_200_p200_200_2_float32.tflite (2574188 bytes)
    5145568, // lstm_aaap_x1_1_200_p200_200_2_int8.tflite (698736 bytes)
    5844304, // lstm_aaaq_x5_1_10_p10_10_2_float32.tflite (66732 bytes)
    5911040, // lstm_aaaq_x5_1_10_p10_10_2_int8.tflite (93440 bytes)
    6004480, // lstm_aaar_x5_1_60_p60_60_2_float32.tflite (290732 bytes)
    6295216, // lstm_aaar_x5_1_60_p60_60_2_int8.tflite (159040 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    2484, // conv_aaaa_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    2624, // conv_aaaa_x1_8_50_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite
    2484, // conv_aaab_x1_8_100_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    2624, // conv_aaab_x1_8_100_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite
    2484, // conv_aaac_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_float32.tflite
    2624, // conv_aaac_x1_8_200_p1_10_1_A_1_10_pool_1_2_1_1_int8.tflite
    16032, // conv_aaad_x1_8_100_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite
    7960, // conv_aaad_x1_8_100_p1_30_2_A_1_20_pool_1_2_1_1_int8.tflite
    16032, // conv_aaae_x1_8_200_p1_30_2_A_1_20_pool_1_2_1_1_float32.tflite
    7960, // conv_aaae_x1_8_200_p1_30_2_A_1_20_pool_1_2_1_1_int8.tflite
    6084, // conv_aaaf_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_float32.tflite
    3520, // conv_aaaf_x1_224_224_p1_10_1_A_10_10_pool_2_2_1_1_int8.tflite
    560, // dense_aaag_x1_1_50_p50_2_1_A_float32.tflite
    560, // dense_aaag_x1_1_50_p50_2_1_A_int8.tflite
    560, // dense_aaah_x1_1_1280_p1280_2_1_A_float32.tflite
    560, // dense_aaah_x1_1_1280_p1280_2_1_A_int8.tflite
    9196, // dense_aaai_x1_1_50_p50_8_4_A_float32.tflite
    4512, // dense_aaai_x1_1_50_p50_8_4_A_int8.tflite
    226860, // dense_aaaj_x1_1_448_p448_8_4_A_float32.tflite
    60208, // dense_aaaj_x1_1_448_p448_8_4_A_int8.tflite
    9068, // lstm_aaak_x1_1_10_p10_10_1_float32.tflite
    10208, // lstm_aaak_x1_1_10_p10_10_1_int8.tflite
    121068, // lstm_aaal_x1_1_60_p60_60_1_float32.tflite
    43008, // lstm_aaal_x1_1_60_p60_60_1_int8.tflite
    1285868, // lstm_aaam_x1_1_200_p200_200_1_float32.tflite
    347648, // lstm_aaam_x1_1_200_p200_200_1_int8.tflite
    20588, // lstm_aaan_x1_1_10_p10_10_2_float32.tflite
    23856, // lstm_aaan_x1_1_10_p10_10_2_int8.tflite
    244588, // lstm_aaao_x1_1_60_p60_60_2_float32.tflite
    89456, // lstm_aaao_x1_1_60_p60_60_2_int8.tflite
    2574188, // lstm_aaap_x1_1_200_p200_200_2_float32.tflite
    698736, // lstm_aaap_x1_1_200_p200_200_2_int8.tflite
    66732, // lstm_aaaq_x5_1_10_p10_10_2_float32.tflite
    93440, // lstm_aaaq_x5_1_10_p10_10_2_int8.tflite
    290732, // lstm_aaar_x5_1_60_p60_60_2_float32.tflite
    159040, // lstm_aaar_x5_1_60_p60_60_2_int8.tflite
};
