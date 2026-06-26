#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Dense-Layers-Seed1/batch_0
// Blob: models_0.bin (616312 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 40

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaab_x1_1_50_p50_2_3_None_float32.tflite
    1, // dense_aaac_x1_1_50_p50_2_4_None_float32.tflite
    1, // dense_aaad_x1_1_50_p50_2_5_None_float32.tflite
    1, // dense_aaae_x1_1_50_p50_2_6_None_float32.tflite
    1, // dense_aaaf_x1_1_50_p50_2_7_None_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_8_None_float32.tflite
    1, // dense_aaah_x1_1_50_p50_2_9_None_float32.tflite
    1, // dense_aaai_x1_1_50_p50_2_10_None_float32.tflite
    1, // dense_aaaj_x1_1_50_p50_2_11_None_float32.tflite
    1, // dense_aaak_x1_1_50_p50_2_12_None_float32.tflite
    1, // dense_aaal_x1_1_50_p50_2_13_None_float32.tflite
    1, // dense_aaam_x1_1_50_p50_2_14_None_float32.tflite
    1, // dense_aaan_x1_1_50_p50_2_15_None_float32.tflite
    1, // dense_aaao_x1_1_50_p50_2_16_None_float32.tflite
    1, // dense_aaap_x1_1_50_p50_2_17_None_float32.tflite
    1, // dense_aaaq_x1_1_50_p50_2_18_None_float32.tflite
    1, // dense_aaar_x1_1_50_p50_2_19_None_float32.tflite
    1, // dense_aaas_x1_1_50_p50_2_20_None_float32.tflite
    1, // dense_aaat_x1_1_50_p50_2_21_None_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaab_x1_1_50_p50_2_3_None_float32.tflite
    1, // dense_aaac_x1_1_50_p50_2_4_None_float32.tflite
    1, // dense_aaad_x1_1_50_p50_2_5_None_float32.tflite
    1, // dense_aaae_x1_1_50_p50_2_6_None_float32.tflite
    1, // dense_aaaf_x1_1_50_p50_2_7_None_float32.tflite
    1, // dense_aaag_x1_1_50_p50_2_8_None_float32.tflite
    1, // dense_aaah_x1_1_50_p50_2_9_None_float32.tflite
    1, // dense_aaai_x1_1_50_p50_2_10_None_float32.tflite
    1, // dense_aaaj_x1_1_50_p50_2_11_None_float32.tflite
    1, // dense_aaak_x1_1_50_p50_2_12_None_float32.tflite
    1, // dense_aaal_x1_1_50_p50_2_13_None_float32.tflite
    1, // dense_aaam_x1_1_50_p50_2_14_None_float32.tflite
    1, // dense_aaan_x1_1_50_p50_2_15_None_float32.tflite
    1, // dense_aaao_x1_1_50_p50_2_16_None_float32.tflite
    1, // dense_aaap_x1_1_50_p50_2_17_None_float32.tflite
    1, // dense_aaaq_x1_1_50_p50_2_18_None_float32.tflite
    1, // dense_aaar_x1_1_50_p50_2_19_None_float32.tflite
    1, // dense_aaas_x1_1_50_p50_2_20_None_float32.tflite
    1, // dense_aaat_x1_1_50_p50_2_21_None_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // dense_aaaa_x1_1_50_p50_2_2_None_float32.tflite (1320 bytes)
    1328, // dense_aaaa_x1_1_50_p50_2_2_None_int8.tflite (1176 bytes)
    2512, // dense_aaab_x1_1_50_p50_2_3_None_float32.tflite (3708 bytes)
    6224, // dense_aaab_x1_1_50_p50_2_3_None_int8.tflite (2144 bytes)
    8368, // dense_aaac_x1_1_50_p50_2_4_None_float32.tflite (5808 bytes)
    14176, // dense_aaac_x1_1_50_p50_2_4_None_int8.tflite (3168 bytes)
    17344, // dense_aaad_x1_1_50_p50_2_5_None_float32.tflite (7748 bytes)
    25104, // dense_aaad_x1_1_50_p50_2_5_None_int8.tflite (4144 bytes)
    29248, // dense_aaae_x1_1_50_p50_2_6_None_float32.tflite (9792 bytes)
    39040, // dense_aaae_x1_1_50_p50_2_6_None_int8.tflite (5176 bytes)
    44224, // dense_aaaf_x1_1_50_p50_2_7_None_float32.tflite (11712 bytes)
    55936, // dense_aaaf_x1_1_50_p50_2_7_None_int8.tflite (6160 bytes)
    62096, // dense_aaag_x1_1_50_p50_2_8_None_float32.tflite (13908 bytes)
    76016, // dense_aaag_x1_1_50_p50_2_8_None_int8.tflite (7240 bytes)
    83264, // dense_aaah_x1_1_50_p50_2_9_None_float32.tflite (15460 bytes)
    98736, // dense_aaah_x1_1_50_p50_2_9_None_int8.tflite (8128 bytes)
    106864, // dense_aaai_x1_1_50_p50_2_10_None_float32.tflite (17448 bytes)
    124320, // dense_aaai_x1_1_50_p50_2_10_None_int8.tflite (9136 bytes)
    133456, // dense_aaaj_x1_1_50_p50_2_11_None_float32.tflite (19336 bytes)
    152800, // dense_aaaj_x1_1_50_p50_2_11_None_int8.tflite (10120 bytes)
    162928, // dense_aaak_x1_1_50_p50_2_12_None_float32.tflite (21472 bytes)
    184400, // dense_aaak_x1_1_50_p50_2_12_None_int8.tflite (11208 bytes)
    195616, // dense_aaal_x1_1_50_p50_2_13_None_float32.tflite (23044 bytes)
    218672, // dense_aaal_x1_1_50_p50_2_13_None_int8.tflite (12088 bytes)
    230768, // dense_aaam_x1_1_50_p50_2_14_None_float32.tflite (25032 bytes)
    255808, // dense_aaam_x1_1_50_p50_2_14_None_int8.tflite (13096 bytes)
    268912, // dense_aaan_x1_1_50_p50_2_15_None_float32.tflite (27288 bytes)
    296208, // dense_aaan_x1_1_50_p50_2_15_None_int8.tflite (14192 bytes)
    310400, // dense_aaao_x1_1_50_p50_2_16_None_float32.tflite (28836 bytes)
    339248, // dense_aaao_x1_1_50_p50_2_16_None_int8.tflite (15096 bytes)
    354352, // dense_aaap_x1_1_50_p50_2_17_None_float32.tflite (30496 bytes)
    384848, // dense_aaap_x1_1_50_p50_2_17_None_int8.tflite (16000 bytes)
    400848, // dense_aaaq_x1_1_50_p50_2_18_None_float32.tflite (32244 bytes)
    433104, // dense_aaaq_x1_1_50_p50_2_18_None_int8.tflite (16968 bytes)
    450080, // dense_aaar_x1_1_50_p50_2_19_None_float32.tflite (34356 bytes)
    484448, // dense_aaar_x1_1_50_p50_2_19_None_int8.tflite (18024 bytes)
    502480, // dense_aaas_x1_1_50_p50_2_20_None_float32.tflite (36348 bytes)
    538832, // dense_aaas_x1_1_50_p50_2_20_None_int8.tflite (19072 bytes)
    557904, // dense_aaat_x1_1_50_p50_2_21_None_float32.tflite (38332 bytes)
    596240, // dense_aaat_x1_1_50_p50_2_21_None_int8.tflite (20072 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    1320, // dense_aaaa_x1_1_50_p50_2_2_None_float32.tflite
    1176, // dense_aaaa_x1_1_50_p50_2_2_None_int8.tflite
    3708, // dense_aaab_x1_1_50_p50_2_3_None_float32.tflite
    2144, // dense_aaab_x1_1_50_p50_2_3_None_int8.tflite
    5808, // dense_aaac_x1_1_50_p50_2_4_None_float32.tflite
    3168, // dense_aaac_x1_1_50_p50_2_4_None_int8.tflite
    7748, // dense_aaad_x1_1_50_p50_2_5_None_float32.tflite
    4144, // dense_aaad_x1_1_50_p50_2_5_None_int8.tflite
    9792, // dense_aaae_x1_1_50_p50_2_6_None_float32.tflite
    5176, // dense_aaae_x1_1_50_p50_2_6_None_int8.tflite
    11712, // dense_aaaf_x1_1_50_p50_2_7_None_float32.tflite
    6160, // dense_aaaf_x1_1_50_p50_2_7_None_int8.tflite
    13908, // dense_aaag_x1_1_50_p50_2_8_None_float32.tflite
    7240, // dense_aaag_x1_1_50_p50_2_8_None_int8.tflite
    15460, // dense_aaah_x1_1_50_p50_2_9_None_float32.tflite
    8128, // dense_aaah_x1_1_50_p50_2_9_None_int8.tflite
    17448, // dense_aaai_x1_1_50_p50_2_10_None_float32.tflite
    9136, // dense_aaai_x1_1_50_p50_2_10_None_int8.tflite
    19336, // dense_aaaj_x1_1_50_p50_2_11_None_float32.tflite
    10120, // dense_aaaj_x1_1_50_p50_2_11_None_int8.tflite
    21472, // dense_aaak_x1_1_50_p50_2_12_None_float32.tflite
    11208, // dense_aaak_x1_1_50_p50_2_12_None_int8.tflite
    23044, // dense_aaal_x1_1_50_p50_2_13_None_float32.tflite
    12088, // dense_aaal_x1_1_50_p50_2_13_None_int8.tflite
    25032, // dense_aaam_x1_1_50_p50_2_14_None_float32.tflite
    13096, // dense_aaam_x1_1_50_p50_2_14_None_int8.tflite
    27288, // dense_aaan_x1_1_50_p50_2_15_None_float32.tflite
    14192, // dense_aaan_x1_1_50_p50_2_15_None_int8.tflite
    28836, // dense_aaao_x1_1_50_p50_2_16_None_float32.tflite
    15096, // dense_aaao_x1_1_50_p50_2_16_None_int8.tflite
    30496, // dense_aaap_x1_1_50_p50_2_17_None_float32.tflite
    16000, // dense_aaap_x1_1_50_p50_2_17_None_int8.tflite
    32244, // dense_aaaq_x1_1_50_p50_2_18_None_float32.tflite
    16968, // dense_aaaq_x1_1_50_p50_2_18_None_int8.tflite
    34356, // dense_aaar_x1_1_50_p50_2_19_None_float32.tflite
    18024, // dense_aaar_x1_1_50_p50_2_19_None_int8.tflite
    36348, // dense_aaas_x1_1_50_p50_2_20_None_float32.tflite
    19072, // dense_aaas_x1_1_50_p50_2_20_None_int8.tflite
    38332, // dense_aaat_x1_1_50_p50_2_21_None_float32.tflite
    20072, // dense_aaat_x1_1_50_p50_2_21_None_int8.tflite
};
