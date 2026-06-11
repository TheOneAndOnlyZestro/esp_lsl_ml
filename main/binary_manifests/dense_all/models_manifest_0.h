#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-Dense-All/batch_0
// Blob: models_0.bin (196560 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 60

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_25_p25_2_2_None_float32.tflite
    1, // dense_aaab_x1_1_25_p25_2_3_None_float32.tflite
    1, // dense_aaac_x1_1_25_p25_2_4_None_float32.tflite
    1, // dense_aaad_x1_1_25_p25_2_5_None_float32.tflite
    1, // dense_aaae_x1_1_25_p25_2_6_None_float32.tflite
    1, // dense_aaaf_x1_1_25_p25_2_7_None_float32.tflite
    1, // dense_aaag_x1_1_25_p25_2_8_None_float32.tflite
    1, // dense_aaah_x1_1_25_p25_2_9_None_float32.tflite
    1, // dense_aaai_x1_1_25_p25_2_10_None_float32.tflite
    1, // dense_aaaj_x1_1_25_p25_2_11_None_float32.tflite
    1, // dense_aaak_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaal_x1_1_100_p100_2_2_None_float32.tflite
    1, // dense_aaam_x1_1_150_p150_2_2_None_float32.tflite
    1, // dense_aaan_x1_1_200_p200_2_2_None_float32.tflite
    1, // dense_aaao_x1_1_250_p250_2_2_None_float32.tflite
    1, // dense_aaap_x1_1_300_p300_2_2_None_float32.tflite
    1, // dense_aaaq_x1_1_350_p350_2_2_None_float32.tflite
    1, // dense_aaar_x1_1_400_p400_2_2_None_float32.tflite
    1, // dense_aaas_x1_1_450_p450_2_2_None_float32.tflite
    1, // dense_aaat_x1_1_500_p500_2_2_None_float32.tflite
    1, // dense_aaau_x1_1_25_p25_1_2_None_float32.tflite
    1, // dense_aaav_x1_1_25_p25_5_2_None_float32.tflite
    1, // dense_aaaw_x1_1_25_p25_10_2_None_float32.tflite
    1, // dense_aaax_x1_1_25_p25_20_2_None_float32.tflite
    1, // dense_aaay_x1_1_25_p25_25_2_None_float32.tflite
    1, // dense_aaaz_x1_1_25_p25_30_2_None_float32.tflite
    1, // dense_aaba_x1_1_25_p25_35_2_None_float32.tflite
    1, // dense_aabb_x1_1_25_p25_40_2_None_float32.tflite
    1, // dense_aabc_x1_1_25_p25_45_2_None_float32.tflite
    1, // dense_aabd_x1_1_25_p25_50_2_None_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    1, // dense_aaaa_x1_1_25_p25_2_2_None_float32.tflite
    1, // dense_aaab_x1_1_25_p25_2_3_None_float32.tflite
    1, // dense_aaac_x1_1_25_p25_2_4_None_float32.tflite
    1, // dense_aaad_x1_1_25_p25_2_5_None_float32.tflite
    1, // dense_aaae_x1_1_25_p25_2_6_None_float32.tflite
    1, // dense_aaaf_x1_1_25_p25_2_7_None_float32.tflite
    1, // dense_aaag_x1_1_25_p25_2_8_None_float32.tflite
    1, // dense_aaah_x1_1_25_p25_2_9_None_float32.tflite
    1, // dense_aaai_x1_1_25_p25_2_10_None_float32.tflite
    1, // dense_aaaj_x1_1_25_p25_2_11_None_float32.tflite
    1, // dense_aaak_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaal_x1_1_100_p100_2_2_None_float32.tflite
    1, // dense_aaam_x1_1_150_p150_2_2_None_float32.tflite
    1, // dense_aaan_x1_1_200_p200_2_2_None_float32.tflite
    1, // dense_aaao_x1_1_250_p250_2_2_None_float32.tflite
    1, // dense_aaap_x1_1_300_p300_2_2_None_float32.tflite
    1, // dense_aaaq_x1_1_350_p350_2_2_None_float32.tflite
    1, // dense_aaar_x1_1_400_p400_2_2_None_float32.tflite
    1, // dense_aaas_x1_1_450_p450_2_2_None_float32.tflite
    1, // dense_aaat_x1_1_500_p500_2_2_None_float32.tflite
    1, // dense_aaau_x1_1_25_p25_1_2_None_float32.tflite
    1, // dense_aaav_x1_1_25_p25_5_2_None_float32.tflite
    1, // dense_aaaw_x1_1_25_p25_10_2_None_float32.tflite
    1, // dense_aaax_x1_1_25_p25_20_2_None_float32.tflite
    1, // dense_aaay_x1_1_25_p25_25_2_None_float32.tflite
    1, // dense_aaaz_x1_1_25_p25_30_2_None_float32.tflite
    1, // dense_aaba_x1_1_25_p25_35_2_None_float32.tflite
    1, // dense_aabb_x1_1_25_p25_40_2_None_float32.tflite
    1, // dense_aabc_x1_1_25_p25_45_2_None_float32.tflite
    1, // dense_aabd_x1_1_25_p25_50_2_None_float32.tflite
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
    28016, // dense_aaaf_x1_1_25_p25_2_7_None_float32.tflite (5760 bytes)
    33776, // dense_aaaf_x1_1_25_p25_2_7_None_int8.tflite (4392 bytes)
    38176, // dense_aaag_x1_1_25_p25_2_8_None_float32.tflite (6508 bytes)
    44688, // dense_aaag_x1_1_25_p25_2_8_None_int8.tflite (5024 bytes)
    49712, // dense_aaah_x1_1_25_p25_2_9_None_float32.tflite (7400 bytes)
    57120, // dense_aaah_x1_1_25_p25_2_9_None_int8.tflite (5696 bytes)
    62816, // dense_aaai_x1_1_25_p25_2_10_None_float32.tflite (8292 bytes)
    71120, // dense_aaai_x1_1_25_p25_2_10_None_int8.tflite (6384 bytes)
    77504, // dense_aaaj_x1_1_25_p25_2_11_None_float32.tflite (8992 bytes)
    86496, // dense_aaaj_x1_1_25_p25_2_11_None_int8.tflite (6976 bytes)
    93472, // dense_aaak_x1_1_50_p50_2_2_None_float32.tflite (1320 bytes)
    94800, // dense_aaak_x1_1_50_p50_2_2_None_int8.tflite (1176 bytes)
    95984, // dense_aaal_x1_1_100_p100_2_2_None_float32.tflite (1720 bytes)
    97712, // dense_aaal_x1_1_100_p100_2_2_None_int8.tflite (1272 bytes)
    98992, // dense_aaam_x1_1_150_p150_2_2_None_float32.tflite (2120 bytes)
    101120, // dense_aaam_x1_1_150_p150_2_2_None_int8.tflite (1376 bytes)
    102496, // dense_aaan_x1_1_200_p200_2_2_None_float32.tflite (2520 bytes)
    105024, // dense_aaan_x1_1_200_p200_2_2_None_int8.tflite (1472 bytes)
    106496, // dense_aaao_x1_1_250_p250_2_2_None_float32.tflite (2920 bytes)
    109424, // dense_aaao_x1_1_250_p250_2_2_None_int8.tflite (1576 bytes)
    111008, // dense_aaap_x1_1_300_p300_2_2_None_float32.tflite (3320 bytes)
    114336, // dense_aaap_x1_1_300_p300_2_2_None_int8.tflite (1672 bytes)
    116016, // dense_aaaq_x1_1_350_p350_2_2_None_float32.tflite (3720 bytes)
    119744, // dense_aaaq_x1_1_350_p350_2_2_None_int8.tflite (1776 bytes)
    121520, // dense_aaar_x1_1_400_p400_2_2_None_float32.tflite (4120 bytes)
    125648, // dense_aaar_x1_1_400_p400_2_2_None_int8.tflite (1872 bytes)
    127520, // dense_aaas_x1_1_450_p450_2_2_None_float32.tflite (4520 bytes)
    132048, // dense_aaas_x1_1_450_p450_2_2_None_int8.tflite (1976 bytes)
    134032, // dense_aaat_x1_1_500_p500_2_2_None_float32.tflite (4920 bytes)
    138960, // dense_aaat_x1_1_500_p500_2_2_None_int8.tflite (2072 bytes)
    141040, // dense_aaau_x1_1_25_p25_1_2_None_float32.tflite (1020 bytes)
    142064, // dense_aaau_x1_1_25_p25_1_2_None_int8.tflite (1088 bytes)
    143152, // dense_aaav_x1_1_25_p25_5_2_None_float32.tflite (1420 bytes)
    144576, // dense_aaav_x1_1_25_p25_5_2_None_int8.tflite (1232 bytes)
    145808, // dense_aaaw_x1_1_25_p25_10_2_None_float32.tflite (1920 bytes)
    147728, // dense_aaaw_x1_1_25_p25_10_2_None_int8.tflite (1416 bytes)
    149152, // dense_aaax_x1_1_25_p25_20_2_None_float32.tflite (2920 bytes)
    152080, // dense_aaax_x1_1_25_p25_20_2_None_int8.tflite (1792 bytes)
    153872, // dense_aaay_x1_1_25_p25_25_2_None_float32.tflite (3420 bytes)
    157296, // dense_aaay_x1_1_25_p25_25_2_None_int8.tflite (1976 bytes)
    159280, // dense_aaaz_x1_1_25_p25_30_2_None_float32.tflite (3920 bytes)
    163200, // dense_aaaz_x1_1_25_p25_30_2_None_int8.tflite (2160 bytes)
    165360, // dense_aaba_x1_1_25_p25_35_2_None_float32.tflite (4420 bytes)
    169792, // dense_aaba_x1_1_25_p25_35_2_None_int8.tflite (2344 bytes)
    172144, // dense_aabb_x1_1_25_p25_40_2_None_float32.tflite (4920 bytes)
    177072, // dense_aabb_x1_1_25_p25_40_2_None_int8.tflite (2528 bytes)
    179600, // dense_aabc_x1_1_25_p25_45_2_None_float32.tflite (5420 bytes)
    185024, // dense_aabc_x1_1_25_p25_45_2_None_int8.tflite (2712 bytes)
    187744, // dense_aabd_x1_1_25_p25_50_2_None_float32.tflite (5920 bytes)
    193664, // dense_aabd_x1_1_25_p25_50_2_None_int8.tflite (2896 bytes)
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
    5760, // dense_aaaf_x1_1_25_p25_2_7_None_float32.tflite
    4392, // dense_aaaf_x1_1_25_p25_2_7_None_int8.tflite
    6508, // dense_aaag_x1_1_25_p25_2_8_None_float32.tflite
    5024, // dense_aaag_x1_1_25_p25_2_8_None_int8.tflite
    7400, // dense_aaah_x1_1_25_p25_2_9_None_float32.tflite
    5696, // dense_aaah_x1_1_25_p25_2_9_None_int8.tflite
    8292, // dense_aaai_x1_1_25_p25_2_10_None_float32.tflite
    6384, // dense_aaai_x1_1_25_p25_2_10_None_int8.tflite
    8992, // dense_aaaj_x1_1_25_p25_2_11_None_float32.tflite
    6976, // dense_aaaj_x1_1_25_p25_2_11_None_int8.tflite
    1320, // dense_aaak_x1_1_50_p50_2_2_None_float32.tflite
    1176, // dense_aaak_x1_1_50_p50_2_2_None_int8.tflite
    1720, // dense_aaal_x1_1_100_p100_2_2_None_float32.tflite
    1272, // dense_aaal_x1_1_100_p100_2_2_None_int8.tflite
    2120, // dense_aaam_x1_1_150_p150_2_2_None_float32.tflite
    1376, // dense_aaam_x1_1_150_p150_2_2_None_int8.tflite
    2520, // dense_aaan_x1_1_200_p200_2_2_None_float32.tflite
    1472, // dense_aaan_x1_1_200_p200_2_2_None_int8.tflite
    2920, // dense_aaao_x1_1_250_p250_2_2_None_float32.tflite
    1576, // dense_aaao_x1_1_250_p250_2_2_None_int8.tflite
    3320, // dense_aaap_x1_1_300_p300_2_2_None_float32.tflite
    1672, // dense_aaap_x1_1_300_p300_2_2_None_int8.tflite
    3720, // dense_aaaq_x1_1_350_p350_2_2_None_float32.tflite
    1776, // dense_aaaq_x1_1_350_p350_2_2_None_int8.tflite
    4120, // dense_aaar_x1_1_400_p400_2_2_None_float32.tflite
    1872, // dense_aaar_x1_1_400_p400_2_2_None_int8.tflite
    4520, // dense_aaas_x1_1_450_p450_2_2_None_float32.tflite
    1976, // dense_aaas_x1_1_450_p450_2_2_None_int8.tflite
    4920, // dense_aaat_x1_1_500_p500_2_2_None_float32.tflite
    2072, // dense_aaat_x1_1_500_p500_2_2_None_int8.tflite
    1020, // dense_aaau_x1_1_25_p25_1_2_None_float32.tflite
    1088, // dense_aaau_x1_1_25_p25_1_2_None_int8.tflite
    1420, // dense_aaav_x1_1_25_p25_5_2_None_float32.tflite
    1232, // dense_aaav_x1_1_25_p25_5_2_None_int8.tflite
    1920, // dense_aaaw_x1_1_25_p25_10_2_None_float32.tflite
    1416, // dense_aaaw_x1_1_25_p25_10_2_None_int8.tflite
    2920, // dense_aaax_x1_1_25_p25_20_2_None_float32.tflite
    1792, // dense_aaax_x1_1_25_p25_20_2_None_int8.tflite
    3420, // dense_aaay_x1_1_25_p25_25_2_None_float32.tflite
    1976, // dense_aaay_x1_1_25_p25_25_2_None_int8.tflite
    3920, // dense_aaaz_x1_1_25_p25_30_2_None_float32.tflite
    2160, // dense_aaaz_x1_1_25_p25_30_2_None_int8.tflite
    4420, // dense_aaba_x1_1_25_p25_35_2_None_float32.tflite
    2344, // dense_aaba_x1_1_25_p25_35_2_None_int8.tflite
    4920, // dense_aabb_x1_1_25_p25_40_2_None_float32.tflite
    2528, // dense_aabb_x1_1_25_p25_40_2_None_int8.tflite
    5420, // dense_aabc_x1_1_25_p25_45_2_None_float32.tflite
    2712, // dense_aabc_x1_1_25_p25_45_2_None_int8.tflite
    5920, // dense_aabd_x1_1_25_p25_50_2_None_float32.tflite
    2896, // dense_aabd_x1_1_25_p25_50_2_None_int8.tflite
};
