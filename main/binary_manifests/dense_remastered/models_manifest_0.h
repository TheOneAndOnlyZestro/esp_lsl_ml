#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Dense-Models-Remastered/batch_0
// Blob: models_0.bin (302264 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 80

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
    1, // dense_aaak_x1_1_25_p25_2_12_None_float32.tflite
    1, // dense_aaal_x1_1_25_p25_2_13_None_float32.tflite
    1, // dense_aaam_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaan_x1_1_100_p100_2_2_None_float32.tflite
    1, // dense_aaao_x1_1_150_p150_2_2_None_float32.tflite
    1, // dense_aaap_x1_1_200_p200_2_2_None_float32.tflite
    1, // dense_aaaq_x1_1_250_p250_2_2_None_float32.tflite
    1, // dense_aaar_x1_1_300_p300_2_2_None_float32.tflite
    1, // dense_aaas_x1_1_350_p350_2_2_None_float32.tflite
    1, // dense_aaat_x1_1_400_p400_2_2_None_float32.tflite
    1, // dense_aaau_x1_1_450_p450_2_2_None_float32.tflite
    1, // dense_aaav_x1_1_500_p500_2_2_None_float32.tflite
    1, // dense_aaaw_x1_1_550_p550_2_2_None_float32.tflite
    1, // dense_aaax_x1_1_600_p600_2_2_None_float32.tflite
    1, // dense_aaay_x1_1_650_p650_2_2_None_float32.tflite
    1, // dense_aaaz_x1_1_700_p700_2_2_None_float32.tflite
    1, // dense_aaba_x1_1_25_p25_1_2_None_float32.tflite
    1, // dense_aabb_x1_1_25_p25_5_2_None_float32.tflite
    1, // dense_aabc_x1_1_25_p25_10_2_None_float32.tflite
    1, // dense_aabd_x1_1_25_p25_15_2_None_float32.tflite
    1, // dense_aabe_x1_1_25_p25_20_2_None_float32.tflite
    1, // dense_aabf_x1_1_25_p25_25_2_None_float32.tflite
    1, // dense_aabg_x1_1_25_p25_30_2_None_float32.tflite
    1, // dense_aabh_x1_1_25_p25_35_2_None_float32.tflite
    1, // dense_aabi_x1_1_25_p25_40_2_None_float32.tflite
    1, // dense_aabj_x1_1_25_p25_45_2_None_float32.tflite
    1, // dense_aabk_x1_1_25_p25_50_2_None_float32.tflite
    1, // dense_aabl_x1_1_25_p25_55_2_None_float32.tflite
    1, // dense_aabm_x1_1_25_p25_65_2_None_float32.tflite
    1, // dense_aabn_x1_1_25_p25_70_2_None_float32.tflite
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
    1, // dense_aaak_x1_1_25_p25_2_12_None_float32.tflite
    1, // dense_aaal_x1_1_25_p25_2_13_None_float32.tflite
    1, // dense_aaam_x1_1_50_p50_2_2_None_float32.tflite
    1, // dense_aaan_x1_1_100_p100_2_2_None_float32.tflite
    1, // dense_aaao_x1_1_150_p150_2_2_None_float32.tflite
    1, // dense_aaap_x1_1_200_p200_2_2_None_float32.tflite
    1, // dense_aaaq_x1_1_250_p250_2_2_None_float32.tflite
    1, // dense_aaar_x1_1_300_p300_2_2_None_float32.tflite
    1, // dense_aaas_x1_1_350_p350_2_2_None_float32.tflite
    1, // dense_aaat_x1_1_400_p400_2_2_None_float32.tflite
    1, // dense_aaau_x1_1_450_p450_2_2_None_float32.tflite
    1, // dense_aaav_x1_1_500_p500_2_2_None_float32.tflite
    1, // dense_aaaw_x1_1_550_p550_2_2_None_float32.tflite
    1, // dense_aaax_x1_1_600_p600_2_2_None_float32.tflite
    1, // dense_aaay_x1_1_650_p650_2_2_None_float32.tflite
    1, // dense_aaaz_x1_1_700_p700_2_2_None_float32.tflite
    1, // dense_aaba_x1_1_25_p25_1_2_None_float32.tflite
    1, // dense_aabb_x1_1_25_p25_5_2_None_float32.tflite
    1, // dense_aabc_x1_1_25_p25_10_2_None_float32.tflite
    1, // dense_aabd_x1_1_25_p25_15_2_None_float32.tflite
    1, // dense_aabe_x1_1_25_p25_20_2_None_float32.tflite
    1, // dense_aabf_x1_1_25_p25_25_2_None_float32.tflite
    1, // dense_aabg_x1_1_25_p25_30_2_None_float32.tflite
    1, // dense_aabh_x1_1_25_p25_35_2_None_float32.tflite
    1, // dense_aabi_x1_1_25_p25_40_2_None_float32.tflite
    1, // dense_aabj_x1_1_25_p25_45_2_None_float32.tflite
    1, // dense_aabk_x1_1_25_p25_50_2_None_float32.tflite
    1, // dense_aabl_x1_1_25_p25_55_2_None_float32.tflite
    1, // dense_aabm_x1_1_25_p25_65_2_None_float32.tflite
    1, // dense_aabn_x1_1_25_p25_70_2_None_float32.tflite
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
    93472, // dense_aaak_x1_1_25_p25_2_12_None_float32.tflite (9996 bytes)
    103472, // dense_aaak_x1_1_25_p25_2_12_None_int8.tflite (7704 bytes)
    111184, // dense_aaal_x1_1_25_p25_2_13_None_float32.tflite (10656 bytes)
    121840, // dense_aaal_x1_1_25_p25_2_13_None_int8.tflite (8296 bytes)
    130144, // dense_aaam_x1_1_50_p50_2_2_None_float32.tflite (1320 bytes)
    131472, // dense_aaam_x1_1_50_p50_2_2_None_int8.tflite (1176 bytes)
    132656, // dense_aaan_x1_1_100_p100_2_2_None_float32.tflite (1720 bytes)
    134384, // dense_aaan_x1_1_100_p100_2_2_None_int8.tflite (1272 bytes)
    135664, // dense_aaao_x1_1_150_p150_2_2_None_float32.tflite (2120 bytes)
    137792, // dense_aaao_x1_1_150_p150_2_2_None_int8.tflite (1376 bytes)
    139168, // dense_aaap_x1_1_200_p200_2_2_None_float32.tflite (2520 bytes)
    141696, // dense_aaap_x1_1_200_p200_2_2_None_int8.tflite (1472 bytes)
    143168, // dense_aaaq_x1_1_250_p250_2_2_None_float32.tflite (2920 bytes)
    146096, // dense_aaaq_x1_1_250_p250_2_2_None_int8.tflite (1576 bytes)
    147680, // dense_aaar_x1_1_300_p300_2_2_None_float32.tflite (3320 bytes)
    151008, // dense_aaar_x1_1_300_p300_2_2_None_int8.tflite (1672 bytes)
    152688, // dense_aaas_x1_1_350_p350_2_2_None_float32.tflite (3720 bytes)
    156416, // dense_aaas_x1_1_350_p350_2_2_None_int8.tflite (1776 bytes)
    158192, // dense_aaat_x1_1_400_p400_2_2_None_float32.tflite (4120 bytes)
    162320, // dense_aaat_x1_1_400_p400_2_2_None_int8.tflite (1872 bytes)
    164192, // dense_aaau_x1_1_450_p450_2_2_None_float32.tflite (4520 bytes)
    168720, // dense_aaau_x1_1_450_p450_2_2_None_int8.tflite (1976 bytes)
    170704, // dense_aaav_x1_1_500_p500_2_2_None_float32.tflite (4920 bytes)
    175632, // dense_aaav_x1_1_500_p500_2_2_None_int8.tflite (2072 bytes)
    177712, // dense_aaaw_x1_1_550_p550_2_2_None_float32.tflite (5320 bytes)
    183040, // dense_aaaw_x1_1_550_p550_2_2_None_int8.tflite (2176 bytes)
    185216, // dense_aaax_x1_1_600_p600_2_2_None_float32.tflite (5720 bytes)
    190944, // dense_aaax_x1_1_600_p600_2_2_None_int8.tflite (2272 bytes)
    193216, // dense_aaay_x1_1_650_p650_2_2_None_float32.tflite (6120 bytes)
    199344, // dense_aaay_x1_1_650_p650_2_2_None_int8.tflite (2376 bytes)
    201728, // dense_aaaz_x1_1_700_p700_2_2_None_float32.tflite (6520 bytes)
    208256, // dense_aaaz_x1_1_700_p700_2_2_None_int8.tflite (2472 bytes)
    210736, // dense_aaba_x1_1_25_p25_1_2_None_float32.tflite (1020 bytes)
    211760, // dense_aaba_x1_1_25_p25_1_2_None_int8.tflite (1088 bytes)
    212848, // dense_aabb_x1_1_25_p25_5_2_None_float32.tflite (1420 bytes)
    214272, // dense_aabb_x1_1_25_p25_5_2_None_int8.tflite (1232 bytes)
    215504, // dense_aabc_x1_1_25_p25_10_2_None_float32.tflite (1920 bytes)
    217424, // dense_aabc_x1_1_25_p25_10_2_None_int8.tflite (1416 bytes)
    218848, // dense_aabd_x1_1_25_p25_15_2_None_float32.tflite (2420 bytes)
    221280, // dense_aabd_x1_1_25_p25_15_2_None_int8.tflite (1608 bytes)
    222896, // dense_aabe_x1_1_25_p25_20_2_None_float32.tflite (2920 bytes)
    225824, // dense_aabe_x1_1_25_p25_20_2_None_int8.tflite (1792 bytes)
    227616, // dense_aabf_x1_1_25_p25_25_2_None_float32.tflite (3420 bytes)
    231040, // dense_aabf_x1_1_25_p25_25_2_None_int8.tflite (1976 bytes)
    233024, // dense_aabg_x1_1_25_p25_30_2_None_float32.tflite (3920 bytes)
    236944, // dense_aabg_x1_1_25_p25_30_2_None_int8.tflite (2160 bytes)
    239104, // dense_aabh_x1_1_25_p25_35_2_None_float32.tflite (4420 bytes)
    243536, // dense_aabh_x1_1_25_p25_35_2_None_int8.tflite (2344 bytes)
    245888, // dense_aabi_x1_1_25_p25_40_2_None_float32.tflite (4920 bytes)
    250816, // dense_aabi_x1_1_25_p25_40_2_None_int8.tflite (2528 bytes)
    253344, // dense_aabj_x1_1_25_p25_45_2_None_float32.tflite (5420 bytes)
    258768, // dense_aabj_x1_1_25_p25_45_2_None_int8.tflite (2712 bytes)
    261488, // dense_aabk_x1_1_25_p25_50_2_None_float32.tflite (5920 bytes)
    267408, // dense_aabk_x1_1_25_p25_50_2_None_int8.tflite (2896 bytes)
    270304, // dense_aabl_x1_1_25_p25_55_2_None_float32.tflite (6420 bytes)
    276736, // dense_aabl_x1_1_25_p25_55_2_None_int8.tflite (3088 bytes)
    279824, // dense_aabm_x1_1_25_p25_65_2_None_float32.tflite (7420 bytes)
    287248, // dense_aabm_x1_1_25_p25_65_2_None_int8.tflite (3456 bytes)
    290704, // dense_aabn_x1_1_25_p25_70_2_None_float32.tflite (7920 bytes)
    298624, // dense_aabn_x1_1_25_p25_70_2_None_int8.tflite (3640 bytes)
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
    9996, // dense_aaak_x1_1_25_p25_2_12_None_float32.tflite
    7704, // dense_aaak_x1_1_25_p25_2_12_None_int8.tflite
    10656, // dense_aaal_x1_1_25_p25_2_13_None_float32.tflite
    8296, // dense_aaal_x1_1_25_p25_2_13_None_int8.tflite
    1320, // dense_aaam_x1_1_50_p50_2_2_None_float32.tflite
    1176, // dense_aaam_x1_1_50_p50_2_2_None_int8.tflite
    1720, // dense_aaan_x1_1_100_p100_2_2_None_float32.tflite
    1272, // dense_aaan_x1_1_100_p100_2_2_None_int8.tflite
    2120, // dense_aaao_x1_1_150_p150_2_2_None_float32.tflite
    1376, // dense_aaao_x1_1_150_p150_2_2_None_int8.tflite
    2520, // dense_aaap_x1_1_200_p200_2_2_None_float32.tflite
    1472, // dense_aaap_x1_1_200_p200_2_2_None_int8.tflite
    2920, // dense_aaaq_x1_1_250_p250_2_2_None_float32.tflite
    1576, // dense_aaaq_x1_1_250_p250_2_2_None_int8.tflite
    3320, // dense_aaar_x1_1_300_p300_2_2_None_float32.tflite
    1672, // dense_aaar_x1_1_300_p300_2_2_None_int8.tflite
    3720, // dense_aaas_x1_1_350_p350_2_2_None_float32.tflite
    1776, // dense_aaas_x1_1_350_p350_2_2_None_int8.tflite
    4120, // dense_aaat_x1_1_400_p400_2_2_None_float32.tflite
    1872, // dense_aaat_x1_1_400_p400_2_2_None_int8.tflite
    4520, // dense_aaau_x1_1_450_p450_2_2_None_float32.tflite
    1976, // dense_aaau_x1_1_450_p450_2_2_None_int8.tflite
    4920, // dense_aaav_x1_1_500_p500_2_2_None_float32.tflite
    2072, // dense_aaav_x1_1_500_p500_2_2_None_int8.tflite
    5320, // dense_aaaw_x1_1_550_p550_2_2_None_float32.tflite
    2176, // dense_aaaw_x1_1_550_p550_2_2_None_int8.tflite
    5720, // dense_aaax_x1_1_600_p600_2_2_None_float32.tflite
    2272, // dense_aaax_x1_1_600_p600_2_2_None_int8.tflite
    6120, // dense_aaay_x1_1_650_p650_2_2_None_float32.tflite
    2376, // dense_aaay_x1_1_650_p650_2_2_None_int8.tflite
    6520, // dense_aaaz_x1_1_700_p700_2_2_None_float32.tflite
    2472, // dense_aaaz_x1_1_700_p700_2_2_None_int8.tflite
    1020, // dense_aaba_x1_1_25_p25_1_2_None_float32.tflite
    1088, // dense_aaba_x1_1_25_p25_1_2_None_int8.tflite
    1420, // dense_aabb_x1_1_25_p25_5_2_None_float32.tflite
    1232, // dense_aabb_x1_1_25_p25_5_2_None_int8.tflite
    1920, // dense_aabc_x1_1_25_p25_10_2_None_float32.tflite
    1416, // dense_aabc_x1_1_25_p25_10_2_None_int8.tflite
    2420, // dense_aabd_x1_1_25_p25_15_2_None_float32.tflite
    1608, // dense_aabd_x1_1_25_p25_15_2_None_int8.tflite
    2920, // dense_aabe_x1_1_25_p25_20_2_None_float32.tflite
    1792, // dense_aabe_x1_1_25_p25_20_2_None_int8.tflite
    3420, // dense_aabf_x1_1_25_p25_25_2_None_float32.tflite
    1976, // dense_aabf_x1_1_25_p25_25_2_None_int8.tflite
    3920, // dense_aabg_x1_1_25_p25_30_2_None_float32.tflite
    2160, // dense_aabg_x1_1_25_p25_30_2_None_int8.tflite
    4420, // dense_aabh_x1_1_25_p25_35_2_None_float32.tflite
    2344, // dense_aabh_x1_1_25_p25_35_2_None_int8.tflite
    4920, // dense_aabi_x1_1_25_p25_40_2_None_float32.tflite
    2528, // dense_aabi_x1_1_25_p25_40_2_None_int8.tflite
    5420, // dense_aabj_x1_1_25_p25_45_2_None_float32.tflite
    2712, // dense_aabj_x1_1_25_p25_45_2_None_int8.tflite
    5920, // dense_aabk_x1_1_25_p25_50_2_None_float32.tflite
    2896, // dense_aabk_x1_1_25_p25_50_2_None_int8.tflite
    6420, // dense_aabl_x1_1_25_p25_55_2_None_float32.tflite
    3088, // dense_aabl_x1_1_25_p25_55_2_None_int8.tflite
    7420, // dense_aabm_x1_1_25_p25_65_2_None_float32.tflite
    3456, // dense_aabm_x1_1_25_p25_65_2_None_int8.tflite
    7920, // dense_aabn_x1_1_25_p25_70_2_None_float32.tflite
    3640, // dense_aabn_x1_1_25_p25_70_2_None_int8.tflite
};
