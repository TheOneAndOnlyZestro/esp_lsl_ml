#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-LSTM-All/batch_0
// Blob: models_0.bin (4826496 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 32

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    3, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite
    3, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite
    3, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite
    3, // lstm_aaaf_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaag_x1_1_50_p50_50_1_float32.tflite
    3, // lstm_aaah_x1_1_100_p100_100_1_float32.tflite
    3, // lstm_aaai_x1_1_150_p150_150_1_float32.tflite
    3, // lstm_aaaj_x1_1_200_p200_200_1_float32.tflite
    3, // lstm_aaak_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaal_x1_1_25_p25_25_2_float32.tflite
    3, // lstm_aaam_x1_1_25_p25_25_3_float32.tflite
    3, // lstm_aaan_x1_1_25_p25_25_4_float32.tflite
    3, // lstm_aaao_x1_1_25_p25_25_5_float32.tflite
    3, // lstm_aaap_x1_1_25_p25_25_6_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    3, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite
    3, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite
    3, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite
    3, // lstm_aaaf_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaag_x1_1_50_p50_50_1_float32.tflite
    3, // lstm_aaah_x1_1_100_p100_100_1_float32.tflite
    3, // lstm_aaai_x1_1_150_p150_150_1_float32.tflite
    3, // lstm_aaaj_x1_1_200_p200_200_1_float32.tflite
    3, // lstm_aaak_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaal_x1_1_25_p25_25_2_float32.tflite
    3, // lstm_aaam_x1_1_25_p25_25_3_float32.tflite
    3, // lstm_aaan_x1_1_25_p25_25_4_float32.tflite
    3, // lstm_aaao_x1_1_25_p25_25_5_float32.tflite
    3, // lstm_aaap_x1_1_25_p25_25_6_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite (26396 bytes)
    26400, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite (16352 bytes)
    42752, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite (42432 bytes)
    85184, // lstm_aaab_x5_1_25_p25_25_1_int8.tflite (44560 bytes)
    129744, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite (62312 bytes)
    192064, // lstm_aaac_x10_1_25_p25_25_1_int8.tflite (79512 bytes)
    271584, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite (83408 bytes)
    354992, // lstm_aaad_x15_1_25_p25_25_1_int8.tflite (115912 bytes)
    470912, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite (103396 bytes)
    574320, // lstm_aaae_x20_1_25_p25_25_1_int8.tflite (151080 bytes)
    725408, // lstm_aaaf_x1_1_25_p25_25_1_float32.tflite (26396 bytes)
    751808, // lstm_aaaf_x1_1_25_p25_25_1_int8.tflite (16352 bytes)
    768160, // lstm_aaag_x1_1_50_p50_50_1_float32.tflite (86396 bytes)
    854560, // lstm_aaag_x1_1_50_p50_50_1_int8.tflite (33760 bytes)
    888320, // lstm_aaah_x1_1_100_p100_100_1_float32.tflite (326396 bytes)
    1214720, // lstm_aaah_x1_1_100_p100_100_1_int8.tflite (98560 bytes)
    1313280, // lstm_aaai_x1_1_150_p150_150_1_float32.tflite (726396 bytes)
    2039680, // lstm_aaai_x1_1_150_p150_150_1_int8.tflite (203360 bytes)
    2243040, // lstm_aaaj_x1_1_200_p200_200_1_float32.tflite (1286396 bytes)
    3529440, // lstm_aaaj_x1_1_200_p200_200_1_int8.tflite (348160 bytes)
    3877600, // lstm_aaak_x1_1_25_p25_25_1_float32.tflite (26396 bytes)
    3904000, // lstm_aaak_x1_1_25_p25_25_1_int8.tflite (16352 bytes)
    3920352, // lstm_aaal_x1_1_25_p25_25_2_float32.tflite (55636 bytes)
    3976000, // lstm_aaal_x1_1_25_p25_25_2_int8.tflite (36584 bytes)
    4012592, // lstm_aaam_x1_1_25_p25_25_3_float32.tflite (82536 bytes)
    4095136, // lstm_aaam_x1_1_25_p25_25_3_int8.tflite (54176 bytes)
    4149312, // lstm_aaan_x1_1_25_p25_25_4_float32.tflite (109436 bytes)
    4258752, // lstm_aaan_x1_1_25_p25_25_4_int8.tflite (71768 bytes)
    4330528, // lstm_aaao_x1_1_25_p25_25_5_float32.tflite (136356 bytes)
    4466896, // lstm_aaao_x1_1_25_p25_25_5_int8.tflite (89368 bytes)
    4556272, // lstm_aaap_x1_1_25_p25_25_6_float32.tflite (163260 bytes)
    4719536, // lstm_aaap_x1_1_25_p25_25_6_int8.tflite (106960 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    26396, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    16352, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite
    42432, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    44560, // lstm_aaab_x5_1_25_p25_25_1_int8.tflite
    62312, // lstm_aaac_x10_1_25_p25_25_1_float32.tflite
    79512, // lstm_aaac_x10_1_25_p25_25_1_int8.tflite
    83408, // lstm_aaad_x15_1_25_p25_25_1_float32.tflite
    115912, // lstm_aaad_x15_1_25_p25_25_1_int8.tflite
    103396, // lstm_aaae_x20_1_25_p25_25_1_float32.tflite
    151080, // lstm_aaae_x20_1_25_p25_25_1_int8.tflite
    26396, // lstm_aaaf_x1_1_25_p25_25_1_float32.tflite
    16352, // lstm_aaaf_x1_1_25_p25_25_1_int8.tflite
    86396, // lstm_aaag_x1_1_50_p50_50_1_float32.tflite
    33760, // lstm_aaag_x1_1_50_p50_50_1_int8.tflite
    326396, // lstm_aaah_x1_1_100_p100_100_1_float32.tflite
    98560, // lstm_aaah_x1_1_100_p100_100_1_int8.tflite
    726396, // lstm_aaai_x1_1_150_p150_150_1_float32.tflite
    203360, // lstm_aaai_x1_1_150_p150_150_1_int8.tflite
    1286396, // lstm_aaaj_x1_1_200_p200_200_1_float32.tflite
    348160, // lstm_aaaj_x1_1_200_p200_200_1_int8.tflite
    26396, // lstm_aaak_x1_1_25_p25_25_1_float32.tflite
    16352, // lstm_aaak_x1_1_25_p25_25_1_int8.tflite
    55636, // lstm_aaal_x1_1_25_p25_25_2_float32.tflite
    36584, // lstm_aaal_x1_1_25_p25_25_2_int8.tflite
    82536, // lstm_aaam_x1_1_25_p25_25_3_float32.tflite
    54176, // lstm_aaam_x1_1_25_p25_25_3_int8.tflite
    109436, // lstm_aaan_x1_1_25_p25_25_4_float32.tflite
    71768, // lstm_aaan_x1_1_25_p25_25_4_int8.tflite
    136356, // lstm_aaao_x1_1_25_p25_25_5_float32.tflite
    89368, // lstm_aaao_x1_1_25_p25_25_5_int8.tflite
    163260, // lstm_aaap_x1_1_25_p25_25_6_float32.tflite
    106960, // lstm_aaap_x1_1_25_p25_25_6_int8.tflite
};
