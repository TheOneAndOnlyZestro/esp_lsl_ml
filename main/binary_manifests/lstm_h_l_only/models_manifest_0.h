#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-LSTM-h-l/batch_0
// Blob: models_0.bin (8708296 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 16

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_10_p10_10_1_float32.tflite
    3, // lstm_aaab_x1_1_60_p60_60_1_float32.tflite
    3, // lstm_aaac_x1_1_200_p200_200_1_float32.tflite
    3, // lstm_aaad_x1_1_400_p400_400_1_float32.tflite
    3, // lstm_aaae_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaaf_x1_1_25_p25_25_2_float32.tflite
    3, // lstm_aaag_x1_1_25_p25_25_3_float32.tflite
    3, // lstm_aaah_x1_1_25_p25_25_4_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_10_p10_10_1_float32.tflite
    3, // lstm_aaab_x1_1_60_p60_60_1_float32.tflite
    3, // lstm_aaac_x1_1_200_p200_200_1_float32.tflite
    3, // lstm_aaad_x1_1_400_p400_400_1_float32.tflite
    3, // lstm_aaae_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaaf_x1_1_25_p25_25_2_float32.tflite
    3, // lstm_aaag_x1_1_25_p25_25_3_float32.tflite
    3, // lstm_aaah_x1_1_25_p25_25_4_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // lstm_aaaa_x1_1_10_p10_10_1_float32.tflite (9068 bytes)
    9072, // lstm_aaaa_x1_1_10_p10_10_1_int8.tflite (10208 bytes)
    19280, // lstm_aaab_x1_1_60_p60_60_1_float32.tflite (121068 bytes)
    140352, // lstm_aaab_x1_1_60_p60_60_1_int8.tflite (43008 bytes)
    183360, // lstm_aaac_x1_1_200_p200_200_1_float32.tflite (1285868 bytes)
    1469232, // lstm_aaac_x1_1_200_p200_200_1_int8.tflite (347648 bytes)
    1816880, // lstm_aaad_x1_1_400_p400_400_1_float32.tflite (5125868 bytes)
    6942752, // lstm_aaad_x1_1_400_p400_400_1_int8.tflite (1326848 bytes)
    8269600, // lstm_aaae_x1_1_25_p25_25_1_float32.tflite (25868 bytes)
    8295472, // lstm_aaae_x1_1_25_p25_25_1_int8.tflite (15840 bytes)
    8311312, // lstm_aaaf_x1_1_25_p25_25_2_float32.tflite (54188 bytes)
    8365504, // lstm_aaaf_x1_1_25_p25_25_2_int8.tflite (35120 bytes)
    8400624, // lstm_aaag_x1_1_25_p25_25_3_float32.tflite (80348 bytes)
    8480976, // lstm_aaag_x1_1_25_p25_25_3_int8.tflite (51976 bytes)
    8532960, // lstm_aaah_x1_1_25_p25_25_4_float32.tflite (106508 bytes)
    8639472, // lstm_aaah_x1_1_25_p25_25_4_int8.tflite (68824 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    9068, // lstm_aaaa_x1_1_10_p10_10_1_float32.tflite
    10208, // lstm_aaaa_x1_1_10_p10_10_1_int8.tflite
    121068, // lstm_aaab_x1_1_60_p60_60_1_float32.tflite
    43008, // lstm_aaab_x1_1_60_p60_60_1_int8.tflite
    1285868, // lstm_aaac_x1_1_200_p200_200_1_float32.tflite
    347648, // lstm_aaac_x1_1_200_p200_200_1_int8.tflite
    5125868, // lstm_aaad_x1_1_400_p400_400_1_float32.tflite
    1326848, // lstm_aaad_x1_1_400_p400_400_1_int8.tflite
    25868, // lstm_aaae_x1_1_25_p25_25_1_float32.tflite
    15840, // lstm_aaae_x1_1_25_p25_25_1_int8.tflite
    54188, // lstm_aaaf_x1_1_25_p25_25_2_float32.tflite
    35120, // lstm_aaaf_x1_1_25_p25_25_2_int8.tflite
    80348, // lstm_aaag_x1_1_25_p25_25_3_float32.tflite
    51976, // lstm_aaag_x1_1_25_p25_25_3_int8.tflite
    106508, // lstm_aaah_x1_1_25_p25_25_4_float32.tflite
    68824, // lstm_aaah_x1_1_25_p25_25_4_int8.tflite
};
