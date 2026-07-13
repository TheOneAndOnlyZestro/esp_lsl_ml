#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/lstm-layers-verification/batch_0
// Blob: models_0.bin (906144 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 10

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_2_float32.tflite
    3, // lstm_aaab_x1_1_25_p25_25_3_float32.tflite
    3, // lstm_aaac_x1_1_25_p25_25_4_float32.tflite
    3, // lstm_aaad_x1_1_25_p25_25_5_float32.tflite
    3, // lstm_aaae_x1_1_25_p25_25_6_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_2_float32.tflite
    3, // lstm_aaab_x1_1_25_p25_25_3_float32.tflite
    3, // lstm_aaac_x1_1_25_p25_25_4_float32.tflite
    3, // lstm_aaad_x1_1_25_p25_25_5_float32.tflite
    3, // lstm_aaae_x1_1_25_p25_25_6_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // lstm_aaaa_x1_1_25_p25_25_2_float32.tflite (55636 bytes)
    55648, // lstm_aaaa_x1_1_25_p25_25_2_int8.tflite (36584 bytes)
    92240, // lstm_aaab_x1_1_25_p25_25_3_float32.tflite (82536 bytes)
    174784, // lstm_aaab_x1_1_25_p25_25_3_int8.tflite (54176 bytes)
    228960, // lstm_aaac_x1_1_25_p25_25_4_float32.tflite (109436 bytes)
    338400, // lstm_aaac_x1_1_25_p25_25_4_int8.tflite (71768 bytes)
    410176, // lstm_aaad_x1_1_25_p25_25_5_float32.tflite (136356 bytes)
    546544, // lstm_aaad_x1_1_25_p25_25_5_int8.tflite (89368 bytes)
    635920, // lstm_aaae_x1_1_25_p25_25_6_float32.tflite (163260 bytes)
    799184, // lstm_aaae_x1_1_25_p25_25_6_int8.tflite (106960 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    55636, // lstm_aaaa_x1_1_25_p25_25_2_float32.tflite
    36584, // lstm_aaaa_x1_1_25_p25_25_2_int8.tflite
    82536, // lstm_aaab_x1_1_25_p25_25_3_float32.tflite
    54176, // lstm_aaab_x1_1_25_p25_25_3_int8.tflite
    109436, // lstm_aaac_x1_1_25_p25_25_4_float32.tflite
    71768, // lstm_aaac_x1_1_25_p25_25_4_int8.tflite
    136356, // lstm_aaad_x1_1_25_p25_25_5_float32.tflite
    89368, // lstm_aaad_x1_1_25_p25_25_5_int8.tflite
    163260, // lstm_aaae_x1_1_25_p25_25_6_float32.tflite
    106960, // lstm_aaae_x1_1_25_p25_25_6_int8.tflite
};
