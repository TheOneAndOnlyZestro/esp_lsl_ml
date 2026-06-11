#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Comprehensive-Brute-LSTM-win/batch_0
// Blob: models_0.bin (4578336 bytes, align=16)

#define BENCHMARK_MODEL_COUNT 10

static const uint32_t BENCHMARK_INPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    3, // lstm_aaac_x25_1_25_p25_25_1_float32.tflite
    3, // lstm_aaad_x100_1_25_p25_25_1_float32.tflite
    3, // lstm_aaae_x300_1_25_p25_25_1_float32.tflite
};

static const uint32_t BENCHMARK_OUTPUT_SIZES[(int)(BENCHMARK_MODEL_COUNT/2)] = {
    3, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    3, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    3, // lstm_aaac_x25_1_25_p25_25_1_float32.tflite
    3, // lstm_aaad_x100_1_25_p25_25_1_float32.tflite
    3, // lstm_aaae_x300_1_25_p25_25_1_float32.tflite
};

static const uint32_t BENCHMARK_MODEL_OFFSETS[BENCHMARK_MODEL_COUNT] = {
    0, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite (25868 bytes)
    25872, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite (15840 bytes)
    41712, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite (40328 bytes)
    82048, // lstm_aaab_x5_1_25_p25_25_1_int8.tflite (42288 bytes)
    124336, // lstm_aaac_x25_1_25_p25_25_1_float32.tflite (113524 bytes)
    237872, // lstm_aaac_x25_1_25_p25_25_1_int8.tflite (174816 bytes)
    412688, // lstm_aaad_x100_1_25_p25_25_1_float32.tflite (386308 bytes)
    799008, // lstm_aaad_x100_1_25_p25_25_1_int8.tflite (669456 bytes)
    1468464, // lstm_aaae_x300_1_25_p25_25_1_float32.tflite (1116992 bytes)
    2585456, // lstm_aaae_x300_1_25_p25_25_1_int8.tflite (1992880 bytes)
};

static const uint32_t BENCHMARK_MODEL_SIZES[BENCHMARK_MODEL_COUNT] = {
    25868, // lstm_aaaa_x1_1_25_p25_25_1_float32.tflite
    15840, // lstm_aaaa_x1_1_25_p25_25_1_int8.tflite
    40328, // lstm_aaab_x5_1_25_p25_25_1_float32.tflite
    42288, // lstm_aaab_x5_1_25_p25_25_1_int8.tflite
    113524, // lstm_aaac_x25_1_25_p25_25_1_float32.tflite
    174816, // lstm_aaac_x25_1_25_p25_25_1_int8.tflite
    386308, // lstm_aaad_x100_1_25_p25_25_1_float32.tflite
    669456, // lstm_aaad_x100_1_25_p25_25_1_int8.tflite
    1116992, // lstm_aaae_x300_1_25_p25_25_1_float32.tflite
    1992880, // lstm_aaae_x300_1_25_p25_25_1_int8.tflite
};
