#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/fidelity-refactored-verification/batch_0
// Blob: models_0.bin (371344 bytes, align=16)
// Layout per config (slot a..f):
//   0 a: firstA float32   3 d: firstA int8
//   1 b: firstB float32   4 e: firstB int8
//   2 c: second float32   5 f: second int8
// Model index = cfg * 6 + slot.
// firstA/firstB: 1 input / 1 output;  second: 3 inputs / 3 outputs.

#define MODEL_COUNT 8
#define CONFIG_COUNT 1
#define MODELS_PER_CONFIG 8

// Number of input tensors per model file (1 = first block, 3 = second block)
static const uint32_t INPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 00_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 00_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
};

// Number of output tensors per model file (1 = first block, 3 = second block)
static const uint32_t OUTPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 00_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 00_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 00_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
};

static const uint32_t MODEL_OFFSETS[MODEL_COUNT] = {
    0, // 00_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite (15084 bytes)
    15088, // 00_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite (5468 bytes)
    20560, // 00_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite (222312 bytes)
    242880, // 00_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite (8408 bytes)
    251296, // 00_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite (8776 bytes)
    260080, // 00_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite (5392 bytes)
    265472, // 00_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite (99952 bytes)
    365424, // 00_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite (5920 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    15084, // 00_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    5468, // 00_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    222312, // 00_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    8408, // 00_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    8776, // 00_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    5392, // 00_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    99952, // 00_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    5920, // 00_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
};
