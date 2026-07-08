#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Model-Samples-Final-Old-Steps-Only/batch_0
// Blob: models_0.bin (6543016 bytes, align=16)
// Layout per config (slot a..f):
//   0 a: firstA float32   3 d: firstA int8
//   1 b: firstB float32   4 e: firstB int8
//   2 c: second float32   5 f: second int8
// Model index = cfg * 6 + slot.
// firstA/firstB: 1 input / 1 output;  second: 3 inputs / 3 outputs.

#define MODEL_COUNT 48
#define CONFIG_COUNT 6
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
    1, // 01_w0500_s2_a_firstA_float32_firstA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_b_firstB_float32_firstB_emg_x1_8_500_p500_500_2.tflite
    3, // 01_w0500_s2_c_second_float32_secondA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_d_second_float32_secondB_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_e_first_int8_firstA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_f_first_int8_firstB_emg_x1_8_500_p500_500_2.tflite
    3, // 01_w0500_s2_g_second_int8_secondA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_h_second_int8_secondB_emg_x1_8_500_p500_500_2.tflite
    1, // 02_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 02_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 02_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
    1, // 03_w0500_s10_a_firstA_float32_firstA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_b_firstB_float32_firstB_emg_x1_8_500_p500_500_10.tflite
    3, // 03_w0500_s10_c_second_float32_secondA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_d_second_float32_secondB_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_e_first_int8_firstA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_f_first_int8_firstB_emg_x1_8_500_p500_500_10.tflite
    3, // 03_w0500_s10_g_second_int8_secondA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_h_second_int8_secondB_emg_x1_8_500_p500_500_10.tflite
    1, // 04_w0500_s25_a_firstA_float32_firstA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_b_firstB_float32_firstB_emg_x1_8_500_p500_500_25.tflite
    3, // 04_w0500_s25_c_second_float32_secondA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_d_second_float32_secondB_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_e_first_int8_firstA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_f_first_int8_firstB_emg_x1_8_500_p500_500_25.tflite
    3, // 04_w0500_s25_g_second_int8_secondA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_h_second_int8_secondB_emg_x1_8_500_p500_500_25.tflite
    1, // 05_w0500_s50_a_firstA_float32_firstA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_b_firstB_float32_firstB_emg_x1_8_500_p500_500_50.tflite
    3, // 05_w0500_s50_c_second_float32_secondA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_d_second_float32_secondB_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_e_first_int8_firstA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_f_first_int8_firstB_emg_x1_8_500_p500_500_50.tflite
    3, // 05_w0500_s50_g_second_int8_secondA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_h_second_int8_secondB_emg_x1_8_500_p500_500_50.tflite
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
    1, // 01_w0500_s2_a_firstA_float32_firstA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_b_firstB_float32_firstB_emg_x1_8_500_p500_500_2.tflite
    3, // 01_w0500_s2_c_second_float32_secondA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_d_second_float32_secondB_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_e_first_int8_firstA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_f_first_int8_firstB_emg_x1_8_500_p500_500_2.tflite
    3, // 01_w0500_s2_g_second_int8_secondA_emg_x1_8_500_p500_500_2.tflite
    1, // 01_w0500_s2_h_second_int8_secondB_emg_x1_8_500_p500_500_2.tflite
    1, // 02_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 02_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 02_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 02_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
    1, // 03_w0500_s10_a_firstA_float32_firstA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_b_firstB_float32_firstB_emg_x1_8_500_p500_500_10.tflite
    3, // 03_w0500_s10_c_second_float32_secondA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_d_second_float32_secondB_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_e_first_int8_firstA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_f_first_int8_firstB_emg_x1_8_500_p500_500_10.tflite
    3, // 03_w0500_s10_g_second_int8_secondA_emg_x1_8_500_p500_500_10.tflite
    1, // 03_w0500_s10_h_second_int8_secondB_emg_x1_8_500_p500_500_10.tflite
    1, // 04_w0500_s25_a_firstA_float32_firstA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_b_firstB_float32_firstB_emg_x1_8_500_p500_500_25.tflite
    3, // 04_w0500_s25_c_second_float32_secondA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_d_second_float32_secondB_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_e_first_int8_firstA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_f_first_int8_firstB_emg_x1_8_500_p500_500_25.tflite
    3, // 04_w0500_s25_g_second_int8_secondA_emg_x1_8_500_p500_500_25.tflite
    1, // 04_w0500_s25_h_second_int8_secondB_emg_x1_8_500_p500_500_25.tflite
    1, // 05_w0500_s50_a_firstA_float32_firstA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_b_firstB_float32_firstB_emg_x1_8_500_p500_500_50.tflite
    3, // 05_w0500_s50_c_second_float32_secondA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_d_second_float32_secondB_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_e_first_int8_firstA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_f_first_int8_firstB_emg_x1_8_500_p500_500_50.tflite
    3, // 05_w0500_s50_g_second_int8_secondA_emg_x1_8_500_p500_500_50.tflite
    1, // 05_w0500_s50_h_second_int8_secondB_emg_x1_8_500_p500_500_50.tflite
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
    371344, // 01_w0500_s2_a_firstA_float32_firstA_emg_x1_8_500_p500_500_2.tflite (15084 bytes)
    386432, // 01_w0500_s2_b_firstB_float32_firstB_emg_x1_8_500_p500_500_2.tflite (5468 bytes)
    391904, // 01_w0500_s2_c_second_float32_secondA_emg_x1_8_500_p500_500_2.tflite (236812 bytes)
    628720, // 01_w0500_s2_d_second_float32_secondB_emg_x1_8_500_p500_500_2.tflite (8648 bytes)
    637376, // 01_w0500_s2_e_first_int8_firstA_emg_x1_8_500_p500_500_2.tflite (8776 bytes)
    646160, // 01_w0500_s2_f_first_int8_firstB_emg_x1_8_500_p500_500_2.tflite (5392 bytes)
    651552, // 01_w0500_s2_g_second_int8_secondA_emg_x1_8_500_p500_500_2.tflite (133624 bytes)
    785184, // 01_w0500_s2_h_second_int8_secondB_emg_x1_8_500_p500_500_2.tflite (6200 bytes)
    791392, // 02_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite (15084 bytes)
    806480, // 02_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite (5468 bytes)
    811952, // 02_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite (281284 bytes)
    1093248, // 02_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite (8648 bytes)
    1101904, // 02_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite (8776 bytes)
    1110688, // 02_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite (5392 bytes)
    1116080, // 02_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite (235592 bytes)
    1351680, // 02_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite (6200 bytes)
    1357888, // 03_w0500_s10_a_firstA_float32_firstA_emg_x1_8_500_p500_500_10.tflite (15084 bytes)
    1372976, // 03_w0500_s10_b_firstB_float32_firstB_emg_x1_8_500_p500_500_10.tflite (5468 bytes)
    1378448, // 03_w0500_s10_c_second_float32_secondA_emg_x1_8_500_p500_500_10.tflite (355404 bytes)
    1733856, // 03_w0500_s10_d_second_float32_secondB_emg_x1_8_500_p500_500_10.tflite (8648 bytes)
    1742512, // 03_w0500_s10_e_first_int8_firstA_emg_x1_8_500_p500_500_10.tflite (8776 bytes)
    1751296, // 03_w0500_s10_f_first_int8_firstB_emg_x1_8_500_p500_500_10.tflite (5392 bytes)
    1756688, // 03_w0500_s10_g_second_int8_secondA_emg_x1_8_500_p500_500_10.tflite (406088 bytes)
    2162784, // 03_w0500_s10_h_second_int8_secondB_emg_x1_8_500_p500_500_10.tflite (6200 bytes)
    2168992, // 04_w0500_s25_a_firstA_float32_firstA_emg_x1_8_500_p500_500_25.tflite (15084 bytes)
    2184080, // 04_w0500_s25_b_firstB_float32_firstB_emg_x1_8_500_p500_500_25.tflite (5468 bytes)
    2189552, // 04_w0500_s25_c_second_float32_secondA_emg_x1_8_500_p500_500_25.tflite (579460 bytes)
    2769024, // 04_w0500_s25_d_second_float32_secondB_emg_x1_8_500_p500_500_25.tflite (8648 bytes)
    2777680, // 04_w0500_s25_e_first_int8_firstA_emg_x1_8_500_p500_500_25.tflite (8776 bytes)
    2786464, // 04_w0500_s25_f_first_int8_firstB_emg_x1_8_500_p500_500_25.tflite (5392 bytes)
    2791856, // 04_w0500_s25_g_second_int8_secondA_emg_x1_8_500_p500_500_25.tflite (932256 bytes)
    3724112, // 04_w0500_s25_h_second_int8_secondB_emg_x1_8_500_p500_500_25.tflite (6200 bytes)
    3730320, // 05_w0500_s50_a_firstA_float32_firstA_emg_x1_8_500_p500_500_50.tflite (15084 bytes)
    3745408, // 05_w0500_s50_b_firstB_float32_firstB_emg_x1_8_500_p500_500_50.tflite (5468 bytes)
    3750880, // 05_w0500_s50_c_second_float32_secondA_emg_x1_8_500_p500_500_50.tflite (951080 bytes)
    4701968, // 05_w0500_s50_d_second_float32_secondB_emg_x1_8_500_p500_500_50.tflite (8648 bytes)
    4710624, // 05_w0500_s50_e_first_int8_firstA_emg_x1_8_500_p500_500_50.tflite (8776 bytes)
    4719408, // 05_w0500_s50_f_first_int8_firstB_emg_x1_8_500_p500_500_50.tflite (5392 bytes)
    4724800, // 05_w0500_s50_g_second_int8_secondA_emg_x1_8_500_p500_500_50.tflite (1812016 bytes)
    6536816, // 05_w0500_s50_h_second_int8_secondB_emg_x1_8_500_p500_500_50.tflite (6200 bytes)
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
    15084, // 01_w0500_s2_a_firstA_float32_firstA_emg_x1_8_500_p500_500_2.tflite
    5468, // 01_w0500_s2_b_firstB_float32_firstB_emg_x1_8_500_p500_500_2.tflite
    236812, // 01_w0500_s2_c_second_float32_secondA_emg_x1_8_500_p500_500_2.tflite
    8648, // 01_w0500_s2_d_second_float32_secondB_emg_x1_8_500_p500_500_2.tflite
    8776, // 01_w0500_s2_e_first_int8_firstA_emg_x1_8_500_p500_500_2.tflite
    5392, // 01_w0500_s2_f_first_int8_firstB_emg_x1_8_500_p500_500_2.tflite
    133624, // 01_w0500_s2_g_second_int8_secondA_emg_x1_8_500_p500_500_2.tflite
    6200, // 01_w0500_s2_h_second_int8_secondB_emg_x1_8_500_p500_500_2.tflite
    15084, // 02_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    5468, // 02_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    281284, // 02_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    8648, // 02_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    8776, // 02_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    5392, // 02_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    235592, // 02_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    6200, // 02_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
    15084, // 03_w0500_s10_a_firstA_float32_firstA_emg_x1_8_500_p500_500_10.tflite
    5468, // 03_w0500_s10_b_firstB_float32_firstB_emg_x1_8_500_p500_500_10.tflite
    355404, // 03_w0500_s10_c_second_float32_secondA_emg_x1_8_500_p500_500_10.tflite
    8648, // 03_w0500_s10_d_second_float32_secondB_emg_x1_8_500_p500_500_10.tflite
    8776, // 03_w0500_s10_e_first_int8_firstA_emg_x1_8_500_p500_500_10.tflite
    5392, // 03_w0500_s10_f_first_int8_firstB_emg_x1_8_500_p500_500_10.tflite
    406088, // 03_w0500_s10_g_second_int8_secondA_emg_x1_8_500_p500_500_10.tflite
    6200, // 03_w0500_s10_h_second_int8_secondB_emg_x1_8_500_p500_500_10.tflite
    15084, // 04_w0500_s25_a_firstA_float32_firstA_emg_x1_8_500_p500_500_25.tflite
    5468, // 04_w0500_s25_b_firstB_float32_firstB_emg_x1_8_500_p500_500_25.tflite
    579460, // 04_w0500_s25_c_second_float32_secondA_emg_x1_8_500_p500_500_25.tflite
    8648, // 04_w0500_s25_d_second_float32_secondB_emg_x1_8_500_p500_500_25.tflite
    8776, // 04_w0500_s25_e_first_int8_firstA_emg_x1_8_500_p500_500_25.tflite
    5392, // 04_w0500_s25_f_first_int8_firstB_emg_x1_8_500_p500_500_25.tflite
    932256, // 04_w0500_s25_g_second_int8_secondA_emg_x1_8_500_p500_500_25.tflite
    6200, // 04_w0500_s25_h_second_int8_secondB_emg_x1_8_500_p500_500_25.tflite
    15084, // 05_w0500_s50_a_firstA_float32_firstA_emg_x1_8_500_p500_500_50.tflite
    5468, // 05_w0500_s50_b_firstB_float32_firstB_emg_x1_8_500_p500_500_50.tflite
    951080, // 05_w0500_s50_c_second_float32_secondA_emg_x1_8_500_p500_500_50.tflite
    8648, // 05_w0500_s50_d_second_float32_secondB_emg_x1_8_500_p500_500_50.tflite
    8776, // 05_w0500_s50_e_first_int8_firstA_emg_x1_8_500_p500_500_50.tflite
    5392, // 05_w0500_s50_f_first_int8_firstB_emg_x1_8_500_p500_500_50.tflite
    1812016, // 05_w0500_s50_g_second_int8_secondA_emg_x1_8_500_p500_500_50.tflite
    6200, // 05_w0500_s50_h_second_int8_secondB_emg_x1_8_500_p500_500_50.tflite
};
