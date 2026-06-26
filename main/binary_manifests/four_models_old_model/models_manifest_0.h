#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Model-Samples-Final-Old-Model/batch_0
// Blob: models_0.bin (9390136 bytes, align=16)
// Layout per config (slot a..f):
//   0 a: firstA float32   3 d: firstA int8
//   1 b: firstB float32   4 e: firstB int8
//   2 c: second float32   5 f: second int8
// Model index = cfg * 6 + slot.
// firstA/firstB: 1 input / 1 output;  second: 3 inputs / 3 outputs.

#define MODEL_COUNT 160
#define CONFIG_COUNT 20
#define MODELS_PER_CONFIG 8

// Number of input tensors per model file (1 = first block, 3 = second block)
static const uint32_t INPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite
    1, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite
    1, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite
    1, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite
    1, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite
    1, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite
    1, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite
    1, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite
    1, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite
    1, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite
    1, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite
    1, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite
    1, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite
    1, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite
    1, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite
    1, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite
    1, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite
    1, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite
    1, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
    1, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
};

// Number of output tensors per model file (1 = first block, 3 = second block)
static const uint32_t OUTPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite
    1, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite
    1, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite
    1, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite
    1, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite
    1, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite
    1, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite
    1, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite
    1, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite
    1, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite
    1, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite
    1, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite
    1, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite
    1, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite
    1, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite
    1, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite
    1, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite
    1, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite
    1, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
    1, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
};

static const uint32_t MODEL_OFFSETS[MODEL_COUNT] = {
    0, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite (15084 bytes)
    15088, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite (5468 bytes)
    20560, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite (222312 bytes)
    242880, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite (8408 bytes)
    251296, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite (8776 bytes)
    260080, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite (5392 bytes)
    265472, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite (99952 bytes)
    365424, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite (5920 bytes)
    371344, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite (15084 bytes)
    386432, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite (5468 bytes)
    391904, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite (281284 bytes)
    673200, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite (8648 bytes)
    681856, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite (8776 bytes)
    690640, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite (5392 bytes)
    696032, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite (240112 bytes)
    936144, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite (6200 bytes)
    942352, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite (15084 bytes)
    957440, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite (5468 bytes)
    962912, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite (222312 bytes)
    1185232, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite (8408 bytes)
    1193648, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite (8776 bytes)
    1202432, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite (5392 bytes)
    1207824, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite (99952 bytes)
    1307776, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite (5920 bytes)
    1313696, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite (15084 bytes)
    1328784, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite (5468 bytes)
    1334256, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite (281284 bytes)
    1615552, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite (8648 bytes)
    1624208, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite (8776 bytes)
    1632992, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite (5392 bytes)
    1638384, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite (235608 bytes)
    1874000, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite (6200 bytes)
    1880208, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite (15084 bytes)
    1895296, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite (5468 bytes)
    1900768, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite (222312 bytes)
    2123088, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite (8408 bytes)
    2131504, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite (8776 bytes)
    2140288, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite (5392 bytes)
    2145680, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite (99952 bytes)
    2245632, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite (5920 bytes)
    2251552, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite (15084 bytes)
    2266640, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite (5468 bytes)
    2272112, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite (281284 bytes)
    2553408, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite (8648 bytes)
    2562064, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite (8776 bytes)
    2570848, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite (5392 bytes)
    2576240, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite (237416 bytes)
    2813664, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite (6200 bytes)
    2819872, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite (15084 bytes)
    2834960, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite (5468 bytes)
    2840432, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite (222312 bytes)
    3062752, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite (8408 bytes)
    3071168, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite (8776 bytes)
    3079952, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite (5392 bytes)
    3085344, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite (99952 bytes)
    3185296, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite (5920 bytes)
    3191216, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite (15084 bytes)
    3206304, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite (5468 bytes)
    3211776, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite (281284 bytes)
    3493072, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite (8648 bytes)
    3501728, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite (8776 bytes)
    3510512, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite (5392 bytes)
    3515904, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite (237400 bytes)
    3753312, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite (6200 bytes)
    3759520, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite (15084 bytes)
    3774608, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite (5468 bytes)
    3780080, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite (222312 bytes)
    4002400, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite (8408 bytes)
    4010816, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite (8776 bytes)
    4019600, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite (5392 bytes)
    4024992, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite (99952 bytes)
    4124944, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite (5920 bytes)
    4130864, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite (15084 bytes)
    4145952, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite (5468 bytes)
    4151424, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite (281284 bytes)
    4432720, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite (8648 bytes)
    4441376, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite (8776 bytes)
    4450160, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite (5392 bytes)
    4455552, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite (235592 bytes)
    4691152, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite (6200 bytes)
    4697360, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite (15084 bytes)
    4712448, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite (5468 bytes)
    4717920, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite (222312 bytes)
    4940240, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite (8408 bytes)
    4948656, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite (8776 bytes)
    4957440, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite (5392 bytes)
    4962832, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite (99952 bytes)
    5062784, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite (5920 bytes)
    5068704, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite (15084 bytes)
    5083792, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite (5468 bytes)
    5089264, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite (281284 bytes)
    5370560, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite (8648 bytes)
    5379216, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite (8776 bytes)
    5388000, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite (5392 bytes)
    5393392, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite (236496 bytes)
    5629888, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite (6200 bytes)
    5636096, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite (15084 bytes)
    5651184, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite (5468 bytes)
    5656656, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite (222312 bytes)
    5878976, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite (8408 bytes)
    5887392, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite (8776 bytes)
    5896176, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite (5392 bytes)
    5901568, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite (99952 bytes)
    6001520, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite (5920 bytes)
    6007440, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite (15084 bytes)
    6022528, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite (5468 bytes)
    6028000, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite (281284 bytes)
    6309296, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite (8648 bytes)
    6317952, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite (8776 bytes)
    6326736, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite (5392 bytes)
    6332128, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite (236496 bytes)
    6568624, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite (6200 bytes)
    6574832, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite (15084 bytes)
    6589920, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite (5468 bytes)
    6595392, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite (222312 bytes)
    6817712, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite (8408 bytes)
    6826128, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite (8776 bytes)
    6834912, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite (5392 bytes)
    6840304, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite (99952 bytes)
    6940256, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite (5920 bytes)
    6946176, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite (15084 bytes)
    6961264, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite (5468 bytes)
    6966736, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite (281284 bytes)
    7248032, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite (8648 bytes)
    7256688, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite (8776 bytes)
    7265472, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite (5392 bytes)
    7270864, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite (236496 bytes)
    7507360, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite (6200 bytes)
    7513568, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite (15084 bytes)
    7528656, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite (5468 bytes)
    7534128, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite (222312 bytes)
    7756448, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite (8408 bytes)
    7764864, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite (8776 bytes)
    7773648, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite (5392 bytes)
    7779040, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite (99952 bytes)
    7878992, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite (5920 bytes)
    7884912, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite (15084 bytes)
    7900000, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite (5468 bytes)
    7905472, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite (281284 bytes)
    8186768, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite (8648 bytes)
    8195424, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite (8776 bytes)
    8204208, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite (5392 bytes)
    8209600, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite (236496 bytes)
    8446096, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite (6200 bytes)
    8452304, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite (15084 bytes)
    8467392, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite (5468 bytes)
    8472864, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite (222312 bytes)
    8695184, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite (8408 bytes)
    8703600, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite (8776 bytes)
    8712384, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite (5392 bytes)
    8717776, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite (99952 bytes)
    8817728, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite (5920 bytes)
    8823648, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite (15084 bytes)
    8838736, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite (5468 bytes)
    8844208, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite (281284 bytes)
    9125504, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite (8648 bytes)
    9134160, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite (8776 bytes)
    9142944, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite (5392 bytes)
    9148336, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite (235592 bytes)
    9383936, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite (6200 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    15084, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    5468, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    222312, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite
    8408, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite
    8776, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    5392, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    99952, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite
    5920, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite
    15084, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    5468, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    281284, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite
    8648, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite
    8776, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    5392, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    240112, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite
    6200, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite
    15084, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    5468, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    222312, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite
    8408, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite
    8776, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    5392, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    99952, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite
    5920, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite
    15084, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    5468, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    281284, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite
    8648, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite
    8776, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    5392, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    235608, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite
    6200, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite
    15084, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    5468, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    222312, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite
    8408, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite
    8776, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    5392, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    99952, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite
    5920, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite
    15084, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    5468, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    281284, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite
    8648, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite
    8776, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    5392, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    237416, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite
    6200, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite
    15084, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    5468, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    222312, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite
    8408, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite
    8776, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    5392, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    99952, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite
    5920, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite
    15084, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    5468, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    281284, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite
    8648, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite
    8776, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    5392, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    237400, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite
    6200, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite
    15084, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    5468, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    222312, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite
    8408, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite
    8776, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    5392, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    99952, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite
    5920, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite
    15084, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    5468, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    281284, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite
    8648, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite
    8776, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    5392, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    235592, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite
    6200, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite
    15084, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    5468, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    222312, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite
    8408, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite
    8776, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    5392, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    99952, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite
    5920, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite
    15084, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    5468, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    281284, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite
    8648, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite
    8776, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    5392, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    236496, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite
    6200, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite
    15084, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    5468, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    222312, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite
    8408, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite
    8776, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    5392, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    99952, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite
    5920, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite
    15084, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    5468, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    281284, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite
    8648, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite
    8776, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    5392, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    236496, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite
    6200, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite
    15084, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    5468, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    222312, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite
    8408, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite
    8776, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    5392, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    99952, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite
    5920, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite
    15084, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    5468, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    281284, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite
    8648, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite
    8776, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    5392, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    236496, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite
    6200, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite
    15084, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    5468, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    222312, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite
    8408, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite
    8776, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    5392, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    99952, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite
    5920, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite
    15084, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    5468, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    281284, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite
    8648, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite
    8776, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    5392, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    236496, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite
    6200, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite
    15084, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    5468, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    222312, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    8408, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    8776, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    5392, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    99952, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    5920, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
    15084, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    5468, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    281284, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    8648, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    8776, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    5392, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    235592, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    6200, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
};
