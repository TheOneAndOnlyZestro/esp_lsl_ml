#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Model-Samples-ESPNN-LSTM-NEW-MODEL/batch_0
// Blob: models_0.bin (9141280 bytes, align=16)
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
    0, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite (9320 bytes)
    9328, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite (5468 bytes)
    14800, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite (222312 bytes)
    237120, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite (5524 bytes)
    242656, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite (7904 bytes)
    250560, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite (5392 bytes)
    255952, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite (99952 bytes)
    355904, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite (4120 bytes)
    360032, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite (9320 bytes)
    369360, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite (5468 bytes)
    374832, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite (281284 bytes)
    656128, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite (5764 bytes)
    661904, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite (7904 bytes)
    669808, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite (5392 bytes)
    675200, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite (235624 bytes)
    910832, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite (4400 bytes)
    915232, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite (9320 bytes)
    924560, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite (5468 bytes)
    930032, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite (222312 bytes)
    1152352, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite (5524 bytes)
    1157888, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite (7904 bytes)
    1165792, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite (5392 bytes)
    1171184, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite (99952 bytes)
    1271136, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite (4120 bytes)
    1275264, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite (9320 bytes)
    1284592, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite (5468 bytes)
    1290064, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite (281284 bytes)
    1571360, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite (5764 bytes)
    1577136, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite (7904 bytes)
    1585040, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite (5392 bytes)
    1590432, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite (235608 bytes)
    1826048, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite (4400 bytes)
    1830448, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite (9320 bytes)
    1839776, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite (5468 bytes)
    1845248, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite (222312 bytes)
    2067568, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite (5524 bytes)
    2073104, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite (7904 bytes)
    2081008, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite (5392 bytes)
    2086400, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite (99952 bytes)
    2186352, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite (4120 bytes)
    2190480, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite (9320 bytes)
    2199808, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite (5468 bytes)
    2205280, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite (281284 bytes)
    2486576, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite (5764 bytes)
    2492352, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite (7904 bytes)
    2500256, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite (5392 bytes)
    2505648, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite (235608 bytes)
    2741264, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite (4400 bytes)
    2745664, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite (9320 bytes)
    2754992, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite (5468 bytes)
    2760464, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite (222312 bytes)
    2982784, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite (5524 bytes)
    2988320, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite (7904 bytes)
    2996224, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite (5392 bytes)
    3001616, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite (99952 bytes)
    3101568, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite (4120 bytes)
    3105696, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite (9320 bytes)
    3115024, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite (5468 bytes)
    3120496, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite (281284 bytes)
    3401792, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite (5764 bytes)
    3407568, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite (7904 bytes)
    3415472, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite (5392 bytes)
    3420864, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite (233800 bytes)
    3654672, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite (4400 bytes)
    3659072, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite (9320 bytes)
    3668400, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite (5468 bytes)
    3673872, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite (222312 bytes)
    3896192, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite (5524 bytes)
    3901728, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite (7904 bytes)
    3909632, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite (5392 bytes)
    3915024, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite (99952 bytes)
    4014976, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite (4120 bytes)
    4019104, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite (9320 bytes)
    4028432, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite (5468 bytes)
    4033904, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite (281284 bytes)
    4315200, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite (5764 bytes)
    4320976, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite (7904 bytes)
    4328880, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite (5392 bytes)
    4334272, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite (233784 bytes)
    4568064, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite (4400 bytes)
    4572464, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite (9320 bytes)
    4581792, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite (5468 bytes)
    4587264, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite (222312 bytes)
    4809584, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite (5524 bytes)
    4815120, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite (7904 bytes)
    4823024, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite (5392 bytes)
    4828416, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite (99952 bytes)
    4928368, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite (4120 bytes)
    4932496, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite (9320 bytes)
    4941824, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite (5468 bytes)
    4947296, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite (281284 bytes)
    5228592, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite (5764 bytes)
    5234368, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite (7904 bytes)
    5242272, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite (5392 bytes)
    5247664, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite (233784 bytes)
    5481456, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite (4400 bytes)
    5485856, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite (9320 bytes)
    5495184, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite (5468 bytes)
    5500656, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite (222312 bytes)
    5722976, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite (5524 bytes)
    5728512, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite (7904 bytes)
    5736416, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite (5392 bytes)
    5741808, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite (99952 bytes)
    5841760, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite (4120 bytes)
    5845888, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite (9320 bytes)
    5855216, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite (5468 bytes)
    5860688, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite (281284 bytes)
    6141984, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite (5764 bytes)
    6147760, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite (7904 bytes)
    6155664, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite (5392 bytes)
    6161056, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite (233784 bytes)
    6394848, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite (4400 bytes)
    6399248, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite (9320 bytes)
    6408576, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite (5468 bytes)
    6414048, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite (222312 bytes)
    6636368, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite (5524 bytes)
    6641904, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite (7904 bytes)
    6649808, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite (5392 bytes)
    6655200, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite (99952 bytes)
    6755152, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite (4120 bytes)
    6759280, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite (9320 bytes)
    6768608, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite (5468 bytes)
    6774080, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite (281284 bytes)
    7055376, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite (5764 bytes)
    7061152, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite (7904 bytes)
    7069056, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite (5392 bytes)
    7074448, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite (233816 bytes)
    7308272, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite (4400 bytes)
    7312672, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite (9320 bytes)
    7322000, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite (5468 bytes)
    7327472, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite (222312 bytes)
    7549792, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite (5524 bytes)
    7555328, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite (7904 bytes)
    7563232, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite (5392 bytes)
    7568624, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite (99952 bytes)
    7668576, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite (4120 bytes)
    7672704, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite (9320 bytes)
    7682032, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite (5468 bytes)
    7687504, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite (281284 bytes)
    7968800, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite (5764 bytes)
    7974576, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite (7904 bytes)
    7982480, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite (5392 bytes)
    7987872, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite (235608 bytes)
    8223488, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite (4400 bytes)
    8227888, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite (9320 bytes)
    8237216, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite (5468 bytes)
    8242688, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite (222312 bytes)
    8465008, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite (5524 bytes)
    8470544, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite (7904 bytes)
    8478448, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite (5392 bytes)
    8483840, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite (99952 bytes)
    8583792, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite (4120 bytes)
    8587920, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite (9320 bytes)
    8597248, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite (5468 bytes)
    8602720, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite (281284 bytes)
    8884016, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite (5764 bytes)
    8889792, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite (7904 bytes)
    8897696, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite (5392 bytes)
    8903088, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite (233784 bytes)
    9136880, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite (4400 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    9320, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    5468, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    222312, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite
    5524, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite
    7904, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    5392, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    99952, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite
    4120, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite
    9320, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    5468, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    281284, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite
    5764, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite
    7904, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    5392, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    235624, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite
    4400, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite
    9320, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    5468, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    222312, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite
    5524, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite
    7904, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    5392, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    99952, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite
    4120, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite
    9320, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    5468, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    281284, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite
    5764, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite
    7904, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    5392, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    235608, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite
    4400, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite
    9320, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    5468, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    222312, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite
    5524, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite
    7904, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    5392, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    99952, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite
    4120, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite
    9320, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    5468, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    281284, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite
    5764, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite
    7904, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    5392, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    235608, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite
    4400, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite
    9320, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    5468, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    222312, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite
    5524, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite
    7904, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    5392, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    99952, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite
    4120, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite
    9320, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    5468, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    281284, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite
    5764, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite
    7904, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    5392, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    233800, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite
    4400, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite
    9320, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    5468, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    222312, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite
    5524, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite
    7904, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    5392, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    99952, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite
    4120, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite
    9320, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    5468, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    281284, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite
    5764, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite
    7904, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    5392, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    233784, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite
    4400, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite
    9320, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    5468, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    222312, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite
    5524, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite
    7904, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    5392, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    99952, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite
    4120, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite
    9320, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    5468, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    281284, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite
    5764, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite
    7904, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    5392, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    233784, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite
    4400, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite
    9320, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    5468, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    222312, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite
    5524, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite
    7904, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    5392, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    99952, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite
    4120, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite
    9320, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    5468, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    281284, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite
    5764, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite
    7904, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    5392, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    233784, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite
    4400, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite
    9320, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    5468, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    222312, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite
    5524, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite
    7904, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    5392, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    99952, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite
    4120, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite
    9320, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    5468, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    281284, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite
    5764, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite
    7904, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    5392, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    233816, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite
    4400, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite
    9320, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    5468, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    222312, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite
    5524, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite
    7904, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    5392, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    99952, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite
    4120, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite
    9320, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    5468, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    281284, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite
    5764, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite
    7904, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    5392, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    235608, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite
    4400, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite
    9320, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    5468, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    222312, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    5524, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    7904, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    5392, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    99952, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    4120, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
    9320, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    5468, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    281284, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    5764, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    7904, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    5392, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    233784, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    4400, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
};
