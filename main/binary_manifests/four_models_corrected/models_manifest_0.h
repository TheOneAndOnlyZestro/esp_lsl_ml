#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Model-Samples-ESPNN-LSTM-NEW-MODEL/batch_0
// Blob: models_0.bin (9116672 bytes, align=16)
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
    0, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite (8564 bytes)
    8576, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite (5468 bytes)
    14048, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite (222312 bytes)
    236368, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite (5524 bytes)
    241904, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite (7152 bytes)
    249056, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite (5392 bytes)
    254448, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite (99952 bytes)
    354400, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite (4120 bytes)
    358528, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite (8564 bytes)
    367104, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite (5468 bytes)
    372576, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite (281284 bytes)
    653872, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite (5764 bytes)
    659648, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite (7152 bytes)
    666800, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite (5392 bytes)
    672192, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite (234704 bytes)
    906896, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite (4400 bytes)
    911296, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite (8564 bytes)
    919872, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite (5468 bytes)
    925344, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite (222312 bytes)
    1147664, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite (5524 bytes)
    1153200, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite (7152 bytes)
    1160352, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite (5392 bytes)
    1165744, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite (99952 bytes)
    1265696, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite (4120 bytes)
    1269824, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite (8564 bytes)
    1278400, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite (5468 bytes)
    1283872, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite (281284 bytes)
    1565168, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite (5764 bytes)
    1570944, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite (7152 bytes)
    1578096, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite (5392 bytes)
    1583488, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite (233816 bytes)
    1817312, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite (4400 bytes)
    1821712, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite (8564 bytes)
    1830288, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite (5468 bytes)
    1835760, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite (222312 bytes)
    2058080, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite (5524 bytes)
    2063616, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite (7152 bytes)
    2070768, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite (5392 bytes)
    2076160, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite (99952 bytes)
    2176112, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite (4120 bytes)
    2180240, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite (8564 bytes)
    2188816, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite (5468 bytes)
    2194288, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite (281284 bytes)
    2475584, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite (5764 bytes)
    2481360, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite (7152 bytes)
    2488512, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite (5392 bytes)
    2493904, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite (235608 bytes)
    2729520, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite (4400 bytes)
    2733920, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite (8564 bytes)
    2742496, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite (5468 bytes)
    2747968, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite (222312 bytes)
    2970288, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite (5524 bytes)
    2975824, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite (7152 bytes)
    2982976, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite (5392 bytes)
    2988368, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite (99952 bytes)
    3088320, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite (4120 bytes)
    3092448, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite (8564 bytes)
    3101024, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite (5468 bytes)
    3106496, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite (281284 bytes)
    3387792, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite (5764 bytes)
    3393568, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite (7152 bytes)
    3400720, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite (5392 bytes)
    3406112, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite (233800 bytes)
    3639920, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite (4400 bytes)
    3644320, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite (8564 bytes)
    3652896, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite (5468 bytes)
    3658368, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite (222312 bytes)
    3880688, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite (5524 bytes)
    3886224, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite (7152 bytes)
    3893376, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite (5392 bytes)
    3898768, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite (99952 bytes)
    3998720, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite (4120 bytes)
    4002848, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite (8564 bytes)
    4011424, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite (5468 bytes)
    4016896, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite (281284 bytes)
    4298192, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite (5764 bytes)
    4303968, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite (7152 bytes)
    4311120, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite (5392 bytes)
    4316512, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite (235608 bytes)
    4552128, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite (4400 bytes)
    4556528, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite (8564 bytes)
    4565104, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite (5468 bytes)
    4570576, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite (222312 bytes)
    4792896, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite (5524 bytes)
    4798432, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite (7152 bytes)
    4805584, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite (5392 bytes)
    4810976, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite (99952 bytes)
    4910928, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite (4120 bytes)
    4915056, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite (8564 bytes)
    4923632, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite (5468 bytes)
    4929104, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite (281284 bytes)
    5210400, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite (5764 bytes)
    5216176, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite (7152 bytes)
    5223328, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite (5392 bytes)
    5228720, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite (235608 bytes)
    5464336, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite (4400 bytes)
    5468736, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite (8564 bytes)
    5477312, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite (5468 bytes)
    5482784, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite (222312 bytes)
    5705104, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite (5524 bytes)
    5710640, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite (7152 bytes)
    5717792, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite (5392 bytes)
    5723184, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite (99952 bytes)
    5823136, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite (4120 bytes)
    5827264, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite (8564 bytes)
    5835840, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite (5468 bytes)
    5841312, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite (281284 bytes)
    6122608, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite (5764 bytes)
    6128384, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite (7152 bytes)
    6135536, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite (5392 bytes)
    6140928, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite (235608 bytes)
    6376544, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite (4400 bytes)
    6380944, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite (8564 bytes)
    6389520, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite (5468 bytes)
    6394992, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite (222312 bytes)
    6617312, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite (5524 bytes)
    6622848, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite (7152 bytes)
    6630000, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite (5392 bytes)
    6635392, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite (99952 bytes)
    6735344, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite (4120 bytes)
    6739472, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite (8564 bytes)
    6748048, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite (5468 bytes)
    6753520, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite (281284 bytes)
    7034816, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite (5764 bytes)
    7040592, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite (7152 bytes)
    7047744, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite (5392 bytes)
    7053136, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite (234720 bytes)
    7287856, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite (4400 bytes)
    7292256, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite (8564 bytes)
    7300832, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite (5468 bytes)
    7306304, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite (222312 bytes)
    7528624, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite (5524 bytes)
    7534160, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite (7152 bytes)
    7541312, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite (5392 bytes)
    7546704, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite (99952 bytes)
    7646656, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite (4120 bytes)
    7650784, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite (8564 bytes)
    7659360, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite (5468 bytes)
    7664832, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite (281284 bytes)
    7946128, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite (5764 bytes)
    7951904, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite (7152 bytes)
    7959056, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite (5392 bytes)
    7964448, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite (235624 bytes)
    8200080, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite (4400 bytes)
    8204480, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite (8564 bytes)
    8213056, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite (5468 bytes)
    8218528, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite (222312 bytes)
    8440848, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite (5524 bytes)
    8446384, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite (7152 bytes)
    8453536, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite (5392 bytes)
    8458928, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite (99952 bytes)
    8558880, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite (4120 bytes)
    8563008, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite (8564 bytes)
    8571584, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite (5468 bytes)
    8577056, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite (281284 bytes)
    8858352, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite (5764 bytes)
    8864128, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite (7152 bytes)
    8871280, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite (5392 bytes)
    8876672, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite (235592 bytes)
    9112272, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite (4400 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    8564, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    5468, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    222312, // 00_w0050_s1_c_second_float32_secondA_emg_x1_8_50_p50_50_1.tflite
    5524, // 00_w0050_s1_d_second_float32_secondB_emg_x1_8_50_p50_50_1.tflite
    7152, // 00_w0050_s1_e_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    5392, // 00_w0050_s1_f_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    99952, // 00_w0050_s1_g_second_int8_secondA_emg_x1_8_50_p50_50_1.tflite
    4120, // 00_w0050_s1_h_second_int8_secondB_emg_x1_8_50_p50_50_1.tflite
    8564, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    5468, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    281284, // 01_w0050_s5_c_second_float32_secondA_emg_x1_8_50_p50_50_5.tflite
    5764, // 01_w0050_s5_d_second_float32_secondB_emg_x1_8_50_p50_50_5.tflite
    7152, // 01_w0050_s5_e_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    5392, // 01_w0050_s5_f_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    234704, // 01_w0050_s5_g_second_int8_secondA_emg_x1_8_50_p50_50_5.tflite
    4400, // 01_w0050_s5_h_second_int8_secondB_emg_x1_8_50_p50_50_5.tflite
    8564, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    5468, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    222312, // 02_w0100_s1_c_second_float32_secondA_emg_x1_8_100_p100_100_1.tflite
    5524, // 02_w0100_s1_d_second_float32_secondB_emg_x1_8_100_p100_100_1.tflite
    7152, // 02_w0100_s1_e_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    5392, // 02_w0100_s1_f_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    99952, // 02_w0100_s1_g_second_int8_secondA_emg_x1_8_100_p100_100_1.tflite
    4120, // 02_w0100_s1_h_second_int8_secondB_emg_x1_8_100_p100_100_1.tflite
    8564, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    5468, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    281284, // 03_w0100_s5_c_second_float32_secondA_emg_x1_8_100_p100_100_5.tflite
    5764, // 03_w0100_s5_d_second_float32_secondB_emg_x1_8_100_p100_100_5.tflite
    7152, // 03_w0100_s5_e_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    5392, // 03_w0100_s5_f_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    233816, // 03_w0100_s5_g_second_int8_secondA_emg_x1_8_100_p100_100_5.tflite
    4400, // 03_w0100_s5_h_second_int8_secondB_emg_x1_8_100_p100_100_5.tflite
    8564, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    5468, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    222312, // 04_w0150_s1_c_second_float32_secondA_emg_x1_8_150_p150_150_1.tflite
    5524, // 04_w0150_s1_d_second_float32_secondB_emg_x1_8_150_p150_150_1.tflite
    7152, // 04_w0150_s1_e_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    5392, // 04_w0150_s1_f_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    99952, // 04_w0150_s1_g_second_int8_secondA_emg_x1_8_150_p150_150_1.tflite
    4120, // 04_w0150_s1_h_second_int8_secondB_emg_x1_8_150_p150_150_1.tflite
    8564, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    5468, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    281284, // 05_w0150_s5_c_second_float32_secondA_emg_x1_8_150_p150_150_5.tflite
    5764, // 05_w0150_s5_d_second_float32_secondB_emg_x1_8_150_p150_150_5.tflite
    7152, // 05_w0150_s5_e_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    5392, // 05_w0150_s5_f_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    235608, // 05_w0150_s5_g_second_int8_secondA_emg_x1_8_150_p150_150_5.tflite
    4400, // 05_w0150_s5_h_second_int8_secondB_emg_x1_8_150_p150_150_5.tflite
    8564, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    5468, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    222312, // 06_w0200_s1_c_second_float32_secondA_emg_x1_8_200_p200_200_1.tflite
    5524, // 06_w0200_s1_d_second_float32_secondB_emg_x1_8_200_p200_200_1.tflite
    7152, // 06_w0200_s1_e_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    5392, // 06_w0200_s1_f_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    99952, // 06_w0200_s1_g_second_int8_secondA_emg_x1_8_200_p200_200_1.tflite
    4120, // 06_w0200_s1_h_second_int8_secondB_emg_x1_8_200_p200_200_1.tflite
    8564, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    5468, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    281284, // 07_w0200_s5_c_second_float32_secondA_emg_x1_8_200_p200_200_5.tflite
    5764, // 07_w0200_s5_d_second_float32_secondB_emg_x1_8_200_p200_200_5.tflite
    7152, // 07_w0200_s5_e_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    5392, // 07_w0200_s5_f_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    233800, // 07_w0200_s5_g_second_int8_secondA_emg_x1_8_200_p200_200_5.tflite
    4400, // 07_w0200_s5_h_second_int8_secondB_emg_x1_8_200_p200_200_5.tflite
    8564, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    5468, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    222312, // 08_w0250_s1_c_second_float32_secondA_emg_x1_8_250_p250_250_1.tflite
    5524, // 08_w0250_s1_d_second_float32_secondB_emg_x1_8_250_p250_250_1.tflite
    7152, // 08_w0250_s1_e_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    5392, // 08_w0250_s1_f_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    99952, // 08_w0250_s1_g_second_int8_secondA_emg_x1_8_250_p250_250_1.tflite
    4120, // 08_w0250_s1_h_second_int8_secondB_emg_x1_8_250_p250_250_1.tflite
    8564, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    5468, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    281284, // 09_w0250_s5_c_second_float32_secondA_emg_x1_8_250_p250_250_5.tflite
    5764, // 09_w0250_s5_d_second_float32_secondB_emg_x1_8_250_p250_250_5.tflite
    7152, // 09_w0250_s5_e_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    5392, // 09_w0250_s5_f_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    235608, // 09_w0250_s5_g_second_int8_secondA_emg_x1_8_250_p250_250_5.tflite
    4400, // 09_w0250_s5_h_second_int8_secondB_emg_x1_8_250_p250_250_5.tflite
    8564, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    5468, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    222312, // 10_w0300_s1_c_second_float32_secondA_emg_x1_8_300_p300_300_1.tflite
    5524, // 10_w0300_s1_d_second_float32_secondB_emg_x1_8_300_p300_300_1.tflite
    7152, // 10_w0300_s1_e_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    5392, // 10_w0300_s1_f_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    99952, // 10_w0300_s1_g_second_int8_secondA_emg_x1_8_300_p300_300_1.tflite
    4120, // 10_w0300_s1_h_second_int8_secondB_emg_x1_8_300_p300_300_1.tflite
    8564, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    5468, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    281284, // 11_w0300_s5_c_second_float32_secondA_emg_x1_8_300_p300_300_5.tflite
    5764, // 11_w0300_s5_d_second_float32_secondB_emg_x1_8_300_p300_300_5.tflite
    7152, // 11_w0300_s5_e_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    5392, // 11_w0300_s5_f_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    235608, // 11_w0300_s5_g_second_int8_secondA_emg_x1_8_300_p300_300_5.tflite
    4400, // 11_w0300_s5_h_second_int8_secondB_emg_x1_8_300_p300_300_5.tflite
    8564, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    5468, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    222312, // 12_w0350_s1_c_second_float32_secondA_emg_x1_8_350_p350_350_1.tflite
    5524, // 12_w0350_s1_d_second_float32_secondB_emg_x1_8_350_p350_350_1.tflite
    7152, // 12_w0350_s1_e_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    5392, // 12_w0350_s1_f_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    99952, // 12_w0350_s1_g_second_int8_secondA_emg_x1_8_350_p350_350_1.tflite
    4120, // 12_w0350_s1_h_second_int8_secondB_emg_x1_8_350_p350_350_1.tflite
    8564, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    5468, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    281284, // 13_w0350_s5_c_second_float32_secondA_emg_x1_8_350_p350_350_5.tflite
    5764, // 13_w0350_s5_d_second_float32_secondB_emg_x1_8_350_p350_350_5.tflite
    7152, // 13_w0350_s5_e_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    5392, // 13_w0350_s5_f_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    235608, // 13_w0350_s5_g_second_int8_secondA_emg_x1_8_350_p350_350_5.tflite
    4400, // 13_w0350_s5_h_second_int8_secondB_emg_x1_8_350_p350_350_5.tflite
    8564, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    5468, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    222312, // 14_w0400_s1_c_second_float32_secondA_emg_x1_8_400_p400_400_1.tflite
    5524, // 14_w0400_s1_d_second_float32_secondB_emg_x1_8_400_p400_400_1.tflite
    7152, // 14_w0400_s1_e_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    5392, // 14_w0400_s1_f_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    99952, // 14_w0400_s1_g_second_int8_secondA_emg_x1_8_400_p400_400_1.tflite
    4120, // 14_w0400_s1_h_second_int8_secondB_emg_x1_8_400_p400_400_1.tflite
    8564, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    5468, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    281284, // 15_w0400_s5_c_second_float32_secondA_emg_x1_8_400_p400_400_5.tflite
    5764, // 15_w0400_s5_d_second_float32_secondB_emg_x1_8_400_p400_400_5.tflite
    7152, // 15_w0400_s5_e_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    5392, // 15_w0400_s5_f_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    234720, // 15_w0400_s5_g_second_int8_secondA_emg_x1_8_400_p400_400_5.tflite
    4400, // 15_w0400_s5_h_second_int8_secondB_emg_x1_8_400_p400_400_5.tflite
    8564, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    5468, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    222312, // 16_w0450_s1_c_second_float32_secondA_emg_x1_8_450_p450_450_1.tflite
    5524, // 16_w0450_s1_d_second_float32_secondB_emg_x1_8_450_p450_450_1.tflite
    7152, // 16_w0450_s1_e_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    5392, // 16_w0450_s1_f_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    99952, // 16_w0450_s1_g_second_int8_secondA_emg_x1_8_450_p450_450_1.tflite
    4120, // 16_w0450_s1_h_second_int8_secondB_emg_x1_8_450_p450_450_1.tflite
    8564, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    5468, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    281284, // 17_w0450_s5_c_second_float32_secondA_emg_x1_8_450_p450_450_5.tflite
    5764, // 17_w0450_s5_d_second_float32_secondB_emg_x1_8_450_p450_450_5.tflite
    7152, // 17_w0450_s5_e_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    5392, // 17_w0450_s5_f_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    235624, // 17_w0450_s5_g_second_int8_secondA_emg_x1_8_450_p450_450_5.tflite
    4400, // 17_w0450_s5_h_second_int8_secondB_emg_x1_8_450_p450_450_5.tflite
    8564, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    5468, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    222312, // 18_w0500_s1_c_second_float32_secondA_emg_x1_8_500_p500_500_1.tflite
    5524, // 18_w0500_s1_d_second_float32_secondB_emg_x1_8_500_p500_500_1.tflite
    7152, // 18_w0500_s1_e_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    5392, // 18_w0500_s1_f_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    99952, // 18_w0500_s1_g_second_int8_secondA_emg_x1_8_500_p500_500_1.tflite
    4120, // 18_w0500_s1_h_second_int8_secondB_emg_x1_8_500_p500_500_1.tflite
    8564, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    5468, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    281284, // 19_w0500_s5_c_second_float32_secondA_emg_x1_8_500_p500_500_5.tflite
    5764, // 19_w0500_s5_d_second_float32_secondB_emg_x1_8_500_p500_500_5.tflite
    7152, // 19_w0500_s5_e_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    5392, // 19_w0500_s5_f_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    235592, // 19_w0500_s5_g_second_int8_secondA_emg_x1_8_500_p500_500_5.tflite
    4400, // 19_w0500_s5_h_second_int8_secondB_emg_x1_8_500_p500_500_5.tflite
};
