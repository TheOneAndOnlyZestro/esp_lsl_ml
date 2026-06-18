#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Model-Samples-Trial/batch_0
// Blob: models_0.bin (9478440 bytes, align=16)
// Layout per config (slot a..f):
//   0 a: firstA float32   3 d: firstA int8
//   1 b: firstB float32   4 e: firstB int8
//   2 c: second float32   5 f: second int8
// Model index = cfg * 6 + slot.
// firstA/firstB: 1 input / 1 output;  second: 3 inputs / 3 outputs.

#define MODEL_COUNT 120
#define CONFIG_COUNT 20
#define MODELS_PER_CONFIG 6

// Number of input tensors per model file (1 = first block, 3 = second block)
static const uint32_t INPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_c_second_float32_second_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_d_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_e_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_f_second_int8_second_emg_x1_8_50_p50_50_1.tflite
    1, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_c_second_float32_second_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_d_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_e_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_f_second_int8_second_emg_x1_8_50_p50_50_5.tflite
    1, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_c_second_float32_second_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_d_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_e_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_f_second_int8_second_emg_x1_8_100_p100_100_1.tflite
    1, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_c_second_float32_second_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_d_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_e_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_f_second_int8_second_emg_x1_8_100_p100_100_5.tflite
    1, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_c_second_float32_second_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_d_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_e_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_f_second_int8_second_emg_x1_8_150_p150_150_1.tflite
    1, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_c_second_float32_second_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_d_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_e_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_f_second_int8_second_emg_x1_8_150_p150_150_5.tflite
    1, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_c_second_float32_second_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_d_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_e_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_f_second_int8_second_emg_x1_8_200_p200_200_1.tflite
    1, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_c_second_float32_second_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_d_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_e_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_f_second_int8_second_emg_x1_8_200_p200_200_5.tflite
    1, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_c_second_float32_second_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_d_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_e_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_f_second_int8_second_emg_x1_8_250_p250_250_1.tflite
    1, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_c_second_float32_second_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_d_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_e_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_f_second_int8_second_emg_x1_8_250_p250_250_5.tflite
    1, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_c_second_float32_second_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_d_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_e_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_f_second_int8_second_emg_x1_8_300_p300_300_1.tflite
    1, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_c_second_float32_second_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_d_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_e_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_f_second_int8_second_emg_x1_8_300_p300_300_5.tflite
    1, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_c_second_float32_second_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_d_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_e_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_f_second_int8_second_emg_x1_8_350_p350_350_1.tflite
    1, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_c_second_float32_second_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_d_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_e_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_f_second_int8_second_emg_x1_8_350_p350_350_5.tflite
    1, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_c_second_float32_second_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_d_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_e_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_f_second_int8_second_emg_x1_8_400_p400_400_1.tflite
    1, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_c_second_float32_second_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_d_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_e_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_f_second_int8_second_emg_x1_8_400_p400_400_5.tflite
    1, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_c_second_float32_second_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_d_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_e_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_f_second_int8_second_emg_x1_8_450_p450_450_1.tflite
    1, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_c_second_float32_second_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_d_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_e_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_f_second_int8_second_emg_x1_8_450_p450_450_5.tflite
    1, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_c_second_float32_second_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_d_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_e_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_f_second_int8_second_emg_x1_8_500_p500_500_1.tflite
    1, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_c_second_float32_second_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_d_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_e_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_f_second_int8_second_emg_x1_8_500_p500_500_5.tflite
};

// Number of output tensors per model file (1 = first block, 3 = second block)
static const uint32_t OUTPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_c_second_float32_second_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_d_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_e_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_f_second_int8_second_emg_x1_8_50_p50_50_1.tflite
    1, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_c_second_float32_second_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_d_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_e_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_f_second_int8_second_emg_x1_8_50_p50_50_5.tflite
    1, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_c_second_float32_second_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_d_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_e_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_f_second_int8_second_emg_x1_8_100_p100_100_1.tflite
    1, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_c_second_float32_second_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_d_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_e_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_f_second_int8_second_emg_x1_8_100_p100_100_5.tflite
    1, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_c_second_float32_second_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_d_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_e_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_f_second_int8_second_emg_x1_8_150_p150_150_1.tflite
    1, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_c_second_float32_second_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_d_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_e_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_f_second_int8_second_emg_x1_8_150_p150_150_5.tflite
    1, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_c_second_float32_second_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_d_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_e_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_f_second_int8_second_emg_x1_8_200_p200_200_1.tflite
    1, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_c_second_float32_second_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_d_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_e_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_f_second_int8_second_emg_x1_8_200_p200_200_5.tflite
    1, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_c_second_float32_second_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_d_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_e_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_f_second_int8_second_emg_x1_8_250_p250_250_1.tflite
    1, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_c_second_float32_second_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_d_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_e_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_f_second_int8_second_emg_x1_8_250_p250_250_5.tflite
    1, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_c_second_float32_second_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_d_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_e_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_f_second_int8_second_emg_x1_8_300_p300_300_1.tflite
    1, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_c_second_float32_second_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_d_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_e_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_f_second_int8_second_emg_x1_8_300_p300_300_5.tflite
    1, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_c_second_float32_second_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_d_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_e_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_f_second_int8_second_emg_x1_8_350_p350_350_1.tflite
    1, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_c_second_float32_second_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_d_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_e_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_f_second_int8_second_emg_x1_8_350_p350_350_5.tflite
    1, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_c_second_float32_second_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_d_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_e_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_f_second_int8_second_emg_x1_8_400_p400_400_1.tflite
    1, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_c_second_float32_second_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_d_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_e_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_f_second_int8_second_emg_x1_8_400_p400_400_5.tflite
    1, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_c_second_float32_second_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_d_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_e_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_f_second_int8_second_emg_x1_8_450_p450_450_1.tflite
    1, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_c_second_float32_second_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_d_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_e_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_f_second_int8_second_emg_x1_8_450_p450_450_5.tflite
    1, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_c_second_float32_second_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_d_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_e_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_f_second_int8_second_emg_x1_8_500_p500_500_1.tflite
    1, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_c_second_float32_second_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_d_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_e_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_f_second_int8_second_emg_x1_8_500_p500_500_5.tflite
};

static const uint32_t MODEL_OFFSETS[MODEL_COUNT] = {
    0, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite (15840 bytes)
    15840, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite (5468 bytes)
    21312, // 00_w0050_s1_c_second_float32_second_emg_x1_8_50_p50_50_1.tflite (230132 bytes)
    251456, // 00_w0050_s1_d_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite (9528 bytes)
    260992, // 00_w0050_s1_e_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite (5392 bytes)
    266384, // 00_w0050_s1_f_second_int8_second_emg_x1_8_50_p50_50_1.tflite (105248 bytes)
    371632, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite (15840 bytes)
    387472, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite (5468 bytes)
    392944, // 01_w0050_s5_c_second_float32_second_emg_x1_8_50_p50_50_5.tflite (291472 bytes)
    684416, // 01_w0050_s5_d_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite (9528 bytes)
    693952, // 01_w0050_s5_e_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite (5392 bytes)
    699344, // 01_w0050_s5_f_second_int8_second_emg_x1_8_50_p50_50_5.tflite (247352 bytes)
    946704, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite (15840 bytes)
    962544, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite (5468 bytes)
    968016, // 02_w0100_s1_c_second_float32_second_emg_x1_8_100_p100_100_1.tflite (230132 bytes)
    1198160, // 02_w0100_s1_d_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite (9528 bytes)
    1207696, // 02_w0100_s1_e_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite (5392 bytes)
    1213088, // 02_w0100_s1_f_second_int8_second_emg_x1_8_100_p100_100_1.tflite (105248 bytes)
    1318336, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite (15840 bytes)
    1334176, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite (5468 bytes)
    1339648, // 03_w0100_s5_c_second_float32_second_emg_x1_8_100_p100_100_5.tflite (291472 bytes)
    1631120, // 03_w0100_s5_d_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite (9528 bytes)
    1640656, // 03_w0100_s5_e_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite (5392 bytes)
    1646048, // 03_w0100_s5_f_second_int8_second_emg_x1_8_100_p100_100_5.tflite (248256 bytes)
    1894304, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite (15840 bytes)
    1910144, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite (5468 bytes)
    1915616, // 04_w0150_s1_c_second_float32_second_emg_x1_8_150_p150_150_1.tflite (230132 bytes)
    2145760, // 04_w0150_s1_d_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite (9528 bytes)
    2155296, // 04_w0150_s1_e_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite (5392 bytes)
    2160688, // 04_w0150_s1_f_second_int8_second_emg_x1_8_150_p150_150_1.tflite (105248 bytes)
    2265936, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite (15840 bytes)
    2281776, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite (5468 bytes)
    2287248, // 05_w0150_s5_c_second_float32_second_emg_x1_8_150_p150_150_5.tflite (291472 bytes)
    2578720, // 05_w0150_s5_d_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite (9528 bytes)
    2588256, // 05_w0150_s5_e_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite (5392 bytes)
    2593648, // 05_w0150_s5_f_second_int8_second_emg_x1_8_150_p150_150_5.tflite (248912 bytes)
    2842560, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite (15840 bytes)
    2858400, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite (5468 bytes)
    2863872, // 06_w0200_s1_c_second_float32_second_emg_x1_8_200_p200_200_1.tflite (230132 bytes)
    3094016, // 06_w0200_s1_d_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite (9528 bytes)
    3103552, // 06_w0200_s1_e_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite (5392 bytes)
    3108944, // 06_w0200_s1_f_second_int8_second_emg_x1_8_200_p200_200_1.tflite (105248 bytes)
    3214192, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite (15840 bytes)
    3230032, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite (5468 bytes)
    3235504, // 07_w0200_s5_c_second_float32_second_emg_x1_8_200_p200_200_5.tflite (291472 bytes)
    3526976, // 07_w0200_s5_d_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite (9528 bytes)
    3536512, // 07_w0200_s5_e_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite (5392 bytes)
    3541904, // 07_w0200_s5_f_second_int8_second_emg_x1_8_200_p200_200_5.tflite (248256 bytes)
    3790160, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite (15840 bytes)
    3806000, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite (5468 bytes)
    3811472, // 08_w0250_s1_c_second_float32_second_emg_x1_8_250_p250_250_1.tflite (230132 bytes)
    4041616, // 08_w0250_s1_d_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite (9528 bytes)
    4051152, // 08_w0250_s1_e_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite (5392 bytes)
    4056544, // 08_w0250_s1_f_second_int8_second_emg_x1_8_250_p250_250_1.tflite (105248 bytes)
    4161792, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite (15840 bytes)
    4177632, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite (5468 bytes)
    4183104, // 09_w0250_s5_c_second_float32_second_emg_x1_8_250_p250_250_5.tflite (291472 bytes)
    4474576, // 09_w0250_s5_d_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite (9528 bytes)
    4484112, // 09_w0250_s5_e_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite (5392 bytes)
    4489504, // 09_w0250_s5_f_second_int8_second_emg_x1_8_250_p250_250_5.tflite (248256 bytes)
    4737760, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite (15840 bytes)
    4753600, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite (5468 bytes)
    4759072, // 10_w0300_s1_c_second_float32_second_emg_x1_8_300_p300_300_1.tflite (230132 bytes)
    4989216, // 10_w0300_s1_d_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite (9528 bytes)
    4998752, // 10_w0300_s1_e_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite (5392 bytes)
    5004144, // 10_w0300_s1_f_second_int8_second_emg_x1_8_300_p300_300_1.tflite (105248 bytes)
    5109392, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite (15840 bytes)
    5125232, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite (5468 bytes)
    5130704, // 11_w0300_s5_c_second_float32_second_emg_x1_8_300_p300_300_5.tflite (291472 bytes)
    5422176, // 11_w0300_s5_d_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite (9528 bytes)
    5431712, // 11_w0300_s5_e_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite (5392 bytes)
    5437104, // 11_w0300_s5_f_second_int8_second_emg_x1_8_300_p300_300_5.tflite (248256 bytes)
    5685360, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite (15840 bytes)
    5701200, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite (5468 bytes)
    5706672, // 12_w0350_s1_c_second_float32_second_emg_x1_8_350_p350_350_1.tflite (230132 bytes)
    5936816, // 12_w0350_s1_d_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite (9528 bytes)
    5946352, // 12_w0350_s1_e_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite (5392 bytes)
    5951744, // 12_w0350_s1_f_second_int8_second_emg_x1_8_350_p350_350_1.tflite (105248 bytes)
    6056992, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite (15840 bytes)
    6072832, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite (5468 bytes)
    6078304, // 13_w0350_s5_c_second_float32_second_emg_x1_8_350_p350_350_5.tflite (291472 bytes)
    6369776, // 13_w0350_s5_d_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite (9528 bytes)
    6379312, // 13_w0350_s5_e_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite (5392 bytes)
    6384704, // 13_w0350_s5_f_second_int8_second_emg_x1_8_350_p350_350_5.tflite (249160 bytes)
    6633872, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite (15840 bytes)
    6649712, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite (5468 bytes)
    6655184, // 14_w0400_s1_c_second_float32_second_emg_x1_8_400_p400_400_1.tflite (230132 bytes)
    6885328, // 14_w0400_s1_d_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite (9528 bytes)
    6894864, // 14_w0400_s1_e_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite (5392 bytes)
    6900256, // 14_w0400_s1_f_second_int8_second_emg_x1_8_400_p400_400_1.tflite (105248 bytes)
    7005504, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite (15840 bytes)
    7021344, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite (5468 bytes)
    7026816, // 15_w0400_s5_c_second_float32_second_emg_x1_8_400_p400_400_5.tflite (291472 bytes)
    7318288, // 15_w0400_s5_d_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite (9528 bytes)
    7327824, // 15_w0400_s5_e_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite (5392 bytes)
    7333216, // 15_w0400_s5_f_second_int8_second_emg_x1_8_400_p400_400_5.tflite (249144 bytes)
    7582368, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite (15840 bytes)
    7598208, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite (5468 bytes)
    7603680, // 16_w0450_s1_c_second_float32_second_emg_x1_8_450_p450_450_1.tflite (230132 bytes)
    7833824, // 16_w0450_s1_d_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite (9528 bytes)
    7843360, // 16_w0450_s1_e_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite (5392 bytes)
    7848752, // 16_w0450_s1_f_second_int8_second_emg_x1_8_450_p450_450_1.tflite (105248 bytes)
    7954000, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite (15840 bytes)
    7969840, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite (5468 bytes)
    7975312, // 17_w0450_s5_c_second_float32_second_emg_x1_8_450_p450_450_5.tflite (291472 bytes)
    8266784, // 17_w0450_s5_d_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite (9528 bytes)
    8276320, // 17_w0450_s5_e_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite (5392 bytes)
    8281712, // 17_w0450_s5_f_second_int8_second_emg_x1_8_450_p450_450_5.tflite (248240 bytes)
    8529952, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite (15840 bytes)
    8545792, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite (5468 bytes)
    8551264, // 18_w0500_s1_c_second_float32_second_emg_x1_8_500_p500_500_1.tflite (230132 bytes)
    8781408, // 18_w0500_s1_d_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite (9528 bytes)
    8790944, // 18_w0500_s1_e_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite (5392 bytes)
    8796336, // 18_w0500_s1_f_second_int8_second_emg_x1_8_500_p500_500_1.tflite (105248 bytes)
    8901584, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite (15840 bytes)
    8917424, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite (5468 bytes)
    8922896, // 19_w0500_s5_c_second_float32_second_emg_x1_8_500_p500_500_5.tflite (291472 bytes)
    9214368, // 19_w0500_s5_d_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite (9528 bytes)
    9223904, // 19_w0500_s5_e_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite (5392 bytes)
    9229296, // 19_w0500_s5_f_second_int8_second_emg_x1_8_500_p500_500_5.tflite (249144 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    15840, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    5468, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    230132, // 00_w0050_s1_c_second_float32_second_emg_x1_8_50_p50_50_1.tflite
    9528, // 00_w0050_s1_d_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    5392, // 00_w0050_s1_e_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    105248, // 00_w0050_s1_f_second_int8_second_emg_x1_8_50_p50_50_1.tflite
    15840, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    5468, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    291472, // 01_w0050_s5_c_second_float32_second_emg_x1_8_50_p50_50_5.tflite
    9528, // 01_w0050_s5_d_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    5392, // 01_w0050_s5_e_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    247352, // 01_w0050_s5_f_second_int8_second_emg_x1_8_50_p50_50_5.tflite
    15840, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    5468, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    230132, // 02_w0100_s1_c_second_float32_second_emg_x1_8_100_p100_100_1.tflite
    9528, // 02_w0100_s1_d_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    5392, // 02_w0100_s1_e_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    105248, // 02_w0100_s1_f_second_int8_second_emg_x1_8_100_p100_100_1.tflite
    15840, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    5468, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    291472, // 03_w0100_s5_c_second_float32_second_emg_x1_8_100_p100_100_5.tflite
    9528, // 03_w0100_s5_d_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    5392, // 03_w0100_s5_e_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    248256, // 03_w0100_s5_f_second_int8_second_emg_x1_8_100_p100_100_5.tflite
    15840, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    5468, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    230132, // 04_w0150_s1_c_second_float32_second_emg_x1_8_150_p150_150_1.tflite
    9528, // 04_w0150_s1_d_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    5392, // 04_w0150_s1_e_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    105248, // 04_w0150_s1_f_second_int8_second_emg_x1_8_150_p150_150_1.tflite
    15840, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    5468, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    291472, // 05_w0150_s5_c_second_float32_second_emg_x1_8_150_p150_150_5.tflite
    9528, // 05_w0150_s5_d_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    5392, // 05_w0150_s5_e_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    248912, // 05_w0150_s5_f_second_int8_second_emg_x1_8_150_p150_150_5.tflite
    15840, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    5468, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    230132, // 06_w0200_s1_c_second_float32_second_emg_x1_8_200_p200_200_1.tflite
    9528, // 06_w0200_s1_d_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    5392, // 06_w0200_s1_e_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    105248, // 06_w0200_s1_f_second_int8_second_emg_x1_8_200_p200_200_1.tflite
    15840, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    5468, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    291472, // 07_w0200_s5_c_second_float32_second_emg_x1_8_200_p200_200_5.tflite
    9528, // 07_w0200_s5_d_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    5392, // 07_w0200_s5_e_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    248256, // 07_w0200_s5_f_second_int8_second_emg_x1_8_200_p200_200_5.tflite
    15840, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    5468, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    230132, // 08_w0250_s1_c_second_float32_second_emg_x1_8_250_p250_250_1.tflite
    9528, // 08_w0250_s1_d_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    5392, // 08_w0250_s1_e_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    105248, // 08_w0250_s1_f_second_int8_second_emg_x1_8_250_p250_250_1.tflite
    15840, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    5468, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    291472, // 09_w0250_s5_c_second_float32_second_emg_x1_8_250_p250_250_5.tflite
    9528, // 09_w0250_s5_d_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    5392, // 09_w0250_s5_e_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    248256, // 09_w0250_s5_f_second_int8_second_emg_x1_8_250_p250_250_5.tflite
    15840, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    5468, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    230132, // 10_w0300_s1_c_second_float32_second_emg_x1_8_300_p300_300_1.tflite
    9528, // 10_w0300_s1_d_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    5392, // 10_w0300_s1_e_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    105248, // 10_w0300_s1_f_second_int8_second_emg_x1_8_300_p300_300_1.tflite
    15840, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    5468, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    291472, // 11_w0300_s5_c_second_float32_second_emg_x1_8_300_p300_300_5.tflite
    9528, // 11_w0300_s5_d_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    5392, // 11_w0300_s5_e_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    248256, // 11_w0300_s5_f_second_int8_second_emg_x1_8_300_p300_300_5.tflite
    15840, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    5468, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    230132, // 12_w0350_s1_c_second_float32_second_emg_x1_8_350_p350_350_1.tflite
    9528, // 12_w0350_s1_d_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    5392, // 12_w0350_s1_e_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    105248, // 12_w0350_s1_f_second_int8_second_emg_x1_8_350_p350_350_1.tflite
    15840, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    5468, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    291472, // 13_w0350_s5_c_second_float32_second_emg_x1_8_350_p350_350_5.tflite
    9528, // 13_w0350_s5_d_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    5392, // 13_w0350_s5_e_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    249160, // 13_w0350_s5_f_second_int8_second_emg_x1_8_350_p350_350_5.tflite
    15840, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    5468, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    230132, // 14_w0400_s1_c_second_float32_second_emg_x1_8_400_p400_400_1.tflite
    9528, // 14_w0400_s1_d_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    5392, // 14_w0400_s1_e_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    105248, // 14_w0400_s1_f_second_int8_second_emg_x1_8_400_p400_400_1.tflite
    15840, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    5468, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    291472, // 15_w0400_s5_c_second_float32_second_emg_x1_8_400_p400_400_5.tflite
    9528, // 15_w0400_s5_d_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    5392, // 15_w0400_s5_e_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    249144, // 15_w0400_s5_f_second_int8_second_emg_x1_8_400_p400_400_5.tflite
    15840, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    5468, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    230132, // 16_w0450_s1_c_second_float32_second_emg_x1_8_450_p450_450_1.tflite
    9528, // 16_w0450_s1_d_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    5392, // 16_w0450_s1_e_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    105248, // 16_w0450_s1_f_second_int8_second_emg_x1_8_450_p450_450_1.tflite
    15840, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    5468, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    291472, // 17_w0450_s5_c_second_float32_second_emg_x1_8_450_p450_450_5.tflite
    9528, // 17_w0450_s5_d_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    5392, // 17_w0450_s5_e_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    248240, // 17_w0450_s5_f_second_int8_second_emg_x1_8_450_p450_450_5.tflite
    15840, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    5468, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    230132, // 18_w0500_s1_c_second_float32_second_emg_x1_8_500_p500_500_1.tflite
    9528, // 18_w0500_s1_d_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    5392, // 18_w0500_s1_e_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    105248, // 18_w0500_s1_f_second_int8_second_emg_x1_8_500_p500_500_1.tflite
    15840, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    5468, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    291472, // 19_w0500_s5_c_second_float32_second_emg_x1_8_500_p500_500_5.tflite
    9528, // 19_w0500_s5_d_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    5392, // 19_w0500_s5_e_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    249144, // 19_w0500_s5_f_second_int8_second_emg_x1_8_500_p500_500_5.tflite
};
