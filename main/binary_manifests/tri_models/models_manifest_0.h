#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Model-Samples-3-Split-No-ELU-conv1d-2d-fix/batch_0
// Blob: models_0.bin (9500872 bytes, align=16)
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
    21312, // 00_w0050_s1_c_second_float32_second_emg_x1_8_50_p50_50_1.tflite (230580 bytes)
    251904, // 00_w0050_s1_d_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite (9528 bytes)
    261440, // 00_w0050_s1_e_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite (5392 bytes)
    266832, // 00_w0050_s1_f_second_int8_second_emg_x1_8_50_p50_50_1.tflite (105368 bytes)
    372208, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite (15840 bytes)
    388048, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite (5468 bytes)
    393520, // 01_w0050_s5_c_second_float32_second_emg_x1_8_50_p50_50_5.tflite (293424 bytes)
    686944, // 01_w0050_s5_d_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite (9528 bytes)
    696480, // 01_w0050_s5_e_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite (5392 bytes)
    701872, // 01_w0050_s5_f_second_int8_second_emg_x1_8_50_p50_50_5.tflite (248808 bytes)
    950688, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite (15840 bytes)
    966528, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite (5468 bytes)
    972000, // 02_w0100_s1_c_second_float32_second_emg_x1_8_100_p100_100_1.tflite (230580 bytes)
    1202592, // 02_w0100_s1_d_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite (9528 bytes)
    1212128, // 02_w0100_s1_e_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite (5392 bytes)
    1217520, // 02_w0100_s1_f_second_int8_second_emg_x1_8_100_p100_100_1.tflite (105368 bytes)
    1322896, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite (15840 bytes)
    1338736, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite (5468 bytes)
    1344208, // 03_w0100_s5_c_second_float32_second_emg_x1_8_100_p100_100_5.tflite (293424 bytes)
    1637632, // 03_w0100_s5_d_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite (9528 bytes)
    1647168, // 03_w0100_s5_e_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite (5392 bytes)
    1652560, // 03_w0100_s5_f_second_int8_second_emg_x1_8_100_p100_100_5.tflite (248552 bytes)
    1901120, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite (15840 bytes)
    1916960, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite (5468 bytes)
    1922432, // 04_w0150_s1_c_second_float32_second_emg_x1_8_150_p150_150_1.tflite (230580 bytes)
    2153024, // 04_w0150_s1_d_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite (9528 bytes)
    2162560, // 04_w0150_s1_e_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite (5392 bytes)
    2167952, // 04_w0150_s1_f_second_int8_second_emg_x1_8_150_p150_150_1.tflite (105368 bytes)
    2273328, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite (15840 bytes)
    2289168, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite (5468 bytes)
    2294640, // 05_w0150_s5_c_second_float32_second_emg_x1_8_150_p150_150_5.tflite (293424 bytes)
    2588064, // 05_w0150_s5_d_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite (9528 bytes)
    2597600, // 05_w0150_s5_e_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite (5392 bytes)
    2602992, // 05_w0150_s5_f_second_int8_second_emg_x1_8_150_p150_150_5.tflite (248776 bytes)
    2851776, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite (15840 bytes)
    2867616, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite (5468 bytes)
    2873088, // 06_w0200_s1_c_second_float32_second_emg_x1_8_200_p200_200_1.tflite (230580 bytes)
    3103680, // 06_w0200_s1_d_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite (9528 bytes)
    3113216, // 06_w0200_s1_e_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite (5392 bytes)
    3118608, // 06_w0200_s1_f_second_int8_second_emg_x1_8_200_p200_200_1.tflite (105368 bytes)
    3223984, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite (15840 bytes)
    3239824, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite (5468 bytes)
    3245296, // 07_w0200_s5_c_second_float32_second_emg_x1_8_200_p200_200_5.tflite (293424 bytes)
    3538720, // 07_w0200_s5_d_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite (9528 bytes)
    3548256, // 07_w0200_s5_e_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite (5392 bytes)
    3553648, // 07_w0200_s5_f_second_int8_second_emg_x1_8_200_p200_200_5.tflite (246952 bytes)
    3800608, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite (15840 bytes)
    3816448, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite (5468 bytes)
    3821920, // 08_w0250_s1_c_second_float32_second_emg_x1_8_250_p250_250_1.tflite (230580 bytes)
    4052512, // 08_w0250_s1_d_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite (9528 bytes)
    4062048, // 08_w0250_s1_e_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite (5392 bytes)
    4067440, // 08_w0250_s1_f_second_int8_second_emg_x1_8_250_p250_250_1.tflite (105368 bytes)
    4172816, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite (15840 bytes)
    4188656, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite (5468 bytes)
    4194128, // 09_w0250_s5_c_second_float32_second_emg_x1_8_250_p250_250_5.tflite (293424 bytes)
    4487552, // 09_w0250_s5_d_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite (9528 bytes)
    4497088, // 09_w0250_s5_e_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite (5392 bytes)
    4502480, // 09_w0250_s5_f_second_int8_second_emg_x1_8_250_p250_250_5.tflite (248760 bytes)
    4751248, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite (15840 bytes)
    4767088, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite (5468 bytes)
    4772560, // 10_w0300_s1_c_second_float32_second_emg_x1_8_300_p300_300_1.tflite (230580 bytes)
    5003152, // 10_w0300_s1_d_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite (9528 bytes)
    5012688, // 10_w0300_s1_e_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite (5392 bytes)
    5018080, // 10_w0300_s1_f_second_int8_second_emg_x1_8_300_p300_300_1.tflite (105368 bytes)
    5123456, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite (15840 bytes)
    5139296, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite (5468 bytes)
    5144768, // 11_w0300_s5_c_second_float32_second_emg_x1_8_300_p300_300_5.tflite (293424 bytes)
    5438192, // 11_w0300_s5_d_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite (9528 bytes)
    5447728, // 11_w0300_s5_e_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite (5392 bytes)
    5453120, // 11_w0300_s5_f_second_int8_second_emg_x1_8_300_p300_300_5.tflite (246968 bytes)
    5700096, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite (15840 bytes)
    5715936, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite (5468 bytes)
    5721408, // 12_w0350_s1_c_second_float32_second_emg_x1_8_350_p350_350_1.tflite (230580 bytes)
    5952000, // 12_w0350_s1_d_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite (9528 bytes)
    5961536, // 12_w0350_s1_e_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite (5392 bytes)
    5966928, // 12_w0350_s1_f_second_int8_second_emg_x1_8_350_p350_350_1.tflite (105368 bytes)
    6072304, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite (15840 bytes)
    6088144, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite (5468 bytes)
    6093616, // 13_w0350_s5_c_second_float32_second_emg_x1_8_350_p350_350_5.tflite (293424 bytes)
    6387040, // 13_w0350_s5_d_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite (9528 bytes)
    6396576, // 13_w0350_s5_e_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite (5392 bytes)
    6401968, // 13_w0350_s5_f_second_int8_second_emg_x1_8_350_p350_350_5.tflite (248760 bytes)
    6650736, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite (15840 bytes)
    6666576, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite (5468 bytes)
    6672048, // 14_w0400_s1_c_second_float32_second_emg_x1_8_400_p400_400_1.tflite (230580 bytes)
    6902640, // 14_w0400_s1_d_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite (9528 bytes)
    6912176, // 14_w0400_s1_e_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite (5392 bytes)
    6917568, // 14_w0400_s1_f_second_int8_second_emg_x1_8_400_p400_400_1.tflite (105368 bytes)
    7022944, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite (15840 bytes)
    7038784, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite (5468 bytes)
    7044256, // 15_w0400_s5_c_second_float32_second_emg_x1_8_400_p400_400_5.tflite (293424 bytes)
    7337680, // 15_w0400_s5_d_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite (9528 bytes)
    7347216, // 15_w0400_s5_e_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite (5392 bytes)
    7352608, // 15_w0400_s5_f_second_int8_second_emg_x1_8_400_p400_400_5.tflite (246968 bytes)
    7599584, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite (15840 bytes)
    7615424, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite (5468 bytes)
    7620896, // 16_w0450_s1_c_second_float32_second_emg_x1_8_450_p450_450_1.tflite (230580 bytes)
    7851488, // 16_w0450_s1_d_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite (9528 bytes)
    7861024, // 16_w0450_s1_e_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite (5392 bytes)
    7866416, // 16_w0450_s1_f_second_int8_second_emg_x1_8_450_p450_450_1.tflite (105368 bytes)
    7971792, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite (15840 bytes)
    7987632, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite (5468 bytes)
    7993104, // 17_w0450_s5_c_second_float32_second_emg_x1_8_450_p450_450_5.tflite (293424 bytes)
    8286528, // 17_w0450_s5_d_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite (9528 bytes)
    8296064, // 17_w0450_s5_e_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite (5392 bytes)
    8301456, // 17_w0450_s5_f_second_int8_second_emg_x1_8_450_p450_450_5.tflite (248760 bytes)
    8550224, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite (15840 bytes)
    8566064, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite (5468 bytes)
    8571536, // 18_w0500_s1_c_second_float32_second_emg_x1_8_500_p500_500_1.tflite (230580 bytes)
    8802128, // 18_w0500_s1_d_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite (9528 bytes)
    8811664, // 18_w0500_s1_e_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite (5392 bytes)
    8817056, // 18_w0500_s1_f_second_int8_second_emg_x1_8_500_p500_500_1.tflite (105368 bytes)
    8922432, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite (15840 bytes)
    8938272, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite (5468 bytes)
    8943744, // 19_w0500_s5_c_second_float32_second_emg_x1_8_500_p500_500_5.tflite (293424 bytes)
    9237168, // 19_w0500_s5_d_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite (9528 bytes)
    9246704, // 19_w0500_s5_e_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite (5392 bytes)
    9252096, // 19_w0500_s5_f_second_int8_second_emg_x1_8_500_p500_500_5.tflite (248776 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    15840, // 00_w0050_s1_a_firstA_float32_firstA_emg_x1_8_50_p50_50_1.tflite
    5468, // 00_w0050_s1_b_firstB_float32_firstB_emg_x1_8_50_p50_50_1.tflite
    230580, // 00_w0050_s1_c_second_float32_second_emg_x1_8_50_p50_50_1.tflite
    9528, // 00_w0050_s1_d_first_int8_firstA_emg_x1_8_50_p50_50_1.tflite
    5392, // 00_w0050_s1_e_first_int8_firstB_emg_x1_8_50_p50_50_1.tflite
    105368, // 00_w0050_s1_f_second_int8_second_emg_x1_8_50_p50_50_1.tflite
    15840, // 01_w0050_s5_a_firstA_float32_firstA_emg_x1_8_50_p50_50_5.tflite
    5468, // 01_w0050_s5_b_firstB_float32_firstB_emg_x1_8_50_p50_50_5.tflite
    293424, // 01_w0050_s5_c_second_float32_second_emg_x1_8_50_p50_50_5.tflite
    9528, // 01_w0050_s5_d_first_int8_firstA_emg_x1_8_50_p50_50_5.tflite
    5392, // 01_w0050_s5_e_first_int8_firstB_emg_x1_8_50_p50_50_5.tflite
    248808, // 01_w0050_s5_f_second_int8_second_emg_x1_8_50_p50_50_5.tflite
    15840, // 02_w0100_s1_a_firstA_float32_firstA_emg_x1_8_100_p100_100_1.tflite
    5468, // 02_w0100_s1_b_firstB_float32_firstB_emg_x1_8_100_p100_100_1.tflite
    230580, // 02_w0100_s1_c_second_float32_second_emg_x1_8_100_p100_100_1.tflite
    9528, // 02_w0100_s1_d_first_int8_firstA_emg_x1_8_100_p100_100_1.tflite
    5392, // 02_w0100_s1_e_first_int8_firstB_emg_x1_8_100_p100_100_1.tflite
    105368, // 02_w0100_s1_f_second_int8_second_emg_x1_8_100_p100_100_1.tflite
    15840, // 03_w0100_s5_a_firstA_float32_firstA_emg_x1_8_100_p100_100_5.tflite
    5468, // 03_w0100_s5_b_firstB_float32_firstB_emg_x1_8_100_p100_100_5.tflite
    293424, // 03_w0100_s5_c_second_float32_second_emg_x1_8_100_p100_100_5.tflite
    9528, // 03_w0100_s5_d_first_int8_firstA_emg_x1_8_100_p100_100_5.tflite
    5392, // 03_w0100_s5_e_first_int8_firstB_emg_x1_8_100_p100_100_5.tflite
    248552, // 03_w0100_s5_f_second_int8_second_emg_x1_8_100_p100_100_5.tflite
    15840, // 04_w0150_s1_a_firstA_float32_firstA_emg_x1_8_150_p150_150_1.tflite
    5468, // 04_w0150_s1_b_firstB_float32_firstB_emg_x1_8_150_p150_150_1.tflite
    230580, // 04_w0150_s1_c_second_float32_second_emg_x1_8_150_p150_150_1.tflite
    9528, // 04_w0150_s1_d_first_int8_firstA_emg_x1_8_150_p150_150_1.tflite
    5392, // 04_w0150_s1_e_first_int8_firstB_emg_x1_8_150_p150_150_1.tflite
    105368, // 04_w0150_s1_f_second_int8_second_emg_x1_8_150_p150_150_1.tflite
    15840, // 05_w0150_s5_a_firstA_float32_firstA_emg_x1_8_150_p150_150_5.tflite
    5468, // 05_w0150_s5_b_firstB_float32_firstB_emg_x1_8_150_p150_150_5.tflite
    293424, // 05_w0150_s5_c_second_float32_second_emg_x1_8_150_p150_150_5.tflite
    9528, // 05_w0150_s5_d_first_int8_firstA_emg_x1_8_150_p150_150_5.tflite
    5392, // 05_w0150_s5_e_first_int8_firstB_emg_x1_8_150_p150_150_5.tflite
    248776, // 05_w0150_s5_f_second_int8_second_emg_x1_8_150_p150_150_5.tflite
    15840, // 06_w0200_s1_a_firstA_float32_firstA_emg_x1_8_200_p200_200_1.tflite
    5468, // 06_w0200_s1_b_firstB_float32_firstB_emg_x1_8_200_p200_200_1.tflite
    230580, // 06_w0200_s1_c_second_float32_second_emg_x1_8_200_p200_200_1.tflite
    9528, // 06_w0200_s1_d_first_int8_firstA_emg_x1_8_200_p200_200_1.tflite
    5392, // 06_w0200_s1_e_first_int8_firstB_emg_x1_8_200_p200_200_1.tflite
    105368, // 06_w0200_s1_f_second_int8_second_emg_x1_8_200_p200_200_1.tflite
    15840, // 07_w0200_s5_a_firstA_float32_firstA_emg_x1_8_200_p200_200_5.tflite
    5468, // 07_w0200_s5_b_firstB_float32_firstB_emg_x1_8_200_p200_200_5.tflite
    293424, // 07_w0200_s5_c_second_float32_second_emg_x1_8_200_p200_200_5.tflite
    9528, // 07_w0200_s5_d_first_int8_firstA_emg_x1_8_200_p200_200_5.tflite
    5392, // 07_w0200_s5_e_first_int8_firstB_emg_x1_8_200_p200_200_5.tflite
    246952, // 07_w0200_s5_f_second_int8_second_emg_x1_8_200_p200_200_5.tflite
    15840, // 08_w0250_s1_a_firstA_float32_firstA_emg_x1_8_250_p250_250_1.tflite
    5468, // 08_w0250_s1_b_firstB_float32_firstB_emg_x1_8_250_p250_250_1.tflite
    230580, // 08_w0250_s1_c_second_float32_second_emg_x1_8_250_p250_250_1.tflite
    9528, // 08_w0250_s1_d_first_int8_firstA_emg_x1_8_250_p250_250_1.tflite
    5392, // 08_w0250_s1_e_first_int8_firstB_emg_x1_8_250_p250_250_1.tflite
    105368, // 08_w0250_s1_f_second_int8_second_emg_x1_8_250_p250_250_1.tflite
    15840, // 09_w0250_s5_a_firstA_float32_firstA_emg_x1_8_250_p250_250_5.tflite
    5468, // 09_w0250_s5_b_firstB_float32_firstB_emg_x1_8_250_p250_250_5.tflite
    293424, // 09_w0250_s5_c_second_float32_second_emg_x1_8_250_p250_250_5.tflite
    9528, // 09_w0250_s5_d_first_int8_firstA_emg_x1_8_250_p250_250_5.tflite
    5392, // 09_w0250_s5_e_first_int8_firstB_emg_x1_8_250_p250_250_5.tflite
    248760, // 09_w0250_s5_f_second_int8_second_emg_x1_8_250_p250_250_5.tflite
    15840, // 10_w0300_s1_a_firstA_float32_firstA_emg_x1_8_300_p300_300_1.tflite
    5468, // 10_w0300_s1_b_firstB_float32_firstB_emg_x1_8_300_p300_300_1.tflite
    230580, // 10_w0300_s1_c_second_float32_second_emg_x1_8_300_p300_300_1.tflite
    9528, // 10_w0300_s1_d_first_int8_firstA_emg_x1_8_300_p300_300_1.tflite
    5392, // 10_w0300_s1_e_first_int8_firstB_emg_x1_8_300_p300_300_1.tflite
    105368, // 10_w0300_s1_f_second_int8_second_emg_x1_8_300_p300_300_1.tflite
    15840, // 11_w0300_s5_a_firstA_float32_firstA_emg_x1_8_300_p300_300_5.tflite
    5468, // 11_w0300_s5_b_firstB_float32_firstB_emg_x1_8_300_p300_300_5.tflite
    293424, // 11_w0300_s5_c_second_float32_second_emg_x1_8_300_p300_300_5.tflite
    9528, // 11_w0300_s5_d_first_int8_firstA_emg_x1_8_300_p300_300_5.tflite
    5392, // 11_w0300_s5_e_first_int8_firstB_emg_x1_8_300_p300_300_5.tflite
    246968, // 11_w0300_s5_f_second_int8_second_emg_x1_8_300_p300_300_5.tflite
    15840, // 12_w0350_s1_a_firstA_float32_firstA_emg_x1_8_350_p350_350_1.tflite
    5468, // 12_w0350_s1_b_firstB_float32_firstB_emg_x1_8_350_p350_350_1.tflite
    230580, // 12_w0350_s1_c_second_float32_second_emg_x1_8_350_p350_350_1.tflite
    9528, // 12_w0350_s1_d_first_int8_firstA_emg_x1_8_350_p350_350_1.tflite
    5392, // 12_w0350_s1_e_first_int8_firstB_emg_x1_8_350_p350_350_1.tflite
    105368, // 12_w0350_s1_f_second_int8_second_emg_x1_8_350_p350_350_1.tflite
    15840, // 13_w0350_s5_a_firstA_float32_firstA_emg_x1_8_350_p350_350_5.tflite
    5468, // 13_w0350_s5_b_firstB_float32_firstB_emg_x1_8_350_p350_350_5.tflite
    293424, // 13_w0350_s5_c_second_float32_second_emg_x1_8_350_p350_350_5.tflite
    9528, // 13_w0350_s5_d_first_int8_firstA_emg_x1_8_350_p350_350_5.tflite
    5392, // 13_w0350_s5_e_first_int8_firstB_emg_x1_8_350_p350_350_5.tflite
    248760, // 13_w0350_s5_f_second_int8_second_emg_x1_8_350_p350_350_5.tflite
    15840, // 14_w0400_s1_a_firstA_float32_firstA_emg_x1_8_400_p400_400_1.tflite
    5468, // 14_w0400_s1_b_firstB_float32_firstB_emg_x1_8_400_p400_400_1.tflite
    230580, // 14_w0400_s1_c_second_float32_second_emg_x1_8_400_p400_400_1.tflite
    9528, // 14_w0400_s1_d_first_int8_firstA_emg_x1_8_400_p400_400_1.tflite
    5392, // 14_w0400_s1_e_first_int8_firstB_emg_x1_8_400_p400_400_1.tflite
    105368, // 14_w0400_s1_f_second_int8_second_emg_x1_8_400_p400_400_1.tflite
    15840, // 15_w0400_s5_a_firstA_float32_firstA_emg_x1_8_400_p400_400_5.tflite
    5468, // 15_w0400_s5_b_firstB_float32_firstB_emg_x1_8_400_p400_400_5.tflite
    293424, // 15_w0400_s5_c_second_float32_second_emg_x1_8_400_p400_400_5.tflite
    9528, // 15_w0400_s5_d_first_int8_firstA_emg_x1_8_400_p400_400_5.tflite
    5392, // 15_w0400_s5_e_first_int8_firstB_emg_x1_8_400_p400_400_5.tflite
    246968, // 15_w0400_s5_f_second_int8_second_emg_x1_8_400_p400_400_5.tflite
    15840, // 16_w0450_s1_a_firstA_float32_firstA_emg_x1_8_450_p450_450_1.tflite
    5468, // 16_w0450_s1_b_firstB_float32_firstB_emg_x1_8_450_p450_450_1.tflite
    230580, // 16_w0450_s1_c_second_float32_second_emg_x1_8_450_p450_450_1.tflite
    9528, // 16_w0450_s1_d_first_int8_firstA_emg_x1_8_450_p450_450_1.tflite
    5392, // 16_w0450_s1_e_first_int8_firstB_emg_x1_8_450_p450_450_1.tflite
    105368, // 16_w0450_s1_f_second_int8_second_emg_x1_8_450_p450_450_1.tflite
    15840, // 17_w0450_s5_a_firstA_float32_firstA_emg_x1_8_450_p450_450_5.tflite
    5468, // 17_w0450_s5_b_firstB_float32_firstB_emg_x1_8_450_p450_450_5.tflite
    293424, // 17_w0450_s5_c_second_float32_second_emg_x1_8_450_p450_450_5.tflite
    9528, // 17_w0450_s5_d_first_int8_firstA_emg_x1_8_450_p450_450_5.tflite
    5392, // 17_w0450_s5_e_first_int8_firstB_emg_x1_8_450_p450_450_5.tflite
    248760, // 17_w0450_s5_f_second_int8_second_emg_x1_8_450_p450_450_5.tflite
    15840, // 18_w0500_s1_a_firstA_float32_firstA_emg_x1_8_500_p500_500_1.tflite
    5468, // 18_w0500_s1_b_firstB_float32_firstB_emg_x1_8_500_p500_500_1.tflite
    230580, // 18_w0500_s1_c_second_float32_second_emg_x1_8_500_p500_500_1.tflite
    9528, // 18_w0500_s1_d_first_int8_firstA_emg_x1_8_500_p500_500_1.tflite
    5392, // 18_w0500_s1_e_first_int8_firstB_emg_x1_8_500_p500_500_1.tflite
    105368, // 18_w0500_s1_f_second_int8_second_emg_x1_8_500_p500_500_1.tflite
    15840, // 19_w0500_s5_a_firstA_float32_firstA_emg_x1_8_500_p500_500_5.tflite
    5468, // 19_w0500_s5_b_firstB_float32_firstB_emg_x1_8_500_p500_500_5.tflite
    293424, // 19_w0500_s5_c_second_float32_second_emg_x1_8_500_p500_500_5.tflite
    9528, // 19_w0500_s5_d_first_int8_firstA_emg_x1_8_500_p500_500_5.tflite
    5392, // 19_w0500_s5_e_first_int8_firstB_emg_x1_8_500_p500_500_5.tflite
    248776, // 19_w0500_s5_f_second_int8_second_emg_x1_8_500_p500_500_5.tflite
};
