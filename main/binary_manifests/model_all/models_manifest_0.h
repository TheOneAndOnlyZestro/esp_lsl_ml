#pragma once
#include <stdint.h>

// Batch 0 of 1. Source folder: src/tflite_models/Model-Samples/batch_0
// Blob: models_0.bin (9718368 bytes, align=16)
// Layout per config: [first_f32, second_f32, first_int8, second_int8]
//   first  block model index = (cfg * 4) + 0 (f32) / + 2 (int8)
//   second block model index = (cfg * 4) + 1 (f32) / + 3 (int8)

#define MODEL_COUNT 80
#define CONFIG_COUNT 20
#define MODELS_PER_CONFIG 4

// Number of input tensors per model file (1 = first block, 3 = second block)
static const uint32_t INPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0050_s1_a_first_float32_first_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_b_second_float32_second_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_c_first_int8_first_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_d_second_int8_second_emg_x1_8_50_p50_50_1.tflite
    1, // 01_w0050_s5_a_first_float32_first_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_b_second_float32_second_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_c_first_int8_first_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_d_second_int8_second_emg_x1_8_50_p50_50_5.tflite
    1, // 02_w0100_s1_a_first_float32_first_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_b_second_float32_second_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_c_first_int8_first_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_d_second_int8_second_emg_x1_8_100_p100_100_1.tflite
    1, // 03_w0100_s5_a_first_float32_first_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_b_second_float32_second_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_c_first_int8_first_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_d_second_int8_second_emg_x1_8_100_p100_100_5.tflite
    1, // 04_w0150_s1_a_first_float32_first_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_b_second_float32_second_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_c_first_int8_first_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_d_second_int8_second_emg_x1_8_150_p150_150_1.tflite
    1, // 05_w0150_s5_a_first_float32_first_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_b_second_float32_second_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_c_first_int8_first_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_d_second_int8_second_emg_x1_8_150_p150_150_5.tflite
    1, // 06_w0200_s1_a_first_float32_first_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_b_second_float32_second_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_c_first_int8_first_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_d_second_int8_second_emg_x1_8_200_p200_200_1.tflite
    1, // 07_w0200_s5_a_first_float32_first_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_b_second_float32_second_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_c_first_int8_first_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_d_second_int8_second_emg_x1_8_200_p200_200_5.tflite
    1, // 08_w0250_s1_a_first_float32_first_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_b_second_float32_second_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_c_first_int8_first_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_d_second_int8_second_emg_x1_8_250_p250_250_1.tflite
    1, // 09_w0250_s5_a_first_float32_first_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_b_second_float32_second_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_c_first_int8_first_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_d_second_int8_second_emg_x1_8_250_p250_250_5.tflite
    1, // 10_w0300_s1_a_first_float32_first_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_b_second_float32_second_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_c_first_int8_first_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_d_second_int8_second_emg_x1_8_300_p300_300_1.tflite
    1, // 11_w0300_s5_a_first_float32_first_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_b_second_float32_second_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_c_first_int8_first_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_d_second_int8_second_emg_x1_8_300_p300_300_5.tflite
    1, // 12_w0350_s1_a_first_float32_first_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_b_second_float32_second_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_c_first_int8_first_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_d_second_int8_second_emg_x1_8_350_p350_350_1.tflite
    1, // 13_w0350_s5_a_first_float32_first_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_b_second_float32_second_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_c_first_int8_first_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_d_second_int8_second_emg_x1_8_350_p350_350_5.tflite
    1, // 14_w0400_s1_a_first_float32_first_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_b_second_float32_second_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_c_first_int8_first_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_d_second_int8_second_emg_x1_8_400_p400_400_1.tflite
    1, // 15_w0400_s5_a_first_float32_first_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_b_second_float32_second_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_c_first_int8_first_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_d_second_int8_second_emg_x1_8_400_p400_400_5.tflite
    1, // 16_w0450_s1_a_first_float32_first_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_b_second_float32_second_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_c_first_int8_first_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_d_second_int8_second_emg_x1_8_450_p450_450_1.tflite
    1, // 17_w0450_s5_a_first_float32_first_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_b_second_float32_second_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_c_first_int8_first_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_d_second_int8_second_emg_x1_8_450_p450_450_5.tflite
    1, // 18_w0500_s1_a_first_float32_first_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_b_second_float32_second_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_c_first_int8_first_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_d_second_int8_second_emg_x1_8_500_p500_500_1.tflite
    1, // 19_w0500_s5_a_first_float32_first_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_b_second_float32_second_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_c_first_int8_first_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_d_second_int8_second_emg_x1_8_500_p500_500_5.tflite
};

// Number of output tensors per model file (1 = first block, 3 = second block)
static const uint32_t OUTPUT_SIZES[MODEL_COUNT] = {
    1, // 00_w0050_s1_a_first_float32_first_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_b_second_float32_second_emg_x1_8_50_p50_50_1.tflite
    1, // 00_w0050_s1_c_first_int8_first_emg_x1_8_50_p50_50_1.tflite
    3, // 00_w0050_s1_d_second_int8_second_emg_x1_8_50_p50_50_1.tflite
    1, // 01_w0050_s5_a_first_float32_first_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_b_second_float32_second_emg_x1_8_50_p50_50_5.tflite
    1, // 01_w0050_s5_c_first_int8_first_emg_x1_8_50_p50_50_5.tflite
    3, // 01_w0050_s5_d_second_int8_second_emg_x1_8_50_p50_50_5.tflite
    1, // 02_w0100_s1_a_first_float32_first_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_b_second_float32_second_emg_x1_8_100_p100_100_1.tflite
    1, // 02_w0100_s1_c_first_int8_first_emg_x1_8_100_p100_100_1.tflite
    3, // 02_w0100_s1_d_second_int8_second_emg_x1_8_100_p100_100_1.tflite
    1, // 03_w0100_s5_a_first_float32_first_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_b_second_float32_second_emg_x1_8_100_p100_100_5.tflite
    1, // 03_w0100_s5_c_first_int8_first_emg_x1_8_100_p100_100_5.tflite
    3, // 03_w0100_s5_d_second_int8_second_emg_x1_8_100_p100_100_5.tflite
    1, // 04_w0150_s1_a_first_float32_first_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_b_second_float32_second_emg_x1_8_150_p150_150_1.tflite
    1, // 04_w0150_s1_c_first_int8_first_emg_x1_8_150_p150_150_1.tflite
    3, // 04_w0150_s1_d_second_int8_second_emg_x1_8_150_p150_150_1.tflite
    1, // 05_w0150_s5_a_first_float32_first_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_b_second_float32_second_emg_x1_8_150_p150_150_5.tflite
    1, // 05_w0150_s5_c_first_int8_first_emg_x1_8_150_p150_150_5.tflite
    3, // 05_w0150_s5_d_second_int8_second_emg_x1_8_150_p150_150_5.tflite
    1, // 06_w0200_s1_a_first_float32_first_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_b_second_float32_second_emg_x1_8_200_p200_200_1.tflite
    1, // 06_w0200_s1_c_first_int8_first_emg_x1_8_200_p200_200_1.tflite
    3, // 06_w0200_s1_d_second_int8_second_emg_x1_8_200_p200_200_1.tflite
    1, // 07_w0200_s5_a_first_float32_first_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_b_second_float32_second_emg_x1_8_200_p200_200_5.tflite
    1, // 07_w0200_s5_c_first_int8_first_emg_x1_8_200_p200_200_5.tflite
    3, // 07_w0200_s5_d_second_int8_second_emg_x1_8_200_p200_200_5.tflite
    1, // 08_w0250_s1_a_first_float32_first_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_b_second_float32_second_emg_x1_8_250_p250_250_1.tflite
    1, // 08_w0250_s1_c_first_int8_first_emg_x1_8_250_p250_250_1.tflite
    3, // 08_w0250_s1_d_second_int8_second_emg_x1_8_250_p250_250_1.tflite
    1, // 09_w0250_s5_a_first_float32_first_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_b_second_float32_second_emg_x1_8_250_p250_250_5.tflite
    1, // 09_w0250_s5_c_first_int8_first_emg_x1_8_250_p250_250_5.tflite
    3, // 09_w0250_s5_d_second_int8_second_emg_x1_8_250_p250_250_5.tflite
    1, // 10_w0300_s1_a_first_float32_first_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_b_second_float32_second_emg_x1_8_300_p300_300_1.tflite
    1, // 10_w0300_s1_c_first_int8_first_emg_x1_8_300_p300_300_1.tflite
    3, // 10_w0300_s1_d_second_int8_second_emg_x1_8_300_p300_300_1.tflite
    1, // 11_w0300_s5_a_first_float32_first_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_b_second_float32_second_emg_x1_8_300_p300_300_5.tflite
    1, // 11_w0300_s5_c_first_int8_first_emg_x1_8_300_p300_300_5.tflite
    3, // 11_w0300_s5_d_second_int8_second_emg_x1_8_300_p300_300_5.tflite
    1, // 12_w0350_s1_a_first_float32_first_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_b_second_float32_second_emg_x1_8_350_p350_350_1.tflite
    1, // 12_w0350_s1_c_first_int8_first_emg_x1_8_350_p350_350_1.tflite
    3, // 12_w0350_s1_d_second_int8_second_emg_x1_8_350_p350_350_1.tflite
    1, // 13_w0350_s5_a_first_float32_first_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_b_second_float32_second_emg_x1_8_350_p350_350_5.tflite
    1, // 13_w0350_s5_c_first_int8_first_emg_x1_8_350_p350_350_5.tflite
    3, // 13_w0350_s5_d_second_int8_second_emg_x1_8_350_p350_350_5.tflite
    1, // 14_w0400_s1_a_first_float32_first_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_b_second_float32_second_emg_x1_8_400_p400_400_1.tflite
    1, // 14_w0400_s1_c_first_int8_first_emg_x1_8_400_p400_400_1.tflite
    3, // 14_w0400_s1_d_second_int8_second_emg_x1_8_400_p400_400_1.tflite
    1, // 15_w0400_s5_a_first_float32_first_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_b_second_float32_second_emg_x1_8_400_p400_400_5.tflite
    1, // 15_w0400_s5_c_first_int8_first_emg_x1_8_400_p400_400_5.tflite
    3, // 15_w0400_s5_d_second_int8_second_emg_x1_8_400_p400_400_5.tflite
    1, // 16_w0450_s1_a_first_float32_first_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_b_second_float32_second_emg_x1_8_450_p450_450_1.tflite
    1, // 16_w0450_s1_c_first_int8_first_emg_x1_8_450_p450_450_1.tflite
    3, // 16_w0450_s1_d_second_int8_second_emg_x1_8_450_p450_450_1.tflite
    1, // 17_w0450_s5_a_first_float32_first_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_b_second_float32_second_emg_x1_8_450_p450_450_5.tflite
    1, // 17_w0450_s5_c_first_int8_first_emg_x1_8_450_p450_450_5.tflite
    3, // 17_w0450_s5_d_second_int8_second_emg_x1_8_450_p450_450_5.tflite
    1, // 18_w0500_s1_a_first_float32_first_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_b_second_float32_second_emg_x1_8_500_p500_500_1.tflite
    1, // 18_w0500_s1_c_first_int8_first_emg_x1_8_500_p500_500_1.tflite
    3, // 18_w0500_s1_d_second_int8_second_emg_x1_8_500_p500_500_1.tflite
    1, // 19_w0500_s5_a_first_float32_first_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_b_second_float32_second_emg_x1_8_500_p500_500_5.tflite
    1, // 19_w0500_s5_c_first_int8_first_emg_x1_8_500_p500_500_5.tflite
    3, // 19_w0500_s5_d_second_int8_second_emg_x1_8_500_p500_500_5.tflite
};

static const uint32_t MODEL_OFFSETS[MODEL_COUNT] = {
    0, // 00_w0050_s1_a_first_float32_first_emg_x1_8_50_p50_50_1.tflite (26376 bytes)
    26384, // 00_w0050_s1_b_second_float32_second_emg_x1_8_50_p50_50_1.tflite (230748 bytes)
    257136, // 00_w0050_s1_c_first_int8_first_emg_x1_8_50_p50_50_1.tflite (20272 bytes)
    277408, // 00_w0050_s1_d_second_int8_second_emg_x1_8_50_p50_50_1.tflite (105288 bytes)
    382704, // 01_w0050_s5_a_first_float32_first_emg_x1_8_50_p50_50_5.tflite (26376 bytes)
    409088, // 01_w0050_s5_b_second_float32_second_emg_x1_8_50_p50_50_5.tflite (293760 bytes)
    702848, // 01_w0050_s5_c_first_int8_first_emg_x1_8_50_p50_50_5.tflite (20272 bytes)
    723120, // 01_w0050_s5_d_second_int8_second_emg_x1_8_50_p50_50_5.tflite (248576 bytes)
    971696, // 02_w0100_s1_a_first_float32_first_emg_x1_8_100_p100_100_1.tflite (26376 bytes)
    998080, // 02_w0100_s1_b_second_float32_second_emg_x1_8_100_p100_100_1.tflite (230748 bytes)
    1228832, // 02_w0100_s1_c_first_int8_first_emg_x1_8_100_p100_100_1.tflite (20272 bytes)
    1249104, // 02_w0100_s1_d_second_int8_second_emg_x1_8_100_p100_100_1.tflite (105288 bytes)
    1354400, // 03_w0100_s5_a_first_float32_first_emg_x1_8_100_p100_100_5.tflite (26376 bytes)
    1380784, // 03_w0100_s5_b_second_float32_second_emg_x1_8_100_p100_100_5.tflite (293760 bytes)
    1674544, // 03_w0100_s5_c_first_int8_first_emg_x1_8_100_p100_100_5.tflite (20272 bytes)
    1694816, // 03_w0100_s5_d_second_int8_second_emg_x1_8_100_p100_100_5.tflite (248800 bytes)
    1943616, // 04_w0150_s1_a_first_float32_first_emg_x1_8_150_p150_150_1.tflite (26376 bytes)
    1970000, // 04_w0150_s1_b_second_float32_second_emg_x1_8_150_p150_150_1.tflite (230748 bytes)
    2200752, // 04_w0150_s1_c_first_int8_first_emg_x1_8_150_p150_150_1.tflite (20272 bytes)
    2221024, // 04_w0150_s1_d_second_int8_second_emg_x1_8_150_p150_150_1.tflite (105288 bytes)
    2326320, // 05_w0150_s5_a_first_float32_first_emg_x1_8_150_p150_150_5.tflite (26376 bytes)
    2352704, // 05_w0150_s5_b_second_float32_second_emg_x1_8_150_p150_150_5.tflite (293760 bytes)
    2646464, // 05_w0150_s5_c_first_int8_first_emg_x1_8_150_p150_150_5.tflite (20272 bytes)
    2666736, // 05_w0150_s5_d_second_int8_second_emg_x1_8_150_p150_150_5.tflite (248592 bytes)
    2915328, // 06_w0200_s1_a_first_float32_first_emg_x1_8_200_p200_200_1.tflite (26376 bytes)
    2941712, // 06_w0200_s1_b_second_float32_second_emg_x1_8_200_p200_200_1.tflite (230748 bytes)
    3172464, // 06_w0200_s1_c_first_int8_first_emg_x1_8_200_p200_200_1.tflite (20272 bytes)
    3192736, // 06_w0200_s1_d_second_int8_second_emg_x1_8_200_p200_200_1.tflite (105288 bytes)
    3298032, // 07_w0200_s5_a_first_float32_first_emg_x1_8_200_p200_200_5.tflite (26376 bytes)
    3324416, // 07_w0200_s5_b_second_float32_second_emg_x1_8_200_p200_200_5.tflite (293760 bytes)
    3618176, // 07_w0200_s5_c_first_int8_first_emg_x1_8_200_p200_200_5.tflite (20272 bytes)
    3638448, // 07_w0200_s5_d_second_int8_second_emg_x1_8_200_p200_200_5.tflite (248800 bytes)
    3887248, // 08_w0250_s1_a_first_float32_first_emg_x1_8_250_p250_250_1.tflite (26376 bytes)
    3913632, // 08_w0250_s1_b_second_float32_second_emg_x1_8_250_p250_250_1.tflite (230748 bytes)
    4144384, // 08_w0250_s1_c_first_int8_first_emg_x1_8_250_p250_250_1.tflite (20272 bytes)
    4164656, // 08_w0250_s1_d_second_int8_second_emg_x1_8_250_p250_250_1.tflite (105288 bytes)
    4269952, // 09_w0250_s5_a_first_float32_first_emg_x1_8_250_p250_250_5.tflite (26376 bytes)
    4296336, // 09_w0250_s5_b_second_float32_second_emg_x1_8_250_p250_250_5.tflite (293760 bytes)
    4590096, // 09_w0250_s5_c_first_int8_first_emg_x1_8_250_p250_250_5.tflite (20272 bytes)
    4610368, // 09_w0250_s5_d_second_int8_second_emg_x1_8_250_p250_250_5.tflite (248800 bytes)
    4859168, // 10_w0300_s1_a_first_float32_first_emg_x1_8_300_p300_300_1.tflite (26376 bytes)
    4885552, // 10_w0300_s1_b_second_float32_second_emg_x1_8_300_p300_300_1.tflite (230748 bytes)
    5116304, // 10_w0300_s1_c_first_int8_first_emg_x1_8_300_p300_300_1.tflite (20272 bytes)
    5136576, // 10_w0300_s1_d_second_int8_second_emg_x1_8_300_p300_300_1.tflite (105288 bytes)
    5241872, // 11_w0300_s5_a_first_float32_first_emg_x1_8_300_p300_300_5.tflite (26376 bytes)
    5268256, // 11_w0300_s5_b_second_float32_second_emg_x1_8_300_p300_300_5.tflite (293760 bytes)
    5562016, // 11_w0300_s5_c_first_int8_first_emg_x1_8_300_p300_300_5.tflite (20272 bytes)
    5582288, // 11_w0300_s5_d_second_int8_second_emg_x1_8_300_p300_300_5.tflite (248816 bytes)
    5831104, // 12_w0350_s1_a_first_float32_first_emg_x1_8_350_p350_350_1.tflite (26376 bytes)
    5857488, // 12_w0350_s1_b_second_float32_second_emg_x1_8_350_p350_350_1.tflite (230748 bytes)
    6088240, // 12_w0350_s1_c_first_int8_first_emg_x1_8_350_p350_350_1.tflite (20272 bytes)
    6108512, // 12_w0350_s1_d_second_int8_second_emg_x1_8_350_p350_350_1.tflite (105288 bytes)
    6213808, // 13_w0350_s5_a_first_float32_first_emg_x1_8_350_p350_350_5.tflite (26376 bytes)
    6240192, // 13_w0350_s5_b_second_float32_second_emg_x1_8_350_p350_350_5.tflite (293760 bytes)
    6533952, // 13_w0350_s5_c_first_int8_first_emg_x1_8_350_p350_350_5.tflite (20272 bytes)
    6554224, // 13_w0350_s5_d_second_int8_second_emg_x1_8_350_p350_350_5.tflite (248576 bytes)
    6802800, // 14_w0400_s1_a_first_float32_first_emg_x1_8_400_p400_400_1.tflite (26376 bytes)
    6829184, // 14_w0400_s1_b_second_float32_second_emg_x1_8_400_p400_400_1.tflite (230748 bytes)
    7059936, // 14_w0400_s1_c_first_int8_first_emg_x1_8_400_p400_400_1.tflite (20272 bytes)
    7080208, // 14_w0400_s1_d_second_int8_second_emg_x1_8_400_p400_400_1.tflite (105288 bytes)
    7185504, // 15_w0400_s5_a_first_float32_first_emg_x1_8_400_p400_400_5.tflite (26376 bytes)
    7211888, // 15_w0400_s5_b_second_float32_second_emg_x1_8_400_p400_400_5.tflite (293760 bytes)
    7505648, // 15_w0400_s5_c_first_int8_first_emg_x1_8_400_p400_400_5.tflite (20272 bytes)
    7525920, // 15_w0400_s5_d_second_int8_second_emg_x1_8_400_p400_400_5.tflite (248816 bytes)
    7774736, // 16_w0450_s1_a_first_float32_first_emg_x1_8_450_p450_450_1.tflite (26376 bytes)
    7801120, // 16_w0450_s1_b_second_float32_second_emg_x1_8_450_p450_450_1.tflite (230748 bytes)
    8031872, // 16_w0450_s1_c_first_int8_first_emg_x1_8_450_p450_450_1.tflite (20272 bytes)
    8052144, // 16_w0450_s1_d_second_int8_second_emg_x1_8_450_p450_450_1.tflite (105288 bytes)
    8157440, // 17_w0450_s5_a_first_float32_first_emg_x1_8_450_p450_450_5.tflite (26376 bytes)
    8183824, // 17_w0450_s5_b_second_float32_second_emg_x1_8_450_p450_450_5.tflite (293760 bytes)
    8477584, // 17_w0450_s5_c_first_int8_first_emg_x1_8_450_p450_450_5.tflite (20272 bytes)
    8497856, // 17_w0450_s5_d_second_int8_second_emg_x1_8_450_p450_450_5.tflite (248576 bytes)
    8746432, // 18_w0500_s1_a_first_float32_first_emg_x1_8_500_p500_500_1.tflite (26376 bytes)
    8772816, // 18_w0500_s1_b_second_float32_second_emg_x1_8_500_p500_500_1.tflite (230748 bytes)
    9003568, // 18_w0500_s1_c_first_int8_first_emg_x1_8_500_p500_500_1.tflite (20272 bytes)
    9023840, // 18_w0500_s1_d_second_int8_second_emg_x1_8_500_p500_500_1.tflite (105288 bytes)
    9129136, // 19_w0500_s5_a_first_float32_first_emg_x1_8_500_p500_500_5.tflite (26376 bytes)
    9155520, // 19_w0500_s5_b_second_float32_second_emg_x1_8_500_p500_500_5.tflite (293760 bytes)
    9449280, // 19_w0500_s5_c_first_int8_first_emg_x1_8_500_p500_500_5.tflite (20272 bytes)
    9469552, // 19_w0500_s5_d_second_int8_second_emg_x1_8_500_p500_500_5.tflite (248816 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    26376, // 00_w0050_s1_a_first_float32_first_emg_x1_8_50_p50_50_1.tflite
    230748, // 00_w0050_s1_b_second_float32_second_emg_x1_8_50_p50_50_1.tflite
    20272, // 00_w0050_s1_c_first_int8_first_emg_x1_8_50_p50_50_1.tflite
    105288, // 00_w0050_s1_d_second_int8_second_emg_x1_8_50_p50_50_1.tflite
    26376, // 01_w0050_s5_a_first_float32_first_emg_x1_8_50_p50_50_5.tflite
    293760, // 01_w0050_s5_b_second_float32_second_emg_x1_8_50_p50_50_5.tflite
    20272, // 01_w0050_s5_c_first_int8_first_emg_x1_8_50_p50_50_5.tflite
    248576, // 01_w0050_s5_d_second_int8_second_emg_x1_8_50_p50_50_5.tflite
    26376, // 02_w0100_s1_a_first_float32_first_emg_x1_8_100_p100_100_1.tflite
    230748, // 02_w0100_s1_b_second_float32_second_emg_x1_8_100_p100_100_1.tflite
    20272, // 02_w0100_s1_c_first_int8_first_emg_x1_8_100_p100_100_1.tflite
    105288, // 02_w0100_s1_d_second_int8_second_emg_x1_8_100_p100_100_1.tflite
    26376, // 03_w0100_s5_a_first_float32_first_emg_x1_8_100_p100_100_5.tflite
    293760, // 03_w0100_s5_b_second_float32_second_emg_x1_8_100_p100_100_5.tflite
    20272, // 03_w0100_s5_c_first_int8_first_emg_x1_8_100_p100_100_5.tflite
    248800, // 03_w0100_s5_d_second_int8_second_emg_x1_8_100_p100_100_5.tflite
    26376, // 04_w0150_s1_a_first_float32_first_emg_x1_8_150_p150_150_1.tflite
    230748, // 04_w0150_s1_b_second_float32_second_emg_x1_8_150_p150_150_1.tflite
    20272, // 04_w0150_s1_c_first_int8_first_emg_x1_8_150_p150_150_1.tflite
    105288, // 04_w0150_s1_d_second_int8_second_emg_x1_8_150_p150_150_1.tflite
    26376, // 05_w0150_s5_a_first_float32_first_emg_x1_8_150_p150_150_5.tflite
    293760, // 05_w0150_s5_b_second_float32_second_emg_x1_8_150_p150_150_5.tflite
    20272, // 05_w0150_s5_c_first_int8_first_emg_x1_8_150_p150_150_5.tflite
    248592, // 05_w0150_s5_d_second_int8_second_emg_x1_8_150_p150_150_5.tflite
    26376, // 06_w0200_s1_a_first_float32_first_emg_x1_8_200_p200_200_1.tflite
    230748, // 06_w0200_s1_b_second_float32_second_emg_x1_8_200_p200_200_1.tflite
    20272, // 06_w0200_s1_c_first_int8_first_emg_x1_8_200_p200_200_1.tflite
    105288, // 06_w0200_s1_d_second_int8_second_emg_x1_8_200_p200_200_1.tflite
    26376, // 07_w0200_s5_a_first_float32_first_emg_x1_8_200_p200_200_5.tflite
    293760, // 07_w0200_s5_b_second_float32_second_emg_x1_8_200_p200_200_5.tflite
    20272, // 07_w0200_s5_c_first_int8_first_emg_x1_8_200_p200_200_5.tflite
    248800, // 07_w0200_s5_d_second_int8_second_emg_x1_8_200_p200_200_5.tflite
    26376, // 08_w0250_s1_a_first_float32_first_emg_x1_8_250_p250_250_1.tflite
    230748, // 08_w0250_s1_b_second_float32_second_emg_x1_8_250_p250_250_1.tflite
    20272, // 08_w0250_s1_c_first_int8_first_emg_x1_8_250_p250_250_1.tflite
    105288, // 08_w0250_s1_d_second_int8_second_emg_x1_8_250_p250_250_1.tflite
    26376, // 09_w0250_s5_a_first_float32_first_emg_x1_8_250_p250_250_5.tflite
    293760, // 09_w0250_s5_b_second_float32_second_emg_x1_8_250_p250_250_5.tflite
    20272, // 09_w0250_s5_c_first_int8_first_emg_x1_8_250_p250_250_5.tflite
    248800, // 09_w0250_s5_d_second_int8_second_emg_x1_8_250_p250_250_5.tflite
    26376, // 10_w0300_s1_a_first_float32_first_emg_x1_8_300_p300_300_1.tflite
    230748, // 10_w0300_s1_b_second_float32_second_emg_x1_8_300_p300_300_1.tflite
    20272, // 10_w0300_s1_c_first_int8_first_emg_x1_8_300_p300_300_1.tflite
    105288, // 10_w0300_s1_d_second_int8_second_emg_x1_8_300_p300_300_1.tflite
    26376, // 11_w0300_s5_a_first_float32_first_emg_x1_8_300_p300_300_5.tflite
    293760, // 11_w0300_s5_b_second_float32_second_emg_x1_8_300_p300_300_5.tflite
    20272, // 11_w0300_s5_c_first_int8_first_emg_x1_8_300_p300_300_5.tflite
    248816, // 11_w0300_s5_d_second_int8_second_emg_x1_8_300_p300_300_5.tflite
    26376, // 12_w0350_s1_a_first_float32_first_emg_x1_8_350_p350_350_1.tflite
    230748, // 12_w0350_s1_b_second_float32_second_emg_x1_8_350_p350_350_1.tflite
    20272, // 12_w0350_s1_c_first_int8_first_emg_x1_8_350_p350_350_1.tflite
    105288, // 12_w0350_s1_d_second_int8_second_emg_x1_8_350_p350_350_1.tflite
    26376, // 13_w0350_s5_a_first_float32_first_emg_x1_8_350_p350_350_5.tflite
    293760, // 13_w0350_s5_b_second_float32_second_emg_x1_8_350_p350_350_5.tflite
    20272, // 13_w0350_s5_c_first_int8_first_emg_x1_8_350_p350_350_5.tflite
    248576, // 13_w0350_s5_d_second_int8_second_emg_x1_8_350_p350_350_5.tflite
    26376, // 14_w0400_s1_a_first_float32_first_emg_x1_8_400_p400_400_1.tflite
    230748, // 14_w0400_s1_b_second_float32_second_emg_x1_8_400_p400_400_1.tflite
    20272, // 14_w0400_s1_c_first_int8_first_emg_x1_8_400_p400_400_1.tflite
    105288, // 14_w0400_s1_d_second_int8_second_emg_x1_8_400_p400_400_1.tflite
    26376, // 15_w0400_s5_a_first_float32_first_emg_x1_8_400_p400_400_5.tflite
    293760, // 15_w0400_s5_b_second_float32_second_emg_x1_8_400_p400_400_5.tflite
    20272, // 15_w0400_s5_c_first_int8_first_emg_x1_8_400_p400_400_5.tflite
    248816, // 15_w0400_s5_d_second_int8_second_emg_x1_8_400_p400_400_5.tflite
    26376, // 16_w0450_s1_a_first_float32_first_emg_x1_8_450_p450_450_1.tflite
    230748, // 16_w0450_s1_b_second_float32_second_emg_x1_8_450_p450_450_1.tflite
    20272, // 16_w0450_s1_c_first_int8_first_emg_x1_8_450_p450_450_1.tflite
    105288, // 16_w0450_s1_d_second_int8_second_emg_x1_8_450_p450_450_1.tflite
    26376, // 17_w0450_s5_a_first_float32_first_emg_x1_8_450_p450_450_5.tflite
    293760, // 17_w0450_s5_b_second_float32_second_emg_x1_8_450_p450_450_5.tflite
    20272, // 17_w0450_s5_c_first_int8_first_emg_x1_8_450_p450_450_5.tflite
    248576, // 17_w0450_s5_d_second_int8_second_emg_x1_8_450_p450_450_5.tflite
    26376, // 18_w0500_s1_a_first_float32_first_emg_x1_8_500_p500_500_1.tflite
    230748, // 18_w0500_s1_b_second_float32_second_emg_x1_8_500_p500_500_1.tflite
    20272, // 18_w0500_s1_c_first_int8_first_emg_x1_8_500_p500_500_1.tflite
    105288, // 18_w0500_s1_d_second_int8_second_emg_x1_8_500_p500_500_1.tflite
    26376, // 19_w0500_s5_a_first_float32_first_emg_x1_8_500_p500_500_5.tflite
    293760, // 19_w0500_s5_b_second_float32_second_emg_x1_8_500_p500_500_5.tflite
    20272, // 19_w0500_s5_c_first_int8_first_emg_x1_8_500_p500_500_5.tflite
    248816, // 19_w0500_s5_d_second_int8_second_emg_x1_8_500_p500_500_5.tflite
};
