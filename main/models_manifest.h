#pragma once
#include <stdint.h>

#define MODEL_COUNT 40

static const uint32_t MODEL_OFFSETS[MODEL_COUNT] = {
    0, // w0050_float32_0_first_liteRT.tflite (26376 bytes)
    26384, // w0050_float32_1_second_liteRT.tflite (223916 bytes)
    250304, // w0050_int8_0_first_liteRT.tflite (20272 bytes)
    270576, // w0050_int8_1_second_liteRT.tflite (222288 bytes)
    492864, // w0100_float32_0_first_liteRT.tflite (26376 bytes)
    519248, // w0100_float32_1_second_liteRT.tflite (223916 bytes)
    743168, // w0100_int8_0_first_liteRT.tflite (20272 bytes)
    763440, // w0100_int8_1_second_liteRT.tflite (222288 bytes)
    985728, // w0150_float32_0_first_liteRT.tflite (26376 bytes)
    1012112, // w0150_float32_1_second_liteRT.tflite (223916 bytes)
    1236032, // w0150_int8_0_first_liteRT.tflite (20272 bytes)
    1256304, // w0150_int8_1_second_liteRT.tflite (222288 bytes)
    1478592, // w0200_float32_0_first_liteRT.tflite (26376 bytes)
    1504976, // w0200_float32_1_second_liteRT.tflite (223916 bytes)
    1728896, // w0200_int8_0_first_liteRT.tflite (20272 bytes)
    1749168, // w0200_int8_1_second_liteRT.tflite (222288 bytes)
    1971456, // w0250_float32_0_first_liteRT.tflite (26376 bytes)
    1997840, // w0250_float32_1_second_liteRT.tflite (223916 bytes)
    2221760, // w0250_int8_0_first_liteRT.tflite (20272 bytes)
    2242032, // w0250_int8_1_second_liteRT.tflite (222288 bytes)
    2464320, // w0300_float32_0_first_liteRT.tflite (26376 bytes)
    2490704, // w0300_float32_1_second_liteRT.tflite (223916 bytes)
    2714624, // w0300_int8_0_first_liteRT.tflite (20272 bytes)
    2734896, // w0300_int8_1_second_liteRT.tflite (222288 bytes)
    2957184, // w0350_float32_0_first_liteRT.tflite (26376 bytes)
    2983568, // w0350_float32_1_second_liteRT.tflite (223916 bytes)
    3207488, // w0350_int8_0_first_liteRT.tflite (20272 bytes)
    3227760, // w0350_int8_1_second_liteRT.tflite (222288 bytes)
    3450048, // w0400_float32_0_first_liteRT.tflite (26376 bytes)
    3476432, // w0400_float32_1_second_liteRT.tflite (223916 bytes)
    3700352, // w0400_int8_0_first_liteRT.tflite (20272 bytes)
    3720624, // w0400_int8_1_second_liteRT.tflite (222288 bytes)
    3942912, // w0450_float32_0_first_liteRT.tflite (26376 bytes)
    3969296, // w0450_float32_1_second_liteRT.tflite (223916 bytes)
    4193216, // w0450_int8_0_first_liteRT.tflite (20272 bytes)
    4213488, // w0450_int8_1_second_liteRT.tflite (222288 bytes)
    4435776, // w0500_float32_0_first_liteRT.tflite (26376 bytes)
    4462160, // w0500_float32_1_second_liteRT.tflite (223916 bytes)
    4686080, // w0500_int8_0_first_liteRT.tflite (20272 bytes)
    4706352, // w0500_int8_1_second_liteRT.tflite (222288 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    26376, // w0050_float32_0_first_liteRT.tflite
    223916, // w0050_float32_1_second_liteRT.tflite
    20272, // w0050_int8_0_first_liteRT.tflite
    222288, // w0050_int8_1_second_liteRT.tflite
    26376, // w0100_float32_0_first_liteRT.tflite
    223916, // w0100_float32_1_second_liteRT.tflite
    20272, // w0100_int8_0_first_liteRT.tflite
    222288, // w0100_int8_1_second_liteRT.tflite
    26376, // w0150_float32_0_first_liteRT.tflite
    223916, // w0150_float32_1_second_liteRT.tflite
    20272, // w0150_int8_0_first_liteRT.tflite
    222288, // w0150_int8_1_second_liteRT.tflite
    26376, // w0200_float32_0_first_liteRT.tflite
    223916, // w0200_float32_1_second_liteRT.tflite
    20272, // w0200_int8_0_first_liteRT.tflite
    222288, // w0200_int8_1_second_liteRT.tflite
    26376, // w0250_float32_0_first_liteRT.tflite
    223916, // w0250_float32_1_second_liteRT.tflite
    20272, // w0250_int8_0_first_liteRT.tflite
    222288, // w0250_int8_1_second_liteRT.tflite
    26376, // w0300_float32_0_first_liteRT.tflite
    223916, // w0300_float32_1_second_liteRT.tflite
    20272, // w0300_int8_0_first_liteRT.tflite
    222288, // w0300_int8_1_second_liteRT.tflite
    26376, // w0350_float32_0_first_liteRT.tflite
    223916, // w0350_float32_1_second_liteRT.tflite
    20272, // w0350_int8_0_first_liteRT.tflite
    222288, // w0350_int8_1_second_liteRT.tflite
    26376, // w0400_float32_0_first_liteRT.tflite
    223916, // w0400_float32_1_second_liteRT.tflite
    20272, // w0400_int8_0_first_liteRT.tflite
    222288, // w0400_int8_1_second_liteRT.tflite
    26376, // w0450_float32_0_first_liteRT.tflite
    223916, // w0450_float32_1_second_liteRT.tflite
    20272, // w0450_int8_0_first_liteRT.tflite
    222288, // w0450_int8_1_second_liteRT.tflite
    26376, // w0500_float32_0_first_liteRT.tflite
    223916, // w0500_float32_1_second_liteRT.tflite
    20272, // w0500_int8_0_first_liteRT.tflite
    222288, // w0500_int8_1_second_liteRT.tflite
};
