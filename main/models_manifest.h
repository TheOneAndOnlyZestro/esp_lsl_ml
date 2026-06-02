#pragma once
#include <stdint.h>

#define MODEL_COUNT 40

static const uint32_t MODEL_OFFSETS[MODEL_COUNT] = {
    0, // w0050_float32_0_first_liteRT.tflite (26376 bytes)
    26384, // w0050_float32_1_second_liteRT.tflite (222492 bytes)
    248880, // w0050_int8_0_first_liteRT.tflite (20272 bytes)
    269152, // w0050_int8_1_second_liteRT.tflite (220400 bytes)
    489552, // w0100_float32_0_first_liteRT.tflite (26376 bytes)
    515936, // w0100_float32_1_second_liteRT.tflite (222492 bytes)
    738432, // w0100_int8_0_first_liteRT.tflite (20272 bytes)
    758704, // w0100_int8_1_second_liteRT.tflite (220400 bytes)
    979104, // w0150_float32_0_first_liteRT.tflite (26376 bytes)
    1005488, // w0150_float32_1_second_liteRT.tflite (222492 bytes)
    1227984, // w0150_int8_0_first_liteRT.tflite (20272 bytes)
    1248256, // w0150_int8_1_second_liteRT.tflite (220400 bytes)
    1468656, // w0200_float32_0_first_liteRT.tflite (26376 bytes)
    1495040, // w0200_float32_1_second_liteRT.tflite (222492 bytes)
    1717536, // w0200_int8_0_first_liteRT.tflite (20272 bytes)
    1737808, // w0200_int8_1_second_liteRT.tflite (220400 bytes)
    1958208, // w0250_float32_0_first_liteRT.tflite (26376 bytes)
    1984592, // w0250_float32_1_second_liteRT.tflite (222492 bytes)
    2207088, // w0250_int8_0_first_liteRT.tflite (20272 bytes)
    2227360, // w0250_int8_1_second_liteRT.tflite (220400 bytes)
    2447760, // w0300_float32_0_first_liteRT.tflite (26376 bytes)
    2474144, // w0300_float32_1_second_liteRT.tflite (222492 bytes)
    2696640, // w0300_int8_0_first_liteRT.tflite (20272 bytes)
    2716912, // w0300_int8_1_second_liteRT.tflite (220400 bytes)
    2937312, // w0350_float32_0_first_liteRT.tflite (26376 bytes)
    2963696, // w0350_float32_1_second_liteRT.tflite (222492 bytes)
    3186192, // w0350_int8_0_first_liteRT.tflite (20272 bytes)
    3206464, // w0350_int8_1_second_liteRT.tflite (220400 bytes)
    3426864, // w0400_float32_0_first_liteRT.tflite (26376 bytes)
    3453248, // w0400_float32_1_second_liteRT.tflite (222492 bytes)
    3675744, // w0400_int8_0_first_liteRT.tflite (20272 bytes)
    3696016, // w0400_int8_1_second_liteRT.tflite (220400 bytes)
    3916416, // w0450_float32_0_first_liteRT.tflite (26376 bytes)
    3942800, // w0450_float32_1_second_liteRT.tflite (222492 bytes)
    4165296, // w0450_int8_0_first_liteRT.tflite (20272 bytes)
    4185568, // w0450_int8_1_second_liteRT.tflite (220400 bytes)
    4405968, // w0500_float32_0_first_liteRT.tflite (26376 bytes)
    4432352, // w0500_float32_1_second_liteRT.tflite (222492 bytes)
    4654848, // w0500_int8_0_first_liteRT.tflite (20272 bytes)
    4675120, // w0500_int8_1_second_liteRT.tflite (220400 bytes)
};

static const uint32_t MODEL_SIZES[MODEL_COUNT] = {
    26376, // w0050_float32_0_first_liteRT.tflite
    222492, // w0050_float32_1_second_liteRT.tflite
    20272, // w0050_int8_0_first_liteRT.tflite
    220400, // w0050_int8_1_second_liteRT.tflite
    26376, // w0100_float32_0_first_liteRT.tflite
    222492, // w0100_float32_1_second_liteRT.tflite
    20272, // w0100_int8_0_first_liteRT.tflite
    220400, // w0100_int8_1_second_liteRT.tflite
    26376, // w0150_float32_0_first_liteRT.tflite
    222492, // w0150_float32_1_second_liteRT.tflite
    20272, // w0150_int8_0_first_liteRT.tflite
    220400, // w0150_int8_1_second_liteRT.tflite
    26376, // w0200_float32_0_first_liteRT.tflite
    222492, // w0200_float32_1_second_liteRT.tflite
    20272, // w0200_int8_0_first_liteRT.tflite
    220400, // w0200_int8_1_second_liteRT.tflite
    26376, // w0250_float32_0_first_liteRT.tflite
    222492, // w0250_float32_1_second_liteRT.tflite
    20272, // w0250_int8_0_first_liteRT.tflite
    220400, // w0250_int8_1_second_liteRT.tflite
    26376, // w0300_float32_0_first_liteRT.tflite
    222492, // w0300_float32_1_second_liteRT.tflite
    20272, // w0300_int8_0_first_liteRT.tflite
    220400, // w0300_int8_1_second_liteRT.tflite
    26376, // w0350_float32_0_first_liteRT.tflite
    222492, // w0350_float32_1_second_liteRT.tflite
    20272, // w0350_int8_0_first_liteRT.tflite
    220400, // w0350_int8_1_second_liteRT.tflite
    26376, // w0400_float32_0_first_liteRT.tflite
    222492, // w0400_float32_1_second_liteRT.tflite
    20272, // w0400_int8_0_first_liteRT.tflite
    220400, // w0400_int8_1_second_liteRT.tflite
    26376, // w0450_float32_0_first_liteRT.tflite
    222492, // w0450_float32_1_second_liteRT.tflite
    20272, // w0450_int8_0_first_liteRT.tflite
    220400, // w0450_int8_1_second_liteRT.tflite
    26376, // w0500_float32_0_first_liteRT.tflite
    222492, // w0500_float32_1_second_liteRT.tflite
    20272, // w0500_int8_0_first_liteRT.tflite
    220400, // w0500_int8_1_second_liteRT.tflite
};
