#include "Model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // sqrtf
#include "esp_heap_caps.h"
#include "esp_log.h"
static const char *H = "HEAP";

Model::Model(ModelFlash* model_flash, const unsigned char* model_data, int arena_size) {
    mflash = model_flash;
    this->arena_size = arena_size;

    //uint8_t* tensor_arena_ptr = mflash.initTAPointer();
    tflite::InitializeTarget();
    printf("\n--- Initializing TFLite Model ---\n");

    ESP_LOGW(H, "FREE_HEAP Cont,%u",
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    tflite_model = tflite::GetModel(model_data);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return;
    }

    // 1. Allocate arena on the heap to avoid stack overflow
    //tensor_arena = (uint8_t*)malloc(arena_size);
    tensor_arena = mflash->allocatePointerOnPSRAM(arena_size);
    if (tensor_arena == nullptr) {
        printf("Failed to allocate %d bytes for tensor arena!\n", arena_size);
        return;
    }

    // 2. Register ops
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddStridedSlice();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddPack();
    resolver.AddShape();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddTanh();
    resolver.AddSplit();
    resolver.AddConcatenation();
    resolver.AddLogistic();
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddFill();
    resolver.AddUnpack();
    resolver.AddTranspose();
    resolver.AddGather();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddElu();
    resolver.AddPad();
    resolver.AddPadV2();

    // 3. Build interpreter
    interpreter = new tflite::MicroInterpreter(
        tflite_model, resolver, tensor_arena, arena_size);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed!\n");
        return;
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    printf("Setup complete. Arena used: %d bytes\n", interpreter->arena_used_bytes());
    initialized = true;
}

Model::~Model() {
    if (interpreter != nullptr) {
        delete interpreter;
    }
    if (tensor_arena != nullptr) {
        free(tensor_arena);
    }
}

bool Model::predict(const float* input_data, int input_length,
                    float* results, int output_length) {
    if (!initialized) {
        printf("Cannot predict: model not initialized!\n");
        return false;
    }

    if (input->type == kTfLiteFloat32) {
        for (int i = 0; i < input_length; i++) {
            input->data.f[i] = input_data[i];
        }
    } else if (input->type == kTfLiteInt8) {
        // Quantize: normalized_float -> int8
        // The scale/zero_point here are the TFLite quantization params,
        for (int i = 0; i < input_length; i++) {
            const float quantized = roundf(
                input_data[i] / input->params.scale
            ) + input->params.zero_point;
            // Clamp to int8 range to prevent overflow
            if      (quantized < -128.0f) input->data.int8[i] = -128;
            else if (quantized >  127.0f) input->data.int8[i] =  127;
            else                          input->data.int8[i] = (int8_t)quantized;
        }
    } else {
        printf("Unsupported input tensor type: %d\n", input->type);
        return false;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke() failed!\n");
        return false;
    }

    if (output->type == kTfLiteFloat32) {
        for (int i = 0; i < output_length; i++) {
            results[i] = output->data.f[i];
        }
    } else if (output->type == kTfLiteInt8) {
        // Dequantize: int8 -> float (still in normalized label space)
        for (int i = 0; i < output_length; i++) {
            results[i] = (static_cast<float>(output->data.int8[i])
                          - output->params.zero_point)
                         * output->params.scale;
        }
    } else {
        printf("Unsupported output tensor type: %d\n", output->type);
        return false;
    }
    return true;
}

size_t Model::getArenaUsedBytes() const {
    if (interpreter) return interpreter->arena_used_bytes();
    return 0;
}