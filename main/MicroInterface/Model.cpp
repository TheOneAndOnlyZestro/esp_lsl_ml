#include "Model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // sqrtf
#include "esp_heap_caps.h"
#include "esp_log.h"
static const char *H = "HEAP";

Model::Model(ModelFlash* model_flash, const unsigned char* model_data, int arena_size, size_t input_size, size_t output_size) {
    mflash = model_flash;
    this->arena_size = arena_size;

    this->input_size = input_size;
    this->output_size = output_size;

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
    resolver.AddExp();
    resolver.AddGreaterEqual();
    resolver.AddSelectV2();
    resolver.AddSlice();
    resolver.AddSub();
    resolver.AddSelect();

    printf("GOING TO ALLOCATE INTERPRETER NOW\n");
    // 3. Build interpreter
    interpreter = new tflite::MicroInterpreter(
        tflite_model, resolver, tensor_arena, arena_size);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed!\n");
        return;
    }
    printf("TENSORS are ready\n");

    for(int i = 0; i < this->input_size; i++) {
        input[i] = interpreter->input(i);
    }

    for(int i = 0; i < this->output_size; i++) {
        output[i] = interpreter->output(i);
    }

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

bool Model::predict(const float** input_data, const int* input_lengths,
                float** results, int* output_lengths) {
    if (!initialized) {
        printf("Cannot predict: model not initialized!\n");
        return false;
    }

    for(int i =0; i < input_size; i++) {
        if (input[i]->type == kTfLiteFloat32) {
        for (int j = 0; j < input_lengths[i]; j++) {
            input[i]->data.f[j] = input_data[i][j];
        }
        } else if (input[i]->type == kTfLiteInt8) {
            // Quantize: normalized_float -> int8
            // The scale/zero_point here are the TFLite quantization params,
            for (int j = 0; j < input_lengths[i]; j++) {
                const float quantized = roundf(
                    input_data[i][j] / input[i]->params.scale
                ) + input[i]->params.zero_point;
                // Clamp to int8 range to prevent overflow
                if      (quantized < -128.0f) input[i]->data.int8[j] = -128;
                else if (quantized >  127.0f) input[i]->data.int8[j] =  127;
                else                          input[i]->data.int8[j] = (int8_t)quantized;
            }
        } else {
            printf("Unsupported input tensor type: %d\n", input[i]->type);
            return false;
        }
    }
    


    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke() failed!\n");
        return false;
    }


    for(int i =0; i < output_size; i++) {
        if (output[i]->type == kTfLiteFloat32) {
        for (int j = 0; j < output_lengths[i]; j++) {
            results[i][j] = output[i]->data.f[j];
        }
        } else if (output[i]->type == kTfLiteInt8) {
            // Dequantize: int8 -> float (still in normalized label space)
            for (int j = 0; j < output_lengths[i]; j++) {
                results[i][j] = (static_cast<float>(output[i]->data.int8[j])
                                - output[i]->params.zero_point)
                                * output[i]->params.scale;
            }
        } else {
            printf("Unsupported output tensor type: %d\n", output[i]->type);
            return false;
        }
    }
    
    return true;
}

size_t Model::getArenaUsedBytes() const {
    if (interpreter) return interpreter->arena_used_bytes();
    return 0;
}