#include "Model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // sqrtf

Model::Model(const unsigned char* model_data, int arena_size) {
    this->arena_size = arena_size;

    tflite::InitializeTarget();
    printf("\n--- Initializing TFLite Model ---\n");

    tflite_model = tflite::GetModel(model_data);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return;
    }

    // 1. Allocate arena on the heap to avoid stack overflow
    tensor_arena = (uint8_t*)malloc(arena_size);
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

    // ----------------------------------------------------------------
    // STEP 1: Normalize features into a temporary buffer.
    //
    // We MUST NOT modify input_data in-place (caller owns it and may
    // reuse it across multiple predict() calls).
    // We allocate on the stack — input_length is at most seq_len×9
    // which for seq=20 is 180 floats = 720 bytes, safe on ESP32 stack.
    // ----------------------------------------------------------------
    float norm_buffer[input_length];
    for (int i = 0; i < input_length; i++) {
        norm_buffer[i] = input_data[i];
    }
    normalizeFeatures(norm_buffer, input_length);

    // ----------------------------------------------------------------
    // STEP 2: Fill the input tensor with normalized values.
    // ----------------------------------------------------------------
    if (input->type == kTfLiteFloat32) {
        for (int i = 0; i < input_length; i++) {
            input->data.f[i] = norm_buffer[i];
        }
    } else if (input->type == kTfLiteInt8) {
        // Quantize: normalized_float -> int8
        // The scale/zero_point here are the TFLite quantization params,
        // NOT the StandardScaler params — they are different things.
        for (int i = 0; i < input_length; i++) {
            const float quantized = roundf(
                norm_buffer[i] / input->params.scale
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

    // ----------------------------------------------------------------
    // STEP 3: Run inference.
    // ----------------------------------------------------------------
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke() failed!\n");
        return false;
    }

    // ----------------------------------------------------------------
    // STEP 4: Read raw model output into results[].
    // ----------------------------------------------------------------
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

    // ----------------------------------------------------------------
    // STEP 5: Denormalize output back to real-world units.
    //   - Euler angles [0..NUM_EULER-1]: inverse StandardScaler -> degrees
    //   - Quaternions  [NUM_EULER..6]  : re-normalize to unit norm
    // ----------------------------------------------------------------
    denormalizeOutput(results, output_length);

    return true;
}

size_t Model::getArenaUsedBytes() const {
    if (interpreter) return interpreter->arena_used_bytes();
    return 0;
}