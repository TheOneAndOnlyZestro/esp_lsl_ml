#include "Model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // sqrtf
#include "esp_heap_caps.h"
#include "esp_log.h"
static const char *H = "HEAP";

Model::Model(ModelFlash* model_flash, const unsigned char* model_data, int arena_size, 
        size_t input_size, size_t output_size, bool usePSRAM,char* report_buffer, int size)
        :profiler()
{
    mflash = model_flash;
    this->arena_size = arena_size;
    this->inPSRAM = usePSRAM;
    this->input_size = input_size;
    this->output_size = output_size;

    tflite::InitializeTarget();
    
    //initialize profiler

    tflite_model = tflite::GetModel(model_data);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return;
    }

    // 1. Allocate arena on the heap to avoid stack overflow
    if(this->inPSRAM)
        tensor_arena = mflash->allocatePointerOnPSRAM(arena_size);
    else
        tensor_arena = (uint8_t*)malloc(arena_size);
    if (tensor_arena == nullptr) {
        printf("Failed to allocate %d bytes for tensor arena!\n", arena_size);
        return;
    }

    // 2. Register op
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
    resolver.AddRelu();
    resolver.AddBroadcastTo();

    //printf("GOING TO ALLOCATE INTERPRETER NOW\n");
    // 3. Build interpreter
    interpreter = new tflite::MicroInterpreter(
        tflite_model, resolver, tensor_arena, arena_size,
        nullptr, &profiler, false);
    
    
        
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed!\n");
        return;
    }
    
    int report_size = strlen(report_buffer);
    snprintf(report_buffer + report_size, size - report_size, "0_Model Arena_Size: %zuB\n",
    interpreter->arena_used_bytes());
    //printf("TENSORS are ready\n");
    
    input = new TfLiteTensor*[input_size];
    output = new TfLiteTensor*[output_size];

    for(int i = 0; i < this->input_size; i++) {
        printf("Input(%d) Type: %d \n", i, interpreter->input(i)->type);
        printf("Input(%d) Dims: %d \n", i, interpreter->input(i)->dims->size);
        for (int j =0; j < interpreter->input(i)->dims->size; j++)
        {
            printf("Input(%d) Dim (%d): \n", i, interpreter->input(i)->dims->data[j]);
        }
        input[i] = interpreter->input(i);
    }

    for(int i = 0; i < this->output_size; i++) {
        printf("Output(%d) Type: %d \n", i, interpreter->output(i)->type);
        printf("Output(%d) Dims: %d \n", i, interpreter->output(i)->dims->size);
        for (int j =0; j < interpreter->output(i)->dims->size; j++)
        {
            printf("Output(%d) Dim (%d): \n", i, interpreter->output(i)->dims->data[j]);
        }
        output[i] = interpreter->output(i);
    }

    printf("Setup complete. Arena used: %d bytes\n", interpreter->arena_used_bytes());

    initialized = true;
}

Model::~Model() {
    if (interpreter != nullptr) {
        delete interpreter;
        interpreter = nullptr;
    }
    if (tensor_arena != nullptr) {
        if(this->inPSRAM)
            heap_caps_free(tensor_arena);
        else
            free(tensor_arena);
            
        tensor_arena = nullptr;
    }
    delete[] input;   input = nullptr;
    delete[] output;  output = nullptr;
}

void Model::callProfilerLog() const
{

    printf("Total Ticks %ld \n",profiler.GetTotalTicks());
}

void Model::getTotalProfileTimePerOp()
{
    //profiler.LogTicksPerTagCsv();
    profiler.Log();
}
void Model::ClearProfiler()
{
    profiler.ClearEvents();
}
bool Model::predict(const float* input_data, const int* input_lengths,
                float* results, const int* output_lengths) {
    
    if (!initialized) {
        printf("Not Intialized\n");
        return false;
    }

    int input_offset = 0;
    for(int i =0; i < input_size; i++) {
        if (input[i]->type == kTfLiteFloat32) {
            for (int j = 0; j < input_lengths[i]; j++) {
                input[i]->data.f[j] = input_data[input_offset + j];
            
            }
        } else if (input[i]->type == kTfLiteInt8) {

            for (int j = 0; j < input_lengths[i]; j++) {
                const float quantized = roundf(
                    input_data[input_offset + j] / input[i]->params.scale
                ) + input[i]->params.zero_point;
                // Clamp to int8 range to prevent overflow
                if      (quantized < -128.0f) input[i]->data.int8[j] = -128;
                else if (quantized >  127.0f) input[i]->data.int8[j] =  127;
                else                          input[i]->data.int8[j] = (int8_t)quantized;
                
            }
        } else {
            printf("Unkown data type in input %d\n", input[i]->type);
            return false;
        }
        input_offset += input_lengths[i];
    }
    

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke() failed!\n");
        return false;
    }

    int output_offset = 0;
    for(int i =0; i < output_size; i++) {
        ////ESP_LOGE("INTERNAL", "OUTPUT_SIZE %d", output_size);
        if (output[i]->type == kTfLiteFloat32) {
            
        for (int j = 0; j < output_lengths[i]; j++) {
            results[output_offset + j] = output[i]->data.f[j];
        }
        } else if (output[i]->type == kTfLiteInt8) {

            for (int j = 0; j < output_lengths[i]; j++) {

                results[output_offset + j] = (static_cast<float>(output[i]->data.int8[j])
                - output[i]->params.zero_point)
                * output[i]->params.scale;
               
            }
        } else {
            printf("Unkown data type in output\n");
            return false;
        }
        output_offset += output_lengths[i];
    }
    
    return true;
}
bool Model::predict(const int8_t* input_data, const int* input_lengths, int8_t* results, const int* output_lengths)
{
    if (!initialized) {
        return false;
    }

    int input_offset = 0;
    for(int i =0; i < input_size; i++) {
        if (input[i]->type == kTfLiteInt8) {
            for (int j = 0; j < input_lengths[i]; j++) {
               input[i]->data.int8[j] = input_data[input_offset + j];
            }
        } else {
            return false;
        }
        input_offset += input_lengths[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke() failed!\n");
        return false;
    }

    int output_offset = 0;
    for(int i =0; i < output_size; i++) {
        if (output[i]->type == kTfLiteInt8) {
            for (int j = 0; j < output_lengths[i]; j++) {
                results[output_offset + j] = output[i]->data.int8[j];
            }
        } else {
            return false;
        }
        output_offset += output_lengths[i];
    }

    return true;


}

bool Model::predict(const float* input_data, const int* input_lengths, int8_t* results, const int* output_lengths)
{
    if (!initialized) {
        return false;
    }

    int input_offset = 0;
    for(int i =0; i < input_size; i++) {
        if (input[i]->type == kTfLiteFloat32) {
        for (int j = 0; j < input_lengths[i]; j++) {
            input[i]->data.f[j] = input_data[input_offset + j];
           
        }
        } else if (input[i]->type == kTfLiteInt8) {

            for (int j = 0; j < input_lengths[i]; j++) {
                const float quantized = roundf(
                    input_data[input_offset + j] / input[i]->params.scale
                ) + input[i]->params.zero_point;
                // Clamp to int8 range to prevent overflow
                if      (quantized < -128.0f) input[i]->data.int8[j] = -128;
                else if (quantized >  127.0f) input[i]->data.int8[j] =  127;
                else                          input[i]->data.int8[j] = (int8_t)quantized;
                
            }
        } else {
            return false;
        }
        input_offset += input_lengths[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke() failed!\n");
        return false;
    }

    int output_offset = 0;
    for(int i =0; i < output_size; i++) {
        if (output[i]->type == kTfLiteFloat32) {
            
            for (int j = 0; j < output_lengths[i]; j++) {
                results[output_offset + j] = output[i]->data.f[j];
            }
        } else if (output[i]->type == kTfLiteInt8) {

            for (int j = 0; j < output_lengths[i]; j++) {

            results[output_offset + j] = output[i]->data.int8[j];
            }
        } else {
            return false;
        }
        output_offset += output_lengths[i];
    }

    return true;
}
size_t Model::getArenaUsedBytes() const {
    if (interpreter) return interpreter->arena_used_bytes();
    return 0;
}