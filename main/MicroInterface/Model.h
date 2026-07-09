#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <math.h>  // sqrtf, roundf
#include <esp_timer.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "model_flash.h"

class Model {
private:
    const tflite::Model* tflite_model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    
    TfLiteTensor** input = nullptr;
    TfLiteTensor** output = nullptr;

    size_t input_size = 0;
    size_t output_size = 0;

    // Pointer to the dynamically allocated memory
    uint8_t* tensor_arena = nullptr;
    int arena_size = 0;

    // Resolver with enough capacity for our benchmarking ops
    tflite::MicroMutableOpResolver<34> resolver;
    
    bool initialized = false;
    ModelFlash* mflash;

    bool inPSRAM;

    bool owns_arena = true;          // false when arena is borrowed (persistent)
    void init_common(const unsigned char* model_data, char* report_buffer, int size);
    
    uint64_t m_predict_time;
    tflite::MicroProfiler profiler;
public:
    // Constructor — override default 80 KB arena if a model needs more
    Model(ModelFlash* model_flash, const unsigned char* model_data, int arena_size, 
        size_t input_size, size_t output_size, bool usePSRAM,char* report_buffer, int size);

     Model(ModelFlash* model_flash, const unsigned char* model_data,
        uint8_t* external_arena, int arena_size,
        size_t input_size, size_t output_size, bool usePSRAM, char* report_buffer, int size);

    // Destructor — frees arena and interpreter
    ~Model();

    bool predict(const float* input_data, const int* input_lengths, float* results, const int* output_lengths);
    bool predict(const int8_t* input_data, const int* input_lengths, int8_t* results, const int* output_lengths);
    bool predict(const float* input_data, const int* input_lengths, int8_t* results, const int* output_lengths);
    bool predict(const int8_t* input_data, const int* input_lengths, float* results, const int* output_lengths);

    float   getInputScale(int i)      const { return input[i]->params.scale; }
    int32_t getInputZeroPoint(int i)  const { return input[i]->params.zero_point; }
    float   getOutputScale(int i)     const { return output[i]->params.scale; }
    int32_t getOutputZeroPoint(int i) const { return output[i]->params.zero_point; }
    int     getInputCount()           const { return (int)input_size; }
    int     getOutputCount()          const { return (int)output_size; }
    TfLiteType getInputType(int i)    const {return input[i]->type; }
    
    void getTotalProfileTimePerOp() ;
    int32_t getTotalTicks() const;
    int32_t getTotalTime() const;
    void callProfilerLog() const;
    bool isInitialized() const { return initialized; }

    void ClearProfiler();
    size_t getArenaUsedBytes() const;
};

#endif // MODEL_H