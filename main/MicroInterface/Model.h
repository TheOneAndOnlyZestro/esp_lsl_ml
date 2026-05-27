#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <math.h>  // sqrtf, roundf
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
    tflite::MicroMutableOpResolver<32> resolver;

    bool initialized = false;
    ModelFlash* mflash;
public:
    // Constructor — override default 80 KB arena if a model needs more
    Model(ModelFlash* model_flash, const unsigned char* model_data, int arena_size = 80 * 1024, size_t input_size = 1, size_t output_size = 1);

    // Destructor — frees arena and interpreter
    ~Model();

    bool predict(const float** input_data, const int* input_lengths, float** results, int* output_lengths);

    bool isInitialized() const { return initialized; }
    size_t getArenaUsedBytes() const;
};

#endif // MODEL_H