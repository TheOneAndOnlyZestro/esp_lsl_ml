#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <math.h>  // sqrtf, roundf
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Auto-generated from scalers/metadata.json via generate_scaler_header.py
// Contains FEATURE_MEAN, FEATURE_SCALE, EULER_MEAN, EULER_SCALE, NUM_FEATURES, NUM_EULER
#include "scaler_params.h"

class Model {
private:
    const tflite::Model* tflite_model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // Pointer to the dynamically allocated memory
    uint8_t* tensor_arena = nullptr;
    int arena_size = 0;

    // Resolver with enough capacity for our benchmarking ops
    tflite::MicroMutableOpResolver<23> resolver;

    bool initialized = false;

    // ----------------------------------------------------------------
    // Scaler helpers (private — called inside predict())
    // ----------------------------------------------------------------

    /**
     * Normalize a flat feature buffer IN-PLACE using the fitted StandardScaler.
     *
     *   normalized[i] = (raw[i] - FEATURE_MEAN[channel]) / FEATURE_SCALE[channel]
     *
     * Buffer layout (row-major, time x features):
     *   [t0_f0, t0_f1 ... t0_f8, t1_f0, t1_f1 ... tN_f8]
     * so channel index = i % NUM_FEATURES.
     *
     * NOTE: call this on a COPY of input_data, never on the original.
     */
    void normalizeFeatures(float* buffer, int length) const {
        for (int i = 0; i < length; i++) {
            const int ch = i % NUM_FEATURES;
            buffer[i] = (buffer[i] - FEATURE_MEAN[ch]) / FEATURE_SCALE[ch];
        }
    }

    /**
     * Denormalize model output IN-PLACE back to real-world units.
     *
     * Euler angles (indices 0 .. NUM_EULER-1):
     *   degrees[i] = normalized[i] * EULER_SCALE[i] + EULER_MEAN[i]
     *
     * Quaternions (indices NUM_EULER .. output_length-1):
     *   Re-normalized to unit norm — TFLite quantization can break the
     *   unit-norm constraint, so we always enforce it here.
     */
    void denormalizeOutput(float* results, int output_length) const {
        // Euler: inverse StandardScaler -> degrees
        for (int i = 0; i < NUM_EULER && i < output_length; i++) {
            results[i] = results[i] * EULER_SCALE[i] + EULER_MEAN[i];
        }
        // Quaternion: re-normalize to unit norm
        const int q_start = NUM_EULER;
        if (output_length > q_start) {
            float norm_sq = 0.0f;
            for (int i = q_start; i < output_length; i++) {
                norm_sq += results[i] * results[i];
            }
            if (norm_sq > 1e-8f) {
                const float inv_norm = 1.0f / sqrtf(norm_sq);
                for (int i = q_start; i < output_length; i++) {
                    results[i] *= inv_norm;
                }
            }
        }
    }

public:
    // Constructor — override default 80 KB arena if a model needs more
    Model(const unsigned char* model_data, int arena_size = 80 * 1024);

    // Destructor — frees arena and interpreter
    ~Model();

    // Runs inference. Returns true on success.
    // input_data : raw (un-normalized) sensor readings, length = seq_len * NUM_FEATURES
    // results    : caller-allocated output buffer, length = output_length (7 for Euler+Quat)
    //              On return: results[0..2] = Euler angles in degrees
    //                         results[3..6] = unit quaternion
    bool predict(const float* input_data, int input_length,
                 float* results, int output_length);

    bool isInitialized() const { return initialized; }
    size_t getArenaUsedBytes() const;
};

#endif // MODEL_H