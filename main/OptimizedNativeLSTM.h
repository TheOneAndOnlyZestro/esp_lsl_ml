// The main goal is to implement LSTM using ESP-NN native

#include "esp_nn.h"
#include "esp_timer.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
class OptimizedNativeLSTM
{
private:
    /* data */
    int8_t* m_x_weights; // size = input_size * h_size * 4 for i,o,g,f gates
    int32_t* m_x_bias;
    int16_t m_x_features;
    float* m_x_weight_scale;
    float* m_x_bias_scale; 

    // 4 * hidden_size = 4 * m_h_features

    int8_t* m_h_weights; // size = h_size * h_size * 4 for i,o,g,f gates
    int32_t* m_h_bias;
    int16_t m_h_features;
    float* m_h_weight_scale;
    float* m_h_bias_scale; 
public:
    OptimizedNativeLSTM();

    void set_x_h_weights_and_bias(const int8_t* x_weights,
                                const int32_t* x_bias,
                                const uint16_t x_features,
                                const float* x_weight_scale,
                                const float* x_bias_scale,

                                const int8_t* h_weights,
                                const int32_t* h_bias,
                                const uint16_t h_features,
                                const float* h_weight_scale,
                                const float* h_bias_scale);
    
    void run_inference(
        const int8_t* x,
        const int32_t x_zeropoint,
        const float x_scale,
    
        int8_t* y,
        const int32_t y_zeropoint,
        const float y_scale);
    ~OptimizedNativeLSTM();
};
