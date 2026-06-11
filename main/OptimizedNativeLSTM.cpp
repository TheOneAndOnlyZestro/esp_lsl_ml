// ============================================================
// tensor([[[  -5,  -92,  127,   31,  -54,   86,  -20,  -49, -128,   14,   44,
//            -27,   25,  -78,   82,   62, -100,   88,  -25,   26,   80,   70,
//             11,   83,   -8]]], dtype=torch.int8)
// ============================================================
// tensor([[[ -81,    9,   66,   94,    1,   -6,  -14,  -34,   32,   40,   57,
//            -64,   84,  -32,   68, -119,   23,  -44,   65,  -65,   32,   38,
//             58,  -73, -112]]], dtype=torch.int8)
// ============================================================
// tensor([[[  50,  -19,   55,   86,  -92,   84,  127,   58,   90,   67,   56,
//             32,  -11,  -95, -118, -128, -116,   54,  -74,  127,  -99,   51,
//            -87,  -11,   68]]], dtype=torch.int8)
// ============================================================
// tensor([[[ 0.0540,  0.0065, -0.0485,  0.4983, -0.5899,  0.2128,  0.3650,
//            0.1405,  0.1955,  0.0371,  0.1967,  0.1156, -0.4262, -0.3160,
//           -0.0788, -0.6411, -0.6167,  0.2650, -0.2469,  0.3399, -0.7626,
//            0.1093, -0.1309, -0.0398,  0.3109]]], grad_fn=<StackBackward0>)
// ============================================================
// tensor([[[ 0.0540,  0.0065, -0.0485,  0.4983, -0.5899,  0.2128,  0.3650,
//            0.1405,  0.1955,  0.0371,  0.1967,  0.1156, -0.4262, -0.3160,
//           -0.0788, -0.6411, -0.6167,  0.2650, -0.2469,  0.3399, -0.7626,
//            0.1093, -0.1309, -0.0398,  0.3109]]], grad_fn=<CopySlices>)
// ============================================================
// tensor([[[ 0.1874,  0.0110, -0.1107,  0.8707, -1.0227,  0.7805,  0.7290,
//            0.3383,  0.6244,  0.2004,  0.2361,  0.1438, -0.8352, -1.1308,
//           -0.4217, -0.8953, -1.4767,  0.6464, -0.7813,  0.5811, -1.3189,
//            0.2125, -0.1998, -0.0793,  0.7818]]], grad_fn=<CopySlices>)



// tensor([[[ 28,  57,  42, -37, -59,  44,  14,  11,  39,  46,  61,   5, -52,  -8,
//           -30,  54,  46,   4,  55, -16,  46, -25,  13, -13,  42]]],
//        dtype=torch.int8)
// ============================================================
// tensor([[[  -6,   41,  -74,   69,   -4,   33,   57,   54,  -76,   35,   96,
//             26,   36,   21,   46,  -41,  -47,  -12,  -25,   46,  -15,  -41,
//             40,   36,   16,   28,  127,    8,   59,  -29,  -32,   80,   85,
//            -18,  -17,  -69,   55,   55,   59,  -12,  -39,  -61,   64,  -31,
//           -128,   15,   38,   55,   52,  109,   94,  -44,  122,  -55,   40,
//              5,  -64,  -31,    9,   24,  -15,   29,   20,  -29,   67,  -15,
//              9,   51,   63,  -60,   28,  -52,   29,  -33,  -51,   61,   90,
//             36,    4,   47,  -45,  -37,    1,   55,   32,  -17,   46,   24,
//            -34,   19,   96,   -5,   25,  -71,   23,   22,  -59,  -91,   10,
//            106]]], dtype=torch.int8)

#include "OptimizedNativeLSTM.h"
OptimizedNativeLSTM::OptimizedNativeLSTM(/* args */)
{

}

OptimizedNativeLSTM::~OptimizedNativeLSTM()
{
}

void OptimizedNativeLSTM::set_x_h_weights_and_bias(  
                                const int8_t* x_weights,
                                const int32_t* x_bias,
                                const uint16_t x_features,
                                const float* x_weight_scale,
                                const float* x_bias_scale,

                                const int8_t* h_weights,
                                const int32_t* h_bias,
                                const uint16_t h_features,
                                const float* h_weight_scale,
                                const float* h_bias_scale)
{
    this->m_x_features = x_features;
    this->m_h_features = h_features;

    //allocate the space for the weights and biases first
    this->m_x_weights = new int8_t[this->m_x_features * this->m_h_features * 4];
    this->m_h_weights = new int8_t[this->m_h_features * this->m_h_features * 4];

    memcpy(this->m_x_weights, x_weights, this->m_x_features * this->m_h_features * 4);
    memcpy(this->m_h_weights, h_weights, this->m_h_features * this->m_h_features * 4);

    if(x_bias)
    {
        this->m_x_bias = new int32_t[this->m_h_features * 4];
        memcpy(this->m_x_bias, x_bias, this->m_h_features * 4 * sizeof(int32_t));

        this->m_x_bias_scale = new float[this->m_h_features * 4];
        memcpy(this->m_x_bias_scale, x_bias_scale, this->m_h_features * 4 * sizeof(float));
    }else { this->m_x_bias = nullptr; this->m_x_bias_scale = nullptr; }
    
    if(h_bias)
    {
        this->m_h_bias = new int32_t[this->m_h_features * 4];
        memcpy(this->m_h_bias, h_bias, this->m_h_features * 4 * sizeof(int32_t));

        this->m_h_bias_scale = new float[this->m_h_features * 4];
        memcpy(this->m_h_bias_scale, h_bias_scale, this->m_h_features * 4 * sizeof(float));
    }else { this->m_h_bias = nullptr; this->m_h_bias_scale = nullptr; }
    
    //allocate quantization params
    this->m_x_weight_scale = new float[this->m_h_features * 4];
    this->m_h_weight_scale = new float[this->m_h_features * 4];

    memcpy(this->m_x_weight_scale, x_weight_scale, this->m_h_features * 4 * sizeof(float));
    memcpy(this->m_h_weight_scale, h_weight_scale, this->m_h_features * 4 * sizeof(float));
}

void OptimizedNativeLSTM::calculate_per_ch_M(const float x_scale, const float y_scale)
{
    m_shifts = new int32_t[this->m_h_features * 4];
    m_mults = new int32_t[this->m_h_features * 4];
    for (size_t i = 0; i < this->m_h_features * 4; i++)
        {
            //printf("weight scale %0.4f\n", m_x_weight_scale[i]);
            float M = (x_scale * m_x_weight_scale[i]) / y_scale;
            m_shifts[i] = 0;
            while (M < 0.5f)
            { M *= 2; m_shifts[i]--; }
            while (M >= 1.0f) 
            { M *= 0.5f;m_shifts[i]++; }
            // M in range between [0.5, 1.0[ 
            // M * (2 ^ 31) range 0.5 * 2^31 - 2 ^ 31
            int64_t q = (int64_t)(M * (float)(1LL << 31) + 0.5f);
            if (q > INT32_MAX) q = INT32_MAX;

            m_mults[i] = (int32_t)q;
        }
}
void OptimizedNativeLSTM::run_inference(
       const int8_t* x,
        const int32_t x_zeropoint,
        int8_t* y,
        const int32_t y_zeropoint
    )
{
    // x_hat = matmul(x_weights, x) + x_bias : fc(input_channels = x_features, output_channels = h_features * 4)
    // h_hat = matmul(h_weights, h) + h_bias: fc(input_channels = h_features, ouput_channels = h_features * 4)
    // all_gates = x_hat + h_hat
    // slice all_gates into i,f,g,o
    // i = sigmoid(all_gates[:h_features])
    // f = sigmoid(all_gates[h_features: 2* h_features])
    // g = tanh(all_gates[2* h_features ; 3 * h_features])
    // o = sigmoid(all_gate[3 * h_features: 4 * h_features])
    // c_new = f mult c + i mult g
    // h_new = o mult tanh(c_new)
    
    //TESTING: x_hat = matmul(x_weights, x)
    // x_weights = (x_features, 4 * h_features)
    // x = (x_features)

    // get array of per channel mult and shifts
    

    //printf("BEGIN mult calculation\n");
    
    //printf("END mult calculation\n");
    // for(int i =0; i < this->m_h_features * 4; i++)
    // {
    //     printf("mult: %ld, shift: %ld\n", out_mults[i], out_shifts[i]);
    // }

    // for(int i=0; i < this->m_x_features; i++)
    // {
    //     printf("x(%d)[%d]\n", i, x[i]);
        
    // }

    // for(int i=0; i < this->m_x_features; i++)
    // {
    //     printf("w(%d)[%d]\n", i, m_x_weights[i]);
    // }
    
    //int8_t* x_gates = new int8_t[this->m_h_features * 4];
    // uint64_t startTime_first = esp_timer_get_time();
    esp_nn_fully_connected_per_ch_s8(
        x, x_zeropoint, this->m_x_features,
        this->m_x_weights, 0, nullptr, y,
        this->m_h_features * 4, y_zeropoint, m_shifts, m_mults,
        -127, 127);
    // uint64_t duration_first = esp_timer_get_time() - startTime_first;
    
    // printf("TIME FOR OP ONLY:  %lld \xCE\xBCs\n", duration_first);

}