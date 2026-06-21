// The main goal is to implement LSTM using ESP-NN native
#ifndef OPTIM_LSTM_H
#define OPTIM_LSTM_H

#include "esp_nn.h"
#include "esp_timer.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "lstm_nn_manifests/lstm_q.h"
#include "esp_heap_caps.h"
#define SIG_SCALE   (1.0f / 256.0f)
#define SIG_ZP      (-128)
#define TANH_SCALE  (1.0f / 128.0f)
#define TANH_ZP     (0)

class OptimizedNativeLSTM
{
    // x_hat = matmul(x_weights, x) + x_bias : fc(input_channels = x_features, output_channels = h_features * 4)
    // h_hat = matmul(h_weights, h) + h_bias: fc(input_channels = h_features, ouput_channels = h_features * 4)
    // all_gates = x_hat + h_hat
    // slice all_gates into i,f,g,o
    //if_gate = self.sig(gates[:, : 2 * self.hidden_size])
    //i_gate = if_gate[:, : self.hidden_size]
    //f_gate = if_gate[:, self.hidden_size : 2 * self.hidden_size]
    // g = tanh(all_gates[2* h_features : 3 * h_features])
    // o = sigmoid(all_gate[3 * h_features: 4 * h_features])
    // c_new = f mult c + i mult g
    // h_new = o mult tanh(c_new)
    
private:
    /* data */

    //weights and biases
    int8_t* m_x_l0_weights; // size = input_size * h_size * 4 for i,o,g,f gates
    int32_t* m_x_l0_bias;

    int8_t* m_h_l0_weights; // size = input_size * h_size * 4 for i,o,g,f gates
    int32_t* m_h_l0_bias;

    int8_t* m_x_l1_weights; // size = input_size * h_size * 4 for i,o,g,f gates
    int32_t* m_x_l1_bias;

    int8_t* m_h_l1_weights; // size = input_size * h_size * 4 for i,o,g,f gates
    int32_t* m_h_l1_bias;

    int x_features;
    int h_features;

    int8_t m_tanh_g_lut[2][256];    // tanh for g-gate (gates space)
    int8_t m_tanh_c_lut[2][256];    // tanh for c_new   (c_new space)
    int8_t m_sig_lut[2][256];       // sigmoid (gates space)  -- promote existing to per-layer

    int8_t* m_tanh_c16_lut[2] = {nullptr, nullptr};
    struct EwParams {
        // gates = x_hat + h_hat   (add: two inputs in different scales)
        int32_t add_in1_mult, add_in1_shift;   // from s_xhat
        int32_t add_in2_mult, add_in2_shift;   // from s_hhat
        int32_t add_out_mult, add_out_shift;   // to s_gates
        int32_t add_left_shift;
 
        // c*f -> c_new scale
        int32_t cf_mult, cf_shift;
        // i*g -> c_new scale
        int32_t ig_mult, ig_shift;
        // (c*f)+(i*g) : both already in s_c_new -> identity-ish add to s_c_new
        int32_t cnew_in1_mult, cnew_in1_shift;
        int32_t cnew_in2_mult, cnew_in2_shift;
        int32_t cnew_out_mult, cnew_out_shift;
        int32_t cnew_left_shift;
 
        // o * tanh(c_new) -> h_new scale
        int32_t oh_mult, oh_shift;
 
        // activation (gate input) scales/zps for the layer
        float s_gates;  int32_t z_gates;
        float s_c_new;  int32_t z_c_new;
        float s_h_new;  int32_t z_h_new;
    };

    EwParams m_ew[2]; //for each layer

    int H;
    int G;

    //activations buffers
    int8_t* m_xhat;     // [G]
    int8_t* m_hhat;     // [G]
    int8_t* m_gates;    // [G]
    int8_t* m_if;       // [2H] sigmoid(i,f)
    int8_t* m_g;        // [H]  tanh(g)
    int8_t* m_o;        // [H]  sigmoid(o)
    int8_t* m_cf;       // [H]  c*f
    int8_t* m_ig;       // [H]  i*g
    int8_t* m_tanh_c;   // [H]  tanh(c_new)
    int16_t* m_cf16   = nullptr;   // [H]
    int16_t* m_ig16   = nullptr;   // [H]
public:

    static void quantize_multiplier(double M, int32_t *mult, int32_t *shift);

    //preparing values for int + int element wise addition
    void prepare_add(float s_in1, float s_in2, float s_out,
        int32_t *m1,int32_t *s1,int32_t *m2,int32_t *s2,
        int32_t *mo,int32_t *so,int32_t *left_shift);

    
    // defualt initialization
    OptimizedNativeLSTM()
    :m_x_l0_weights(nullptr), m_x_l0_bias(nullptr),
      m_h_l0_weights(nullptr), m_h_l0_bias(nullptr),
      m_x_l1_weights(nullptr), m_x_l1_bias(nullptr),
      m_h_l1_weights(nullptr), m_h_l1_bias(nullptr),
      x_features(0), h_features(0),
      m_xhat(nullptr), m_hhat(nullptr), m_gates(nullptr),
      m_if(nullptr), m_g(nullptr), m_o(nullptr),
      m_cf(nullptr), m_ig(nullptr), m_tanh_c(nullptr) {}


    void set_weights(
        int xf, int hf,
        int8_t* x0w,int32_t* x0b, int8_t* h0w,int32_t* h0b,
        int8_t* x1w,int32_t* x1b, int8_t* h1w,int32_t* h1b
    );

    // Function to build all LUTs and all elementwise multiplier from header
    void prepare();

    // ESP-NN does not provide a default implementation for tanh activation
    static void build_tanh_lut(int8_t *lut, int32_t in_zp, float in_scale);
    void build_tanh_lut_c16(int8_t* lut, int32_t in_zp, float in_scale);
    
    void run_cell_step_c16(
    int layer,
    const int8_t* in_x, int row_len_x, int32_t xin_off,
    const int8_t* in_h, int32_t hin_off, int32_t cin_off,
    const int16_t* in_c,
    int16_t* out_c, int8_t* out_h);

    static inline int32_t requant_s64(int64_t acc, int32_t mult, int32_t shift);
    
    void run_timestep_q_c16(const int8_t* x_q, int8_t* h_q,
                            int16_t* c_q, int8_t* y_out);
    void run_cell_step(
        int layer,
        const int8_t* in_x, int row_len_x, int32_t xin_off,
        const int8_t* in_h, int32_t hin_off, int32_t cin_off,
        const int8_t* in_c,
        int8_t* out_c, int8_t* out_h
    );

    void run_timestep_q(
        const int8_t* x_q, int8_t* h_q, 
        int8_t* c_q, int8_t* y_out
    );
    ~OptimizedNativeLSTM() {
        free(m_xhat); free(m_hhat); free(m_gates);
        free(m_if); free(m_g); free(m_o);
        free(m_cf); free(m_ig); free(m_tanh_c);
    }
};
#endif