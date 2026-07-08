#include "OptimizedNativeLSTM.h"

// round-to-nearest fixed-point requant with int64 acc (handles int16 products)
int32_t OptimizedNativeLSTM::requant_s64(int64_t acc, int32_t mult, int32_t shift)
{
    // acc * mult is Q31; shift is the (usually negative) exponent
    int64_t prod = acc * (int64_t)mult;
    // round to nearest at the (31 - shift) bit position
    int total_shift = 31 - shift;   // shift is negative -> total_shift > 31
    int64_t rounding = (int64_t)1 << (total_shift - 1);
    int64_t res = (prod + rounding) >> total_shift;
    return (int32_t)res;
}

void OptimizedNativeLSTM::build_tanh_lut_c16(int8_t* lut, int32_t in_zp, float in_scale)
{
    for (int i = 0; i < 65536; ++i) {
        int16_t qx = (int16_t)i;                      // reinterpret index as int16
        float dq = ((int32_t)qx - in_zp) * in_scale;  // dequant c_new
        float t  = tanhf(dq);
        int32_t q = (int32_t)lroundf(t * 128.0f) + TANH_ZP;   // tanh out scale 1/128, int8
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        lut[i] = (int8_t)q;
    }
}
void OptimizedNativeLSTM::quantize_multiplier(double M, int32_t *mult, int32_t *shift)
{
    if (M == 0.0) { *mult = 0; *shift = 0; return; }
    int exp;
    double mant = frexp(M, &exp);          // M = mant * 2^exp, mant in [0.5,1)
    int64_t q = (int64_t)llround(mant * (double)(1LL << 31));
    if (q == (1LL << 31)) { q >>= 1; exp += 1; }
    if (exp < -31) { *mult = 0; *shift = 0; return; }
    *mult  = (int32_t)q;
    *shift = exp;
}

void OptimizedNativeLSTM::prepare_add(float s_in1, float s_in2, float s_out,
                     int32_t *m1,int32_t *s1,int32_t *m2,int32_t *s2,
                     int32_t *mo,int32_t *so,int32_t *left_shift)
{
    const int LS = 20;
    double twice = (s_in1 > s_in2) ? s_in1 : s_in2;
    double in1 = s_in1 / (twice * (double)(1 << LS));
    double in2 = s_in2 / (twice * (double)(1 << LS));
    double out = twice / s_out;
    quantize_multiplier(in1, m1, s1);
    quantize_multiplier(in2, m2, s2);
    quantize_multiplier(out, mo, so);
    *left_shift = LS;
}
void OptimizedNativeLSTM::prepare()
{
    H = LSTM_Q_HIDDEN;
    G = 4 * H;

    // layer 0
    esp_nn_logistic_s8_prepare(m_sig_lut[0],  LSTM_Q_L0_GATES_ZP,  LSTM_Q_L0_GATES_SCALE);
    build_tanh_lut(m_tanh_g_lut[0], LSTM_Q_L0_GATES_ZP,  LSTM_Q_L0_GATES_SCALE);
    build_tanh_lut(m_tanh_c_lut[0], LSTM_Q_L0_C_NEW_ZP,  LSTM_Q_L0_C_NEW_SCALE);

    // layer 1
    esp_nn_logistic_s8_prepare(m_sig_lut[1],  LSTM_Q_L1_GATES_ZP,  LSTM_Q_L1_GATES_SCALE);
    build_tanh_lut(m_tanh_g_lut[1], LSTM_Q_L1_GATES_ZP,  LSTM_Q_L1_GATES_SCALE);
    build_tanh_lut(m_tanh_c_lut[1], LSTM_Q_L1_C_NEW_ZP,  LSTM_Q_L1_C_NEW_SCALE);

    // layer 0
    // gates = x_hat + h_hat 
    prepare_add(LSTM_Q_L0_X_HAT_SCALE, LSTM_Q_L0_H_HAT_SCALE, LSTM_Q_L0_GATES_SCALE,
            &m_ew[0].add_in1_mult,  &m_ew[0].add_in1_shift,
            &m_ew[0].add_in2_mult,  &m_ew[0].add_in2_shift,
            &m_ew[0].add_out_mult,  &m_ew[0].add_out_shift,
            &m_ew[0].add_left_shift);
    
    quantize_multiplier((double)LSTM_Q_L0_C_NEW_SCALE /* placeholder s_c */ * SIG_SCALE
                            / LSTM_Q_L0_C_NEW_SCALE, &m_ew[0].cf_mult,&m_ew[0].cf_shift);
    
    quantize_multiplier((double)SIG_SCALE * TANH_SCALE / LSTM_Q_L0_C_NEW_SCALE,
                        &m_ew[0].ig_mult,&m_ew[0].ig_shift);

    
    // c_new = cf +ig
    prepare_add(LSTM_Q_L0_C_NEW_SCALE, LSTM_Q_L0_C_NEW_SCALE, LSTM_Q_L0_C_NEW_SCALE,
            &m_ew[0].cnew_in1_mult,&m_ew[0].cnew_in1_shift,
            &m_ew[0].cnew_in2_mult,&m_ew[0].cnew_in2_shift,
            &m_ew[0].cnew_out_mult,&m_ew[0].cnew_out_shift,&m_ew[0].cnew_left_shift);
    quantize_multiplier((double)SIG_SCALE * TANH_SCALE / LSTM_Q_L0_H_NEW_SCALE,
                &m_ew[0].oh_mult,&m_ew[0].oh_shift);
    m_ew[0].s_gates=LSTM_Q_L0_GATES_SCALE; m_ew[0].z_gates=LSTM_Q_L0_GATES_ZP;
    m_ew[0].s_c_new=LSTM_Q_L0_C_NEW_SCALE; m_ew[0].z_c_new=LSTM_Q_L0_C_NEW_ZP;
    m_ew[0].s_h_new=LSTM_Q_L0_H_NEW_SCALE; m_ew[0].z_h_new=LSTM_Q_L0_H_NEW_ZP;

    // layer 1
    prepare_add(LSTM_Q_L1_X_HAT_SCALE, LSTM_Q_L1_H_HAT_SCALE, LSTM_Q_L1_GATES_SCALE,
            &m_ew[1].add_in1_mult,&m_ew[1].add_in1_shift,
            &m_ew[1].add_in2_mult,&m_ew[1].add_in2_shift,
            &m_ew[1].add_out_mult,&m_ew[1].add_out_shift,&m_ew[1].add_left_shift);
    quantize_multiplier((double)LSTM_Q_L1_C_NEW_SCALE * SIG_SCALE / LSTM_Q_L1_C_NEW_SCALE,
                        &m_ew[1].cf_mult,&m_ew[1].cf_shift);
    quantize_multiplier((double)SIG_SCALE * TANH_SCALE / LSTM_Q_L1_C_NEW_SCALE,
                        &m_ew[1].ig_mult,&m_ew[1].ig_shift);
    prepare_add(LSTM_Q_L1_C_NEW_SCALE, LSTM_Q_L1_C_NEW_SCALE, LSTM_Q_L1_C_NEW_SCALE,
        &m_ew[1].cnew_in1_mult,&m_ew[1].cnew_in1_shift,
        &m_ew[1].cnew_in2_mult,&m_ew[1].cnew_in2_shift,
        &m_ew[1].cnew_out_mult,&m_ew[1].cnew_out_shift,&m_ew[1].cnew_left_shift);

    quantize_multiplier((double)SIG_SCALE * TANH_SCALE / LSTM_Q_L1_H_NEW_SCALE,
                        &m_ew[1].oh_mult,&m_ew[1].oh_shift);
    m_ew[1].s_gates=LSTM_Q_L1_GATES_SCALE; m_ew[1].z_gates=LSTM_Q_L1_GATES_ZP;
    m_ew[1].s_c_new=LSTM_Q_L1_C_NEW_SCALE; m_ew[1].z_c_new=LSTM_Q_L1_C_NEW_ZP;
    m_ew[1].s_h_new=LSTM_Q_L1_H_NEW_SCALE; m_ew[1].z_h_new=LSTM_Q_L1_H_NEW_ZP;

    // initialize activation buffers
    m_xhat=(int8_t*)malloc(G); m_hhat=(int8_t*)malloc(G); m_gates=(int8_t*)malloc(G);
    m_if=(int8_t*)malloc(2*H); m_g=(int8_t*)malloc(H); m_o=(int8_t*)malloc(H);
    m_cf=(int8_t*)malloc(H); m_ig=(int8_t*)malloc(H); m_tanh_c=(int8_t*)malloc(H);

    m_tanh_c16_lut[0] = (int8_t*)heap_caps_malloc(65536, MALLOC_CAP_SPIRAM);
    m_tanh_c16_lut[1] = (int8_t*)heap_caps_malloc(65536, MALLOC_CAP_SPIRAM);
    build_tanh_lut_c16(m_tanh_c16_lut[0], LSTM_Q_L0_C_NEW_ZP, LSTM_Q_L0_C_NEW_SCALE);
    build_tanh_lut_c16(m_tanh_c16_lut[1], LSTM_Q_L1_C_NEW_ZP, LSTM_Q_L1_C_NEW_SCALE);

    m_cf16 = (int16_t*)malloc(H * sizeof(int16_t));
    m_ig16 = (int16_t*)malloc(H * sizeof(int16_t));
}

void OptimizedNativeLSTM::build_tanh_lut(int8_t *lut, int32_t in_zp, float in_scale)
{
    for (int i = 0; i < 256; ++i) {
        int8_t input_val = (int8_t)i;
        float dq = (input_val - in_zp) * in_scale;
        float t = tanhf(dq);
        int32_t q = (int32_t)lroundf(t * 128.0f) + TANH_ZP;  // scale 1/128
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        lut[i] = (int8_t)q;
    }
}

void OptimizedNativeLSTM::run_cell_step(
    int layer,
    const int8_t* in_x, int row_len_x, int32_t xin_off,
    const int8_t* in_h, int32_t hin_off, int32_t cin_off,
    const int8_t* in_c,
    int8_t* out_c, int8_t* out_h
)
{
    const EwParams& P = m_ew[layer];
 
    int8_t  *xw,*hw; int32_t *xb,*hb;
    const int32_t *x_mult,*x_shift,*h_mult,*h_shift;
    int32_t x_in_off, x_out_off, h_in_off, h_out_off;


    if (layer==0) {
        xw=m_x_l0_weights; xb=m_x_l0_bias; hw=m_h_l0_weights; hb=m_h_l0_bias;
        x_mult=lstm_q_L0_x_out_mult; x_shift=lstm_q_L0_x_out_shift;
        h_mult=lstm_q_L0_h_out_mult; h_shift=lstm_q_L0_h_out_shift;
        x_in_off=LSTM_Q_L0_X_INPUT_OFFSET; x_out_off=LSTM_Q_L0_X_OUT_OFFSET;
        h_in_off=LSTM_Q_L0_H_INPUT_OFFSET; h_out_off=LSTM_Q_L0_H_OUT_OFFSET;
    } else {
        xw=m_x_l1_weights; xb=m_x_l1_bias; hw=m_h_l1_weights; hb=m_h_l1_bias;
        x_mult=lstm_q_L1_x_out_mult; x_shift=lstm_q_L1_x_out_shift;
        h_mult=lstm_q_L1_h_out_mult; h_shift=lstm_q_L1_h_out_shift;
        x_in_off=LSTM_Q_L1_X_INPUT_OFFSET; x_out_off=LSTM_Q_L1_X_OUT_OFFSET;
        h_in_off=LSTM_Q_L1_H_INPUT_OFFSET; h_out_off=LSTM_Q_L1_H_OUT_OFFSET;
    }

    // if (layer == 0)
    // {
    //     for (int i =0; i < (56); i++)
    //     {
    //         //dequant first
    //         float dequant = ((int32_t)in_h[i] - LSTM_Q_IN_H_ZP) * LSTM_Q_IN_H_SCALE;
    //         printf("First H output (%d): %0.6f\n", i, dequant);
    //     }
    // }  
    // Calculate x_hat, WORKS
    esp_nn_fully_connected_per_ch_s8(
        in_x, x_in_off, (uint16_t)row_len_x, xw, /*filter_offset*/0,
        xb, m_xhat, (uint16_t)G, x_out_off, x_shift, x_mult, -128, 127);
    
    
    // Calculate h_hat, WORKS
    esp_nn_fully_connected_per_ch_s8(
            in_h, h_in_off, (uint16_t)H, hw, 0,
            hb, m_hhat, (uint16_t)G, h_out_off, h_shift, h_mult, -128, 127);
    
    // if (layer == 0)
    // {
    //     for (int i =0; i < (56 * 4); i++)
    //     {
    //         //dequant first
    //         float dequant = ((int32_t)m_hhat[i] - LSTM_Q_L0_H_HAT_ZP) * LSTM_Q_L0_H_HAT_SCALE;
    //         printf("First HHAT output (%d): %0.6f\n", i, dequant);
    //     }
    // }    
    // Calculate gates, WORKS
    esp_nn_add_elementwise_s8(
    m_xhat, m_hhat,
    -x_out_off, -h_out_off,                       // re-center each input
    P.add_in1_mult, P.add_in2_mult,
    P.add_in1_shift, P.add_in2_shift,
    P.add_left_shift,
    m_gates, P.z_gates, P.add_out_mult, P.add_out_shift,
    -128, 127, G);
    

    
    // apply and slice gates
    esp_nn_logistic_s8(m_gates, m_if, 2*H, m_sig_lut[layer]);
    

    // tanh(g): g is gates[2H:3H]
    for (int k=0;k<H;k++) m_g[k]=m_tanh_g_lut[layer][(uint8_t)m_gates[2*H+k]];
    // sigmoid(o): gates[3H:4H]
    for (int k=0;k<H;k++) m_o[k]=m_sig_lut[layer][(uint8_t)m_gates[3*H+k]];

    const int8_t* i_gate = m_if;            // [0:H]
    const int8_t* f_gate = m_if + H;        // [H:2H]

    // c * f, trying negation of zero points
    esp_nn_mul_elementwise_s8(
        in_c, f_gate, -cin_off, -SIG_ZP,
        m_cf, P.z_c_new, P.cf_mult, P.cf_shift, -128, 127, H);

    // i * g
    esp_nn_mul_elementwise_s8(
            i_gate, m_g, -SIG_ZP, -TANH_ZP,
            m_ig, P.z_c_new, P.ig_mult, P.ig_shift, -128, 127, H);
    
    // c_new =. c * f + i*g
    esp_nn_add_elementwise_s8(
            m_cf, m_ig, -P.z_c_new, -P.z_c_new,
            P.cnew_in1_mult, P.cnew_in2_mult,
            P.cnew_in1_shift, P.cnew_in2_shift, P.cnew_left_shift,
            out_c, P.z_c_new, P.cnew_out_mult, P.cnew_out_shift, -128, 127, H);
    
    // tanh(c_new)
    for (int k=0;k<H;k++) m_tanh_c[k]=m_tanh_c_lut[layer][(uint8_t)out_c[k]];

    // h_new = o * tanh(c_new)
    esp_nn_mul_elementwise_s8(
            m_o, m_tanh_c, -SIG_ZP, -TANH_ZP,
            out_h, P.z_h_new, P.oh_mult, P.oh_shift, -128, 127, H);

    // if (layer == 0)
    // {
    //     printf("IG_mult: %ld, IG_SHIFT: %ld\n", P.ig_mult, P.ig_shift);
    //     for (int i =0; i < (56); i++)
    //     {
    //         //dequant first
    //         float dequant = ((int32_t)i_gate[i] - SIG_ZP) * SIG_SCALE;;
    //         printf("First i output (%d): %0.6e\n", i, dequant);
    //     }
    //     for (int i =0; i < (56); i++)
    //     {
    //         //dequant first
    //         float dequant = ((int32_t)m_g[i] - TANH_ZP) * TANH_SCALE;
    //         printf("First g output (%d): %0.6e\n", i, dequant);
    //     }
    //     for (int i =0; i < (56); i++)
    //     {
    //         //dequant first
    //         float dequant = ((int32_t)m_ig[i] - LSTM_Q_L0_C_NEW_ZP) * LSTM_Q_L0_C_NEW_SCALE;
    //         printf("First ig output (%d): %0.6e\n", i, dequant);
    //     }
    //     for (int i =0; i < (56); i++)
    //     {
    //         //dequant first
    //         float dequant = ((int32_t)m_cf[i] - LSTM_Q_L0_C_NEW_ZP) * LSTM_Q_L0_C_NEW_SCALE;
    //         printf("First cf output (%d): %0.6e\n", i, dequant);
    //     }

    //     for (int i =0; i < (56); i++)
    //     {
    //         //dequant first
    //         float dequant = ((int32_t)out_h[i] - LSTM_Q_L0_H_NEW_ZP) * LSTM_Q_L0_H_NEW_SCALE;
    //         printf("First h_new output (%d): %0.6e\n", i, dequant);
    //     }
    // }
}

void OptimizedNativeLSTM::run_cell_step_c16(
    int layer,
    const int8_t* in_x, int row_len_x, int32_t xin_off,
    const int8_t* in_h, int32_t hin_off, int32_t cin_off,
    const int16_t* in_c,
    int16_t* out_c, int8_t* out_h)
{
    // issue in h_hat, second layer
    const EwParams& P = m_ew[layer];
    int8_t *xw,*hw; int32_t *xb,*hb;
    const int32_t *x_mult,*x_shift,*h_mult,*h_shift;
    int32_t x_in_off, x_out_off, h_in_off, h_out_off;

    if (layer==0) {
        xw=m_x_l0_weights; xb=m_x_l0_bias; hw=m_h_l0_weights; hb=m_h_l0_bias;
        x_mult=lstm_q_L0_x_out_mult; x_shift=lstm_q_L0_x_out_shift;
        h_mult=lstm_q_L0_h_out_mult; h_shift=lstm_q_L0_h_out_shift;
        x_in_off=LSTM_Q_L0_X_INPUT_OFFSET; x_out_off=LSTM_Q_L0_X_OUT_OFFSET;
        h_in_off=LSTM_Q_L0_H_INPUT_OFFSET; h_out_off=LSTM_Q_L0_H_OUT_OFFSET;
    } else {
        xw=m_x_l1_weights; xb=m_x_l1_bias; hw=m_h_l1_weights; hb=m_h_l1_bias;
        x_mult=lstm_q_L1_x_out_mult; x_shift=lstm_q_L1_x_out_shift;
        h_mult=lstm_q_L1_h_out_mult; h_shift=lstm_q_L1_h_out_shift;
        x_in_off=LSTM_Q_L1_X_INPUT_OFFSET; x_out_off=LSTM_Q_L1_X_OUT_OFFSET;
        h_in_off=LSTM_Q_L1_H_INPUT_OFFSET; h_out_off=LSTM_Q_L1_H_OUT_OFFSET;
    }
    
    // gates: identical int8 path (x_hat, h_hat, add, sig/tanh-g) -- unchanged
    esp_nn_fully_connected_per_ch_s8(in_x, x_in_off, (uint16_t)row_len_x, xw, 0,
        xb, m_xhat, (uint16_t)G, x_out_off, x_shift, x_mult, -128, 127);

    
    esp_nn_fully_connected_per_ch_s8(in_h, h_in_off, (uint16_t)H, hw, 0,
        hb, m_hhat, (uint16_t)G, h_out_off, h_shift, h_mult, -128, 127);

    esp_nn_add_elementwise_s8(m_xhat, m_hhat, -x_out_off, -h_out_off,
        P.add_in1_mult, P.add_in2_mult, P.add_in1_shift, P.add_in2_shift, P.add_left_shift,
        m_gates, P.z_gates, P.add_out_mult, P.add_out_shift, -128, 127, G);

    esp_nn_logistic_s8(m_gates, m_if, 2*H, m_sig_lut[layer]);

    

    for (int k=0;k<H;k++) m_g[k]=m_tanh_g_lut[layer][(uint8_t)m_gates[2*H+k]];
    for (int k=0;k<H;k++) m_o[k]=m_sig_lut[layer][(uint8_t)m_gates[3*H+k]];

    const int8_t* i_gate = m_if;
    const int8_t* f_gate = m_if + H;

    // ---- int16 cell-state path (hand-written) ----
    const int32_t c_zp     = cin_off;        // raw zp passed in (IN_C_ZP or prev C_NEW_ZP)
    const int32_t cnew_zp  = P.z_c_new;      // int16 c_new zero-point

    // c * f  -> int16 c_new space
    for (int k=0;k<H;k++) {
        int32_t cv = (int32_t)in_c[k] - c_zp;            // int16 c recentered
        int32_t fv = (int32_t)f_gate[k] + (-SIG_ZP);     // == f_gate[k] + 128
        int64_t acc = (int64_t)cv * (int64_t)fv;
        int32_t q = requant_s64(acc, P.cf_mult, P.cf_shift) + cnew_zp;
        if (q < -32768) q = -32768; else if (q > 32767) q = 32767;
        m_cf16[k] = (int16_t)q;
    }

    // i * g -> int16 c_new space
    for (int k=0;k<H;k++) {
        int32_t iv = (int32_t)i_gate[k] + (-SIG_ZP);     // +128
        int32_t gv = (int32_t)m_g[k]    + (-TANH_ZP);    // +0
        int64_t acc = (int64_t)iv * (int64_t)gv;
        int32_t q = requant_s64(acc, P.ig_mult, P.ig_shift) + cnew_zp;
        if (q < -32768) q = -32768; else if (q > 32767) q = 32767;
        m_ig16[k] = (int16_t)q;
    }

    // c_new = cf + ig  (dual-scale add, both in c_new space -> c_new space, int16)
    for (int k=0;k<H;k++) {
        int32_t a = (int32_t)m_cf16[k] - cnew_zp;
        int32_t b = (int32_t)m_ig16[k] - cnew_zp;
        // mirror esp_nn add: shift up by left_shift, apply per-input mult, sum, requant out
        int64_t sa = (int64_t)(a << P.cnew_left_shift);
        int64_t sb = (int64_t)(b << P.cnew_left_shift);
        int64_t ra = (int64_t)requant_s64(sa, P.cnew_in1_mult, P.cnew_in1_shift);
        int64_t rb = (int64_t)requant_s64(sb, P.cnew_in2_mult, P.cnew_in2_shift);
        int32_t q  = requant_s64(ra + rb, P.cnew_out_mult, P.cnew_out_shift) + cnew_zp;
        if (q < -32768) q = -32768; else if (q > 32767) q = 32767;
        out_c[k] = (int16_t)q;
    }

    // tanh(c_new): int16 -> int8 via 64K LUT
    for (int k=0;k<H;k++) m_tanh_c[k] = m_tanh_c16_lut[layer][(uint16_t)out_c[k]];

    // h_new = o * tanh(c_new) -> int8 (unchanged int8 mul)
    esp_nn_mul_elementwise_s8(m_o, m_tanh_c, -SIG_ZP, -TANH_ZP,
        out_h, P.z_h_new, P.oh_mult, P.oh_shift, -128, 127, H);


    //printf("IG_mult: %ld, IG_SHIFT: %ld\n", P.ig_mult, P.ig_shift);

    if(layer == 0)
    {
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)in_h[i] - LSTM_Q_IN_H_ZP) * LSTM_Q_IN_H_SCALE;
        //     printf("l(%d) h (%d): %0.6e\n", layer, i, dequant);
        // }
        // for (int i =0; i < (56 * 4); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_hhat[i] - LSTM_Q_L0_H_HAT_ZP) * LSTM_Q_L0_H_HAT_SCALE;
        //     printf("l(%d) h_hat (%d): %0.6e\n", layer, i, dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)i_gate[i] - SIG_ZP) * SIG_SCALE;
        //     printf("l(%d) i output (%d): %0.6e\n", layer, i, dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_g[i] - TANH_ZP) * TANH_SCALE;
        //     printf("l(%d) g output (%d): %0.6e\n", layer,i, dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_ig16[i] - LSTM_Q_L0_C_NEW_ZP) * LSTM_Q_L0_C_NEW_SCALE;
        //     printf("l(%d) ig output (%d):%d, %0.6e\n", layer,i, m_ig16[i],dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_cf16[i] - LSTM_Q_L0_C_NEW_ZP) * LSTM_Q_L0_C_NEW_SCALE;
        //     printf("l(%d) cf output (%d): %0.6e\n", layer,i, dequant);
        // }

        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)out_h[i] - LSTM_Q_L0_H_NEW_ZP) * LSTM_Q_L0_H_NEW_SCALE;
        //     printf("l(%d) h_new output (%d): %0.6e\n", layer,i, dequant);
        // }

        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)out_c[i] - LSTM_Q_L0_C_NEW_ZP) * LSTM_Q_L0_C_NEW_SCALE;
        //     printf("l(%d) c_new output (%d): %0.6e\n", layer,i, dequant);
        // }
    }else{
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)in_h[i] - LSTM_Q_L1_H_INPUT_OFFSET) * LSTM_Q_L1_H_S_IN;
        //     printf("l(%d) h (%d): %0.6e\n", layer, i, dequant);
        // }
        // for (int i =0; i < (56 * 4); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_hhat[i] - LSTM_Q_L1_H_HAT_ZP) * LSTM_Q_L1_H_HAT_SCALE;
        //     printf("l(%d) h_hat (%d): %0.6e\n", layer, i, dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)i_gate[i] - SIG_ZP) * SIG_SCALE;
        //     printf("l(%d) i output (%d): %0.6e\n", layer, i, dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_g[i] - TANH_ZP) * TANH_SCALE;
        //     printf("l(%d) g output (%d): %0.6e\n", layer,i, dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_ig16[i] - LSTM_Q_L1_C_NEW_ZP) * LSTM_Q_L1_C_NEW_SCALE;
        //     printf("l(%d) ig output (%d):%d, %0.6e\n", layer,i, m_ig16[i],dequant);
        // }
        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)m_cf16[i] - LSTM_Q_L1_C_NEW_ZP) * LSTM_Q_L1_C_NEW_SCALE;
        //     printf("l(%d) cf output (%d): %0.6e\n", layer,i, dequant);
        // }

        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)out_h[i] - LSTM_Q_L1_H_NEW_ZP) * LSTM_Q_L1_H_NEW_SCALE;
        //     printf("l(%d) h_new output (%d): %0.6e\n", layer,i, dequant);
        // }

        // for (int i =0; i < (56); i++)
        // {
        //     //dequant first
        //     float dequant = ((int32_t)out_c[i] - LSTM_Q_L1_C_NEW_ZP) * LSTM_Q_L1_C_NEW_SCALE;
        //     printf("l(%d) c_new output (%d): %0.6e\n", layer,i, dequant);
        // }
    }
    
    
}

void OptimizedNativeLSTM::run_timestep_q(const int8_t* x_q, int8_t* h_q, int8_t* c_q, int8_t* y_out)
{
    // layer 0: input = external x
    run_cell_step(0, x_q, x_features, LSTM_Q_L0_X_INPUT_OFFSET,
                h_q + 0*H, LSTM_Q_L0_H_INPUT_OFFSET, LSTM_Q_IN_C_ZP,
                c_q + 0*H, c_q + 0*H, h_q + 0*H);
    
    // layer 1: input = h_new[0] (just written into h_q[0])
    run_cell_step(1, h_q + 0*H, H, LSTM_Q_L1_X_INPUT_OFFSET,
                h_q + 1*H, LSTM_Q_L1_H_INPUT_OFFSET, LSTM_Q_L1_C_NEW_ZP,
                c_q + 1*H, c_q + 1*H, h_q + 1*H);

    // output = top layer hidden
    memcpy(y_out, h_q + 1*H, H);
}

void OptimizedNativeLSTM::run_timestep_q_c16(const int8_t* x_q, int8_t* h_q,
                                             int16_t* c_q, int8_t* y_out)
{
    run_cell_step_c16(0, x_q, x_features, LSTM_Q_L0_X_INPUT_OFFSET,
                      h_q + 0*H, LSTM_Q_L0_H_INPUT_OFFSET, LSTM_Q_IN_C_ZP,
                      c_q + 0*H, c_q + 0*H, h_q + 0*H);

    run_cell_step_c16(1, h_q + 0*H, H, LSTM_Q_L1_X_INPUT_OFFSET,
                      h_q + 1*H, LSTM_Q_L1_H_INPUT_OFFSET, LSTM_Q_L1_C_NEW_ZP,
                      c_q + 1*H, c_q + 1*H, h_q + 1*H);
    memcpy(y_out, h_q + 1*H, H);
}
void OptimizedNativeLSTM::set_weights(
        int xf, int hf,
        int8_t* x0w,int32_t* x0b, int8_t* h0w,int32_t* h0b,
        int8_t* x1w,int32_t* x1b, int8_t* h1w,int32_t* h1b
){
    x_features=xf; h_features=hf;
    m_x_l0_weights=x0w; m_x_l0_bias=x0b; m_h_l0_weights=h0w; m_h_l0_bias=h0b;
    m_x_l1_weights=x1w; m_x_l1_bias=x1b; m_h_l1_weights=h1w; m_h_l1_bias=h1b;
}