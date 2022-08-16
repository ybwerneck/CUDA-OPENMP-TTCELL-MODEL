#pragma once
#define stim_state  0 
#define stim_amplitude  -5.20e+01
#define stim_period  1.0e+03
#define stim_start  5.0e+00
#define stim_duration  1.0e+00
#define R  8.3144720e+03
#define T  3.10e+02
#define F  9.64853415e+04
#define Na_o  1.40e+02
#define K_o  5.40e+00
#define P_kna  3.0e-02
#define Ca_o  2.0e+00
#define g_K1  5.4050e+00
#define g_Kr  0.096 //0.134
#define g_Ks  0.245 //0.270
#define g_Na  1.48380e+01
#define g_bna  2.90e-04
#define g_CaL  1.750e-04
#define g_bca  5.920e-04
#define g_to  2.940e-01
#define P_NaK  1.3620e+00
#define K_mk  1.0e+00
#define K_NaCa  1.0e+03
#define gamma  3.50e-01
#define alpha  2.50e+00
#define Km_Nai  8.750e+01
#define Km_Ca  1.380e+00
#define K_sat  1.0e-01
#define g_pCa  8.250e-01
#define K_pCa  5.0e-04
#define g_pK 1.460e-02
#define a_rel  1.64640e-02
#define b_rel  2.50e-01
#define c_rel  8.2320e-03
#define Vmax_up  4.250e-04
#define K_up  2.50e-04
#define V_leak  8.0e-05
#define tau_g  2.0e+00
#define Buf_c  1.50e-01
#define K_buf_c  1.0e-03
#define Buf_sr  1.0e+01
#define K_buf_sr  3.0e-01
#define V_c  1.64040e-02
#define Cm  1.850e-01
#define V_sr  1.0940e-03
#define K_mNa  4.0e+01

#define g_Na  args[tid + 2 * N]
#define atp  args[tid + 4 * N]
#define K_o  args[tid + 3 * N]
#define g_CaL  args[tid + N]
#define g_pCa  1 / (1 + pow((1.4 / atp), 2.6))
#define g_atp  1 / (1 + pow((atp / 0.25), 2.0))





#define V_old_ Y_old_[0]
#define Ca_i_old_ Y_old_[1]
#define Ca_SR_old_ Y_old_[2]
#define Na_i_old_ Y_old_[3]
#define K_i_old_ Y_old_[4]
#define ikatp_old Y_old_[5]

#define Xr1_old_ Y_old_[6]
#define Xr2_old_ Y_old_[7]
#define Xs_old_ Y_old_[8]
#define m_old_ Y_old_[9]
#define h_old_ Y_old_[10]
#define j_old_ Y_old_[11]
#define d_old_ Y_old_[12]
#define f_old_ Y_old_[13]
#define fCa_old_ Y_old_[14]
#define s_old_ Y_old_[15]
#define r_old_ Y_old_[16]
#define g_old_ Y_old_[17]

#define V_f_ rhs[0]
#define Ca_i_f_ rhs[1]
#define Ca_SR_f_ rhs[2]
#define Na_i_f_ rhs[3]
#define K_i_f_ rhs[4]
#define Xr1_f_ rhs[6]
#define Xr2_f_ rhs[7]
#define Xs_f_ rhs[8]
#define m_f_ rhs[9]
#define h_f_ rhs[10]
#define j_f_ rhs[11]
#define d_f_ rhs[12]
#define f_f_ rhs[13]
#define fCa_f_ rhs[14]
#define s_f_ rhs[15]
#define r_f_ rhs[16]
#define g_f_ rhs[17]
#define ikatp_f rhs[5]

#define Xr1_a_ a[0] //45
#define Xr2_a_ a[1] 
#define Xs_a_ a[2]
#define m_a_ a[3] //48
#define h_a_ a[4]
#define j_a_ a[5]
#define d_a_ a[6] //51
#define f_a_ a[7]
#define fCa_a_ a[8]
#define s_a_ a[9] //54
#define r_a_ a[10]
#define g_a_ a[11] //56

#define Xr1_b_ b[0]
#define Xr2_b_ b[1]
#define Xs_b_ b[2]
#define m_b_ b[3]
#define h_b_ b[4]
#define j_b_ b[5]
#define d_b_ b[6]
#define f_b_ b[7]
#define fCa_b_ b[8]
#define s_b_ b[9]
#define r_b_ b[10]
#define g_b_ b[11]

#define calc_i_Stim algs[0] 	 
#define calc_E_Na algs[1] 	 
#define calc_E_K algs[2] 	 
#define calc_E_Ks algs[3] 	 
#define calc_E_Ca algs[4] 	 
#define calc_alpha_K1 algs[5] 	 
#define calc_beta_K1 algs[6] 	 
#define calc_xK1_inf algs[7] 	 
#define calc_i_K1 algs[8] 	 
#define calc_i_Kr algs[9] 	 
#define calc_xr1_inf algs[10] 	 
#define calc_alpha_xr1 algs[11] 	 
#define calc_beta_xr1 algs[12] 	 
#define calc_tau_xr1 algs[13] 	 
#define calc_xr2_inf algs[14] 	 
#define calc_alpha_xr2 algs[15] 	 
#define calc_beta_xr2 algs[16] 	 
#define calc_tau_xr2 algs[17] 	 
#define calc_i_Ks algs[18] 	 
#define calc_xs_inf algs[19] 	 
#define calc_alpha_xs algs[20] 	 
#define calc_beta_xs algs[21] 	 
#define calc_tau_xs algs[22] 	 
#define calc_i_Na algs[23] 	 
#define calc_m_inf algs[24] 	 
#define calc_alpha_m algs[25] 	 
#define calc_beta_m algs[26] 	 
#define calc_tau_m algs[27] 	 
#define calc_h_inf algs[28] 	 
#define calc_alpha_h algs[29] 	 
#define calc_beta_h algs[30] 	 
#define calc_tau_h algs[31] 	 
#define calc_j_inf algs[32] 	 
#define calc_alpha_j algs[33] 	 
#define calc_beta_j algs[34] 	 
#define calc_tau_j algs[35] 	 
#define calc_i_b_Na algs[36] 	 
#define calc_i_CaL algs[37] 	 
#define calc_d_inf algs[38] 	 
#define calc_alpha_d algs[39] 	 
#define calc_beta_d algs[40] 	 
#define calc_gamma_d algs[41] 	 
#define calc_tau_d algs[42] 	 
#define calc_f_inf algs[43] 	 
#define calc_tau_f algs[44] 	 
#define calc_alpha_fCa algs[45] 	 
#define calc_beta_fCa algs[46] 	 
#define calc_gama_fCa algs[47] 	 
#define calc_fCa_inf algs[48] 	 
#define calc_tau_fCa algs[49] 	 
#define calc_d_fCa algs[50] 	 
#define calc_i_b_Ca algs[51] 	 
#define calc_i_to algs[52] 	 
#define calc_s_inf algs[53] 	 
#define calc_tau_s algs[54] 	 
#define calc_r_inf algs[55] 	 
#define calc_tau_r algs[56] 	 
#define calc_i_NaK algs[57] 	 
#define calc_i_NaCa algs[58] 	 
#define calc_i_p_Ca algs[59] 	 
#define calc_i_p_K algs[60] 	 
#define calc_i_rel algs[61] 	 
#define calc_i_up algs[62] 	 
#define calc_i_leak algs[63] 	 
#define calc_g_inf algs[64] 	 
#define calc_d_g algs[65] 	 
#define calc_Ca_i_bufc algs[66] 	 
#define calc_Ca_sr_bufsr algs[67] 	 
#define EPSILON 1e-8
#define MKStart 18
#define MKEnd 18
#define NLStart 0
#define NLEnd 6
#define nStates 18
#define nStates_HH 12
#define nAlgs 68
#define nStates_MKM_max 0
#define HHStart 6




