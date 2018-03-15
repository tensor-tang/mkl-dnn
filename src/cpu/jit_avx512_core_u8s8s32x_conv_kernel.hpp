/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_JIT_AVX512_CORE_U8S8S32X_CONV_KERNEL_HPP
#define CPU_JIT_AVX512_CORE_U8S8S32X_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "cpu_memory.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_core_u8s8s32x_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8s8s32x_conv_fwd_ker_t)

    enum { STATE_FIRST_DST_LOAD = 0x1U };

    jit_avx512_core_u8s8s32x_fwd_kernel(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr) : jcp(ajcp), attr_(attr)
    {
        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode();
    }
    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd,
            cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd,
            const primitive_attr_t &attr,
            bool with_relu = false,
            float relu_negative_slope = 0.);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
#ifdef FUSE_CONV
    using reg32_t = const Xbyak::Reg32;
#endif
    using zmm_t = const Xbyak::Zmm;
    using xmm_t = const Xbyak::Xmm;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };

    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
#ifdef FUSE_CONV    
    reg64_t reg_ptr_out1x1 = r10;  // replace gre out
#else
    reg64_t reg_out = r10;  // when fuse 1x1, do not need 3x3 out
#endif
    reg64_t aux_reg_inp = r11;
    reg64_t reg_ptr_sum_scale = r11;
    reg64_t aux_reg_ker = r12;
    reg64_t reg_acc_s32 = r13;

#ifdef FUSE_CONV
    reg64_t reg_scratch = r15;  // the r14 is used for acc1x1 for whole life
#else
    reg64_t reg_scratch = r14;
#endif
    reg64_t reg_kj   = rax;
    reg64_t reg_ptr_scales = rax;
    reg64_t reg_oi   = rbx;
    reg64_t reg_bias = rdx;
    reg64_t reg_kh   = abi_not_param1;
    reg64_t param    = abi_param1;
    reg64_t reg_channel = r15;
    reg64_t reg_tmp = rbp;
    reg64_t imm_addr64 = r15;

    zmm_t zmm_tmp = zmm_t(28);
    zmm_t zmm_one = zmm_t(29);
    zmm_t zmm_scales = zmm_t(30);
    zmm_t zmm_bcast = zmm_t(30);
    zmm_t zmm_zero = zmm_t(31);
    zmm_t zmm_wei = zmm_t(31);

#ifdef FUSE_CONV
    // for conv 1x1
    reg64_t reg_ptr_acc1x1 = r14;  // use r14 which should always be used in kernel
    reg64_t aux_reg_ptr_acc1x1 = r11; // this is a tmp_reg for acc1x1 add offset // use aux_reg_ker, used only in compute_loop

    reg64_t reg_ptr_bia1x1 = rdx;  // use reg_bias, can use channel reg either i think
    reg64_t reg_ptr_wei1x1 = r11;  // used reg_ptr_sum_scale reg
    reg64_t reg_ocb3x3 = r15;  // use reg_channel
    reg64_t reg_ptr_scales1x1 = rax;  // use reg_ptr_scales
    reg32_t reg_1x1_src_4u8 = r15d;  // use reg_channel reg

    zmm_t zmm_1x1_src_bcast_u8 = zmm_t(31);  // use use zero zmm
    zmm_t zmm_1x1_wei = zmm_t(30);  // use zmm_bcast zmm 
#endif
    zmm_t zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return zmm_t(idx);
    }
    xmm_t xmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xmm_t(idx);
    }
    zmm_t zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return zmm_t(idx);
    }
#ifdef FUSE_CONV
    // 1x1 acc use 3x3 input. size is ur_w * zmm
    // range: (0~ur_w, jcp.nb_oc_blocking)
    #define zmm_1x1out zmm_inp
    
    // just change zmm to xmm. should keep the same with zmm_inp
    xmm_t xmm_1x1out(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return xmm_t(idx);
    }
#endif

    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0, (pad_l - ki + jcp.stride_w - 1) / jcp.stride_w);
    }
    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w - nstl::max(0,
            (ki + pad_r - (jcp.kw - 1) + jcp.stride_w - 1) / jcp.stride_w);
    }
    bool maybe_relu(int position);
    void prepare_output(int ur_w);
    void store_output(int ur_w);
#ifdef FUSE_CONV
    void prepare_1x1output(int ur_w);
    void store_1x1output(int ur_w, int ocb1x1);
#endif
    void compute_loop(int ur_w, int pad_l, int pad_r);
    void generate();
};

}
}
}

#endif
