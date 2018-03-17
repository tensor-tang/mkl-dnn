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

#ifndef JIT_AVX512_CONCAT_KERNEL_HPP
#define JIT_AVX512_CONCAT_KERNEL_HPP

#include "c_types_map.hpp"
#include "cpu_memory.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_concat_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_concat_kernel_t)

    enum { STATE_FIRST_DST_LOAD = 0x1U };

    jit_avx512_concat_kernel(jit_concat_conf_t ajcp,
            const primitive_attr_t &attr) : jcp(ajcp), attr_(attr)
    {
        generate();
        jit_ker = (void (*)(jit_concat_call_s *))getCode();
    }
    static status_t init_conf(jit_concat_conf_t &jcp,
            nstl::vector<cpu_memory_t::pd_t> &src_pds,
            cpu_memory_t::pd_t &dst_pd,
            const primitive_attr_t &attr,
            bool with_relu = false,
            float relu_negative_slope = 0.);

    static bool post_ops_ok(jit_concat_conf_t &jcp,
                const primitive_attr_t &attr);

    jit_concat_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_concat_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using zmm_t = const Xbyak::Zmm;
    using xmm_t = const Xbyak::Xmm;

    reg64_t param         = abi_param1;
    reg64_t reg_ptr_src   = r8;
    reg64_t reg_ptr_nb_ic = r9;
    reg64_t reg_ptr_dst   = r10;
    reg64_t reg_ptr_src_i = r11;
    reg64_t reg_ninputs   = r12;
    reg32_t reg_nb        = r15d;

    xmm_t xmm_src  = xmm_t(30);
    zmm_t zmm_src  = zmm_t(30);
    zmm_t zmm_zero = zmm_t(31);

    void compute_one_input();
    void generate();

/*
    reg64_t reg_ker = r9;
    reg64_t aux_reg_inp = r11;
    reg64_t reg_ptr_sum_scale = r11;
    reg64_t aux_reg_ker = r12;
    reg64_t reg_acc_s32 = r13;
    reg64_t reg_scratch = r14;
    reg64_t reg_kj   = rax;
    reg64_t reg_ptr_scales = rax;
    reg64_t reg_oi   = rbx;
    reg64_t reg_bias = rdx;
    reg64_t reg_kh   = abi_not_param1;
    reg64_t reg_channel = r15;
    reg64_t reg_tmp = rbp;
    reg64_t imm_addr64 = r15;

    zmm_t zmm_tmp = zmm_t(28);
    zmm_t zmm_one = zmm_t(29);
    zmm_t zmm_scales = zmm_t(30);
    zmm_t zmm_bcast = zmm_t(30);
    zmm_t zmm_wei = zmm_t(31);

    bool maybe_relu(int position);
    void prepare_output(int ur_w);
    void store_output(int ur_w);
    void compute_loop(int ur_w, int pad_l, int pad_r);
    */

};

}
}
}

#endif
