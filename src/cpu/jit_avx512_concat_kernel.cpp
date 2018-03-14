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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_concat_kernel.hpp"

#define GET_OFF(field) offsetof(jit_concat_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

void jit_avx512_concat_kernel::compute_one_input() {
  Label l_next_block;
  int shift_c = jcp.typesize * jcp.block;
  mov(reg_nb, dword[reg_ptr_nb_ic]);
  mov(reg_ptr_src_i, ptr[reg_ptr_src]);

  L(l_next_block); {
    // load from dst
    vmovups(zmm_src, EVEX_compress_addr(reg_ptr_src_i, 0));

    // relu
    // vmaxps(zmm_src, zmm_zero, zmm_src);

    // save to dst
    vmovups(EVEX_compress_addr(reg_ptr_dst, 0), zmm_src);
    add(reg_ptr_src_i, shift_c);
    add(reg_ptr_dst, shift_c);
    dec(reg_nb);
    cmp(reg_nb, 0);
    jg(l_next_block, T_NEAR);
  }
}

void jit_avx512_concat_kernel::generate()
{
  preamble();

  mov(reg_ptr_src, ptr[param + GET_OFF(src)]);
  mov(reg_ptr_nb_ic, ptr[param + GET_OFF(nb_ic)]);
  mov(reg_ptr_dst, ptr[param + GET_OFF(dst)]);

  vpxord(zmm_zero, zmm_zero, zmm_zero);

  xor_(reg_ninputs, reg_ninputs);
  Label l_next_input;
  L(l_next_input); {
    compute_one_input();
    add(reg_ptr_src, sizeof(void*)); // move 64bits
    add(reg_ptr_nb_ic, sizeof(int));  // move one int
    inc(reg_ninputs);
    cmp(reg_ninputs, jcp.n_inputs);
    jl(l_next_input, T_NEAR);
  }

  postamble();
}

bool jit_avx512_concat_kernel::post_ops_ok(
        jit_concat_conf_t &jcp, const primitive_attr_t &attr)
{
  const auto &p = attr.post_ops_;

  auto is_relu = [&](int idx) {
      return p.entry_[idx].kind == primitive_kind::eltwise
          && p.entry_[idx].eltwise.scale == 1.
          && p.entry_[idx].eltwise.alg == alg_kind::eltwise_relu
          && p.entry_[idx].eltwise.alpha == 0.;
  };

  switch (p.len_) {
    case 0: return true;
    case 1: return true && jcp.with_relu;
    default: return false;
  }

  return false;
}

status_t jit_avx512_concat_kernel::init_conf(jit_concat_conf_t &jcp,
        nstl::vector<cpu_memory_t::pd_t> &src_pds,
        cpu_memory_t::pd_t &dst_pd,
        const primitive_attr_t &attr,
        bool with_relu,
        float relu_negative_slope) {
  const memory_desc_wrapper dst_d(&dst_pd);

  jcp = zero<decltype(jcp)>();
  jcp.bs = dst_d.dims()[0];
  jcp.oc = dst_d.dims()[1];
  jcp.h = dst_d.dims()[2];
  jcp.w = dst_d.dims()[3];
  jcp.n_inputs = src_pds.size();
  jcp.with_relu = with_relu;

  if (!post_ops_ok(jcp, attr)) {
    return status::unimplemented;
  }

  if (jcp.n_inputs > 16) {
    return status::unimplemented;
  }

  jcp.dtype = dst_d.data_type();
  jcp.typesize = types::data_type_size(dst_d.data_type());
  jcp.block = 64 / jcp.typesize;
  for (int i = 0; i < jcp.n_inputs; ++i) {
    const memory_desc_wrapper src_d(&src_pds[i]);
    if (src_d.dims()[1] % jcp.block != 0) {  
      // all input channels should be dividable
      // input is s32 or float s32, channels should be 16x
      // input is s8 or u8, then 64x
      return status::unimplemented;
    }
  }

  return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
