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

#ifndef CPU_JIT_GEMM_CONVOLUTION_UTILS_HPP
#define CPU_JIT_GEMM_CONVOLUTION_UTILS_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"
#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace jit_gemm_convolution_utils {

    void im2col (jit_gemm_conv_conf_t &jcp, const float *im, float *col);
    void im2col_u8(jit_gemm_conv_conf_t &jcp, const uint8_t *im, uint8_t *col);
    void col2im (jit_gemm_conv_conf_t &jcp, const float *col, float *im);

    void init_conf(jit_gemm_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        bool with_relu = false, float relu_negative_slope = -1.0);

    template <typename src_t>
    status_t prepare_ws_col(jit_gemm_conv_conf_t &jcp, src_t **col,
            const int nthr);
    status_t prepare_ws_wei_reduction(jit_gemm_conv_conf_t &jcp,
            float **wei_reduction, size_t wei_sz, const int nthr);
    template <typename acc_t>
    status_t prepare_ws_acc(jit_gemm_conv_conf_t &jcp, acc_t **acc,
            const int nthr);

    void bwd_weights_balance(int ithr, int nthr,
        int ngroups, int mb, int &ithr_g, int &nthr_g, int &ithr_mb,
            int &nthr_mb);
    void bwd_weights_reduction_par(int ithr, int nthr,
        const jit_gemm_conv_conf_t &jcp, const float *weights_reduce_ws,
            float *weights);
};

}
}
}

#endif
