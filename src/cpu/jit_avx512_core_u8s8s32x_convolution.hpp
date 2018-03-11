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

#ifndef CPU_JIT_AVX512_CORE_U8S8S32X_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_U8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_transpose_src_utils.hpp"
#include "cpu_reducer.hpp"
#include "cpu_barrier.hpp"

#include "jit_avx512_core_u8s8s32x_conv_kernel.hpp"

#if defined(LOAD_SAVE_DATA) && defined(FUSE_CONV)
template<typename dtype>
static void fscanf_data(FILE* fp, dtype* pdata) {
  printf("Error: unkown data type\n");
}

template<>
void fscanf_data<int8_t>(FILE* fp, int8_t* pdata) {
  int32_t tmp32 = 0;
  int res = fscanf(fp, "%d,", &tmp32);
  *pdata = static_cast<int8_t>(tmp32);
}

template<>
void fscanf_data<int32_t>(FILE* fp, int32_t* pdata) {
  int res = fscanf(fp, "%d,", pdata);
}

template<typename dtype>
static void load_x(const char* filename, dtype* pdata, const size_t size) {
  FILE *fp = NULL;\
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: no such file %s\n", filename);
  }
  for (size_t i = 0; i < size; ++i) {
     fscanf_data<dtype>(fp, pdata + i);
  }
  fclose(fp);
}

template<typename dtype>
static void save_nhwc(const char* filename, dtype* pdata, const int bs, const int height,
  const int width, const int channel) {
  FILE *fp = NULL;\
  fp = fopen(filename, "w");
  for (int n = 0; n < bs; ++n) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channel; ++c) {
          fprintf(fp, "%d,", pdata[c + w*channel + h*width*channel + n*height*width*channel]);
        }
        fprintf(fp, "\n");
      }
    }
  }
  fclose(fp);
}
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

static const int case_id = 3;

template <bool with_relu, impl::data_type_t dst_type>
struct _jit_avx512_core_u8s8s32x_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_({})
        {
        }
        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx512_core, ""),
                _jit_avx512_core_u8s8s32x_convolution_fwd_t<with_relu,
                dst_type>);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                    && utils::one_of(this->cdesc_().prop_kind, forward_training,
                               forward_inference)
                    && this->cdesc_().alg_kind == alg_kind::convolution_direct
                    && this->cdesc_().dst_desc.data_type == dst_type
                    && utils::implication(this->with_bias(), utils::one_of(
                            this->cdesc_().bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                    && this->cdesc_().accum_data_type == data_type::s32;
            if (!ok)
                return status::unimplemented;

            return jit_avx512_core_u8s8s32x_fwd_kernel::init_conf(
                    jcp_, this->cdesc_(), this->src_pd_, this->weights_pd_,
                    this->dst_pd_,this->bias_pd_, *this->attr(),
                    with_relu, this->negative_slope());
        }

        jit_conv_conf_t jcp_;
    };

    _jit_avx512_core_u8s8s32x_convolution_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    {
        kernel_ = new jit_avx512_core_u8s8s32x_fwd_kernel(conf_.jcp_,
                    *conf_.attr());

        const int nthreads = omp_get_max_threads();
        ws_per_thread_ = conf_.jcp_.oh * conf_.jcp_.ow * conf_.jcp_.oc_block
                            * conf_.jcp_.nb_oc_blocking;
        ws_ = (acc_data_t *)malloc(
                nthreads * ws_per_thread_ * sizeof(acc_data_t), 64);
#ifdef FUSE_CONV
        // conv acc 1x1
        // acc format (oc/16, ow, 16o)
        ws1x1_per_thread_ = conf_.jcp_.ow * conf_.jcp_.oc1x1;
        ws1x1_ = (acc_data_t *)malloc(
                nthreads * ws1x1_per_thread_ * sizeof(acc_data_t), 64);

        // TODO: move to outside interface
        // wei1x1, out1x1 and scales, allocate here
        // this oc should be the same in "init_conf", which should be load from jcp_
        wei1x1_ = (wei_data_t *)malloc(conf_.jcp_.oc1x1 * conf_.jcp_.oc * sizeof(wei_data_t), 64);
        bia1x1_ = (acc_data_t *)malloc(conf_.jcp_.oc1x1 * sizeof(acc_data_t), 64);  // bias1x1 only support s32 yet
        out1x1_ = (int32_t *)malloc(conf_.jcp_.mb * conf_.jcp_.oh * conf_.jcp_.ow * conf_.jcp_.oc1x1 * sizeof(int32_t), 64);

#ifdef LOAD_SAVE_DATA
        // bia
        std::string filename;
        filename = "bia1x1_";
        filename += std::to_string(case_id);
        filename += ".txt";
        std::cout << "Pre-load " << filename << std::endl;
        load_x<acc_data_t>(filename.c_str(), bia1x1_, conf_.jcp_.oc1x1);
        // wei
        filename = "wei1x1_";
        filename += std::to_string(case_id);
        filename += ".txt";
        std::cout << "Pre-load " << filename << std::endl;
        load_x<wei_data_t>(filename.c_str(), wei1x1_, conf_.jcp_.oc1x1 * conf_.jcp_.oc);
#endif
#endif
    }

    ~_jit_avx512_core_u8s8s32x_convolution_fwd_t() {
        free(ws_);
#ifdef FUSE_CONV

#ifdef LOAD_SAVE_DATA
        std::string filename;
        filename = "fuse1x1_dst_";
        filename += std::to_string(case_id);
        filename += ".txt";
        std::cout << "Save fused 1x1 relu dst " << filename << std::endl;
        save_nhwc<int32_t>(filename.c_str(), out1x1_, conf_.jcp_.mb, conf_.jcp_.oh, conf_.jcp_.ow, conf_.jcp_.oc1x1);
#endif
        free(ws1x1_);
        free(wei1x1_);
        free(bia1x1_);
        free(out1x1_);
#endif
        delete kernel_;
    };

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e)
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_avx512_core_u8s8s32x_fwd_kernel *kernel_;
    size_t ws_per_thread_;
    acc_data_t *ws_;

#ifdef FUSE_CONV
    size_t ws1x1_per_thread_;
    acc_data_t *ws1x1_;
    // tmp here
    wei_data_t *wei1x1_;
    acc_data_t *bia1x1_;
    int32_t *out1x1_;
#endif
};

template <impl::data_type_t dst_type>
using jit_avx512_core_u8s8s32x_convolution_fwd_t =
    _jit_avx512_core_u8s8s32x_convolution_fwd_t<false, dst_type>;

template <impl::data_type_t dst_type>
using jit_avx512_core_u8s8s32x_convolution_relu_t =
    _jit_avx512_core_u8s8s32x_convolution_fwd_t<true, dst_type>;

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
