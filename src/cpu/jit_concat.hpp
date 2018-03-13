/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef JIT_CONCAT_HPP
#define JIT_CONCAT_HPP

#include "cpu_concat.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_avx512_concat_kernel.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct jit_concat_t: public cpu_primitive_t {
  using cpu_memory_pd_t = cpu_memory_t::pd_t;
  typedef typename prec_traits<data_type>::type data_t;

  struct pd_t: public cpu_concat_pd_t {
      pd_t(const memory_desc_t *output_d, int n, int concat_dim,
              const cpu_memory_pd_t **input_pds, const primitive_attr_t *attr)
          : cpu_concat_pd_t(output_d, n, concat_dim, input_pds, attr)
          , jcp_({})
      {}

      DECLARE_CPU_CONCAT_PD_T("jit:any", jit_concat_t);

      virtual status_t init() override {
          using namespace mkldnn::impl::memory_format;

          bool ok = true
              && cpu_concat_pd_t::init() == success
              && concat_dim_ == 1;

          for (size_t i = 0; i < src_pds_.size(); ++i) {
              const memory_desc_wrapper src_d(&src_pds_[i]);
              const memory_desc_wrapper img_d(&src_image_pds_[i]);
              ok = ok
                  && utils::everyone_is(data_type, src_d.data_type(),
                          img_d.data_type())
                  // only support nhwc yet
                  && utils::everyone_is(src_d.format(), img_d.format(), nhwc);
          }
          if (!ok)
              return status::unimplemented;

          return jit_avx512_concat_kernel::init_conf(
                  jcp_, this->src_pds_, this->dst_pd_, *this->attr(),
                  false /*with _relu*/,
                  0. /*negative_slope*/);
      }
      jit_concat_conf_t jcp_;
  };

  jit_concat_t(const pd_t *conf, const input_vector &inputs,
          const output_vector &outputs)
      : cpu_primitive_t(&conf_, inputs, outputs), conf_(*conf) {

        kernel_ = new jit_avx512_concat_kernel(
            conf_.jcp_/*this jcp_ pass to kernel*/, *conf_.attr());

        const int num_srcs = conf_.n_inputs();
        src_ = (const data_t **)malloc(num_srcs*sizeof(data_t *), 64);
        src_with_offset_ = (const data_t **)malloc(num_srcs*sizeof(data_t *), 64);
        ic_ = (int *)malloc(num_srcs*sizeof(int), 64);
      }

  ~jit_concat_t() {
      free(src_);
      free(src_with_offset_);
      free(ic_);
      delete kernel_;
  }

  virtual void execute(event_t *e) {
      execute_forward();
      e->set_state(event_t::ready);
  }

private:
  void execute_forward();
  pd_t conf_;
  jit_avx512_concat_kernel *kernel_;

  const data_t **src_;
  const data_t **src_with_offset_;
  int *ic_;
};

}
}
}
#endif
