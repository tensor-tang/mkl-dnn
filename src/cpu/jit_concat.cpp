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

#include "mkldnn_types.h"
#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;


template <data_type_t data_type>
void jit_concat_t<data_type>::execute_forward() {
  const auto &jcp = kernel_->jcp;
  // update srcs data
  for (int i = 0; i < jcp.n_inputs; ++i) {
    const memory_desc_wrapper src_d(conf_.src_pd(i));
    src_[i] = reinterpret_cast<const data_t *>(this->input_memory(i));
  }

  auto dst = reinterpret_cast<data_t *>(this->memory());
  const int work_amount = jcp.bs  * jcp.h * jcp.w;
  const int max = omp_get_max_threads();

  if (work_amount < max) {
#   pragma omp parallel for schedule(static) collapse(1)
    for (int iwork = 0; iwork < max; ++iwork) {
      int n{0}, h{0}, w{0};
      nd_iterator_init(iwork, n, jcp.bs, h, jcp.h, w, jcp.w);
      int nhw = n*(jcp.h*jcp.w) + h*(jcp.w) + w;
      auto srcs = src_with_offset_ + iwork * jcp.n_inputs;
      for (int i = 0; i < jcp.n_inputs; ++i) {
        srcs[i] = src_[i] + (nhw*ic_[i]);
      }
      jit_concat_call_s p = { 0 };
      p.src = reinterpret_cast<const void **>(srcs);
      p.nb_ic = reinterpret_cast<const int *>(nb_ic_);
      p.dst = reinterpret_cast<void *>(dst + nhw * jcp.oc);
      kernel_->jit_ker(&p);
    }
  } else {
    // if work amount > max omp threads, need balance
#   pragma omp parallel
    {
      int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();
      int start{0}, end{0};
      balance211(work_amount, nthr, ithr, start, end);
      int n{0}, h{0}, w{0};
      nd_iterator_init(start, n, jcp.bs, h, jcp.h, w, jcp.w);
      jit_concat_call_s p = { 0 };
      auto srcs = src_with_offset_ + ithr * jcp.n_inputs;
      for (int iwork = start; iwork < end; ++iwork) {
        int nhw = n*(jcp.h*jcp.w) + h*(jcp.w) + w;
        for (int i = 0; i < jcp.n_inputs; ++i) {
          srcs[i] = src_[i] + (nhw*ic_[i]);
        }
        jit_concat_call_s p = { 0 };
        p.src = reinterpret_cast<const void **>(srcs);
        p.nb_ic = reinterpret_cast<const int *>(nb_ic_);
        p.dst = reinterpret_cast<void *>(dst + nhw * jcp.oc);
        kernel_->jit_ker(&p);
        nd_iterator_step(n, jcp.bs, h, jcp.h, w, jcp.w);
      }
    }
  }
}

template struct jit_concat_t<data_type::f32>;
template struct jit_concat_t<data_type::u8>;
template struct jit_concat_t<data_type::s8>;
template struct jit_concat_t<data_type::s32>;

}
}
}
