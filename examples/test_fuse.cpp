/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#include <iostream>
#include <numeric>
#include <string>
#include "mkldnn.hpp"

#include <sys/time.h>
#include <stdint.h>


using namespace mkldnn;

static int burning_iter = 20;
static int iter = 10;

void test_reorder() {
  auto cpu_engine = engine(engine::cpu, 0);
  memory::dims wgt_dims = {32, 8, 2, 2};  // oihw
  std::vector<float> src(std::accumulate(wgt_dims.begin(),
      wgt_dims.end(), 1, std::multiplies<uint32_t>()));
  std::vector<float> dst(std::accumulate(wgt_dims.begin(),
      wgt_dims.end(), 1, std::multiplies<uint32_t>()));

  auto src_mem = memory({{{wgt_dims}, memory::data_type::f32,
    memory::format::oihw}, cpu_engine}, src.data());
  auto dst_mem = memory({{{wgt_dims}, memory::data_type::f32,
    memory::format::OhIw16o4i}, cpu_engine}, dst.data());

#pragma omp parallel for collapse(4)
  for (auto o = 0; o < wgt_dims[0]; ++o) {
    for (auto i = 0; i < wgt_dims[1]; ++i) {
      for (auto h = 0; h < wgt_dims[2]; ++h) {
        for (auto w = 0; w < wgt_dims[3]; ++w) {
          auto ele = wgt_dims[3] * h + w;
          ele += i * wgt_dims[3] * wgt_dims[2];
          ele += o * wgt_dims[3] * wgt_dims[2] *  wgt_dims[1]; 
          src[ele] = ele;
        }
      }
    }
  }

  auto print_data = [&](float* p) {
    for (auto o = 0; o < wgt_dims[0]; ++o) {
      for (auto i = 0; i < wgt_dims[1]; ++i) {
        for (auto h = 0; h < wgt_dims[2]; ++h) {
          for (auto w = 0; w < wgt_dims[3]; ++w) {
            auto ele = wgt_dims[3] * h + w;
            ele += i * wgt_dims[3] * wgt_dims[2];
            ele += o * wgt_dims[3] * wgt_dims[2] *  wgt_dims[1]; 
            std::cout << p[ele] << ",";
          }
        }
        std::cout << std::endl;
      }
    }
  };

  print_data(src.data());

  auto r = reorder(src_mem, dst_mem);
  stream(stream::kind::eager).submit({r}).wait();

  std::cout << "after reorder-----------------------------------" << std::endl;
  print_data(dst.data());
}

template <typename data_t> struct data_traits { };
template <> struct data_traits<float> {
    static const auto data_type = mkldnn::memory::data_type::f32;
};
template <> struct data_traits<uint8_t> {
    static const auto data_type = mkldnn::memory::data_type::u8;
};
template <> struct data_traits<int8_t> {
    static const auto data_type = mkldnn::memory::data_type::s8;
};
template <> struct data_traits<int16_t> {
    static const auto data_type = mkldnn::memory::data_type::s16;
};
template <> struct data_traits<int32_t> {
    static const auto data_type = mkldnn::memory::data_type::s32;
};

struct test_convolution_sizes_t {
    test_convolution_sizes_t(
        int mb,
        int ng,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb;
    int ng;
    int ic, ih, iw;
    int oc, oh, ow;
    int kh, kw;
    int padh, padw;
    int strh, strw;
    int dilh, dilw;
};

void test_conv(bool with_fuse = true) {
    auto eng = engine(engine::cpu, 0);

    // conv desc
    test_convolution_sizes_t cd(
      2, 1,  // bs, gp
      32, 258, 258,  // ic, ih, iw
      64, 256, 256,  // oc, oh, ow
      3, 3,  // kh, kw
      0, 0,  // ph, pw
      1, 1   // sh, sw
      );

    using u8 = uint8_t;
    using s8 = int8_t;
    using s32 = int32_t;

    using data_t_src = u8;
    using data_t_dst = s32;
    using data_t_wei = s8;
    
    memory::data_type data_type_src = data_traits<data_t_src>::data_type;
    memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
    memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;
    
    memory::format src_format = memory::format::any;
    memory::format weights_format = memory::format::any;
    memory::format bias_format = memory::format::x;
    memory::format dst_format = memory::format::any;
    
    auto aprop_kind = prop_kind::forward_inference;
    algorithm aalgorithm = algorithm::convolution_direct;
    bool with_bias = bias_format != memory::format::format_undef;
    
    auto c_src_desc = memory::desc({ cd.mb, cd.ic, cd.ih, cd.iw }, data_type_src, src_format);
    auto c_weights_desc = cd.ng > 1 ?
            memory::desc({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                    data_type_wei, weights_format) :
            memory::desc({ cd.oc, cd.ic, cd.kh, cd.kw },
                    data_type_wei,weights_format);
    auto c_dst_desc = memory::desc({ cd.mb, cd.oc, cd.oh, cd.ow }, data_type_dst, dst_format);
    auto c_bias_desc = with_bias ?
            memory::desc({ cd.oc }, data_type_dst, bias_format) :
            memory::desc({}, data_type_dst, bias_format);

    std::vector<int> padR = { cd.padh, cd.padw };
    for (int i = 0; i < 2; ++i) {
        if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR[0])
            / cd.strh + 1 != cd.oh)
            ++padR[0];
        if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR[1])
            / cd.strw + 1 != cd.ow)
            ++padR[1];
    }

    auto conv_desc = with_bias ?
        convolution_forward::desc(aprop_kind, aalgorithm,
            c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
            { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
            { cd.padh, cd.padw }, padR, padding_kind::zero) :
        convolution_forward::desc(aprop_kind, aalgorithm,
            c_src_desc, c_weights_desc, c_dst_desc,
            { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
            { cd.padh, cd.padw }, padR, padding_kind::zero);
    // attribute
    auto rmode = round_mode::round_nearest;
    primitive_attr mkl_attr = mkldnn::primitive_attr();
    mkl_attr.set_int_output_round_mode(rmode);
    const int count = 1;  // number of scales
    const int mask = 0;  // multi-channel 1
    const float scale = 0.3f;
    std::vector<float> s(count, scale);
    mkl_attr.set_output_scales(mask, s);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, mkl_attr, eng);
    
    std::unique_ptr<memory> src, wgt, dst, bis;
    src.reset(new memory(conv_pd.src_primitive_desc()));
    wgt.reset(new memory(conv_pd.weights_primitive_desc()));
    dst.reset(new memory(conv_pd.dst_primitive_desc()));
    if (with_bias) {
      bis.reset(new memory(conv_pd.bias_primitive_desc()));
    }

    std::vector<primitive> pipeline;
    std::unique_ptr<primitive> fwd;
    std::unique_ptr<primitive> fwd_relu;

    float negative_slope = 0.f;
    if (with_fuse) {
      auto conv_relu_desc =
          convolution_relu_forward::desc(conv_desc, negative_slope);
      auto conv_primitive_desc =
          convolution_relu_forward::primitive_desc(conv_relu_desc, eng);

      if (with_bias) {
        fwd.reset(new convolution_relu_forward(conv_primitive_desc, *src,
                          *wgt, *bis, *dst));
      } else {
        fwd.reset(new convolution_relu_forward(conv_primitive_desc, *src,
                          *wgt, *dst));
      }
      pipeline.push_back(*fwd);
    } else {  // without fusion
      if (with_bias) {
        fwd.reset(new convolution_forward(conv_pd, *src,
                          *wgt, *bis, *dst));
      } else {
        fwd.reset(new convolution_forward(conv_pd, *src,
                          *wgt, *dst));
      }
      pipeline.push_back(*fwd);

      // relu
      auto relu_desc = eltwise_forward::desc(aprop_kind,
                                   algorithm::eltwise_relu,
                                   conv_pd.dst_primitive_desc().desc(),
                                   0.f,
                                   0.f);
      auto relu_pd = eltwise_forward::primitive_desc(relu_desc, eng);
      fwd_relu.reset(new eltwise_forward(relu_pd, *dst, *dst));

      pipeline.push_back(*fwd_relu);

    }


    for (auto i = 0; i < burning_iter; ++i) {
      stream(stream::kind::eager).submit(pipeline).wait();
    }

    auto get_current_ms = []() -> double {
      struct timeval time;
      gettimeofday(&time, NULL);
      return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
    };

    auto t_start = get_current_ms();
    for (auto i = 0; i < iter; ++i) {
      stream(stream::kind::eager).submit(pipeline).wait();
    }
    auto t_stop = get_current_ms();

    if (with_fuse) {
      std::cout << "Fused ";
    } else {
      std::cout << "No Fused ";
    }

#ifdef ENABLE_VNNI
    std::cout << "with VNNI ";
#else
    std::cout << "without VNNI ";
#endif

    std::cout << "conv relu avg time: " << (t_stop - t_start) / (double) iter << " ms" << std::endl;
    
}

// usage ./test_fuse with_fuse buring_iter valid_iter
int main(int argc, char **argv) {
    std::cout << argv[0] << std::endl;
    bool with_fuse = true;
    if (argc >= 2) {
      std::string in(argv[1]);
      assert(in == "0" || in == "1");
      with_fuse = static_cast<bool>(std::stoi(in));
    }
    if (argc >= 3) {
      std::string in(argv[2]);
      burning_iter = std::stoi(in);
      assert(burning_iter > 0 && burning_iter <= 5000);
    }
    if (argc >= 4) {
      std::string in(argv[3]);
      iter = std::stoi(in);
      assert(iter > 0 && iter <= 5000);
    }
    try {
        test_conv(with_fuse);
        // test_reorder();
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
