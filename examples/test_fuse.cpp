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
static mkldnn::engine eng = mkldnn::engine(engine::cpu, 0);

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

using u8 = uint8_t;
using s8 = int8_t;
using s32 = int32_t;

std::unique_ptr<convolution_forward::desc> get_conv_desc(
    const test_convolution_sizes_t& cd, bool with_bias, memory::data_type data_type_dst = data_traits<s32>::data_type) {
  // dtype
  // src: u8
  // wei: s8
  // bia: s32
  // dst: s32 / u8, default s32
  using data_t_src = u8;
  using data_t_wei = s8;
  using data_t_bia = s32;

  memory::data_type data_type_src = data_traits<data_t_src>::data_type;
  memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;
  memory::data_type data_type_bia = data_traits<data_t_bia>::data_type;

  memory::format src_format = memory::format::any;
  memory::format wei_format = memory::format::any;
  memory::format bia_format = memory::format::x;
  memory::format dst_format = memory::format::any;
    
  auto aprop_kind = prop_kind::forward_inference;
  algorithm aalgorithm = algorithm::convolution_direct;
  auto c_src_desc = memory::desc({ cd.mb, cd.ic, cd.ih, cd.iw }, data_type_src, src_format);
  auto c_weights_desc = cd.ng > 1 ?
            memory::desc({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                    data_type_wei, wei_format) :
            memory::desc({ cd.oc, cd.ic, cd.kh, cd.kw },
                    data_type_wei,wei_format);
  auto c_bias_desc = with_bias ?
          memory::desc({ cd.oc }, data_type_bia, bia_format) :
          memory::desc({}, data_type_bia, bia_format);
  auto c_dst_desc = memory::desc({ cd.mb, cd.oc, cd.oh, cd.ow }, data_type_dst, dst_format);

  std::vector<int> padR = { cd.padh, cd.padw };
  for (int i = 0; i < 2; ++i) {
      if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR[0])
          / cd.strh + 1 != cd.oh)
          ++padR[0];
      if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR[1])
          / cd.strw + 1 != cd.ow)
          ++padR[1];
  }
  
  if (with_bias) {
    return std::unique_ptr<convolution_forward::desc>(
        new convolution_forward::desc(aprop_kind, aalgorithm,
              c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
              { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
              { cd.padh, cd.padw }, padR, padding_kind::zero));
  } else {
    return std::unique_ptr<convolution_forward::desc>(
        new convolution_forward::desc(aprop_kind, aalgorithm,
              c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
              { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
              { cd.padh, cd.padw }, padR, padding_kind::zero));
  }
}

std::unique_ptr<convolution_forward::primitive_desc> get_conv_pd(
    const std::unique_ptr<convolution_forward::desc>& conv_desc, bool with_relu = false, float negative_slope = 0.f) {
  // attribute
  auto rmode = round_mode::round_nearest;
  primitive_attr mkl_attr = mkldnn::primitive_attr();
  mkl_attr.set_int_output_round_mode(rmode);
  const int count = 1;  // number of scales
  const int mask = 0;  // multi-channel 1
  const float scale = 0.3f;
  std::vector<float> s(count, scale);
  mkl_attr.set_output_scales(mask, s);

  if (with_relu) {
    post_ops ops;
    ops.append_eltwise(1.0f, algorithm::eltwise_relu, negative_slope, 0.f);
    mkl_attr.set_post_ops(ops);
  }
   
  return std::unique_ptr<convolution_forward::primitive_desc>(
      new convolution_forward::primitive_desc(*conv_desc, mkl_attr, eng));
}

std::unique_ptr<eltwise_forward::primitive_desc> get_relu_pd(const memory::desc md) {
  auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_relu, md, 0.f, 0.f);
  return std::unique_ptr<eltwise_forward::primitive_desc>(
      new eltwise_forward::primitive_desc(relu_desc, eng));
}

std::unique_ptr<convolution_relu_forward::primitive_desc> get_conv_relu_pd(
    const std::unique_ptr<convolution_forward::desc>& conv_desc, float negative_slope = 0.f) {
  auto conv_relu_desc = convolution_relu_forward::desc(*conv_desc, negative_slope);
  return std::unique_ptr<convolution_relu_forward::primitive_desc>(
      new convolution_relu_forward::primitive_desc(conv_relu_desc, eng));
}

void test_conv(const test_convolution_sizes_t& cd, bool with_fuse = true, bool with_bias = true) {
    std::vector<primitive> pipeline;
    std::unique_ptr<primitive> fwd;
    std::unique_ptr<primitive> fwd_relu;
    std::unique_ptr<convolution_forward::desc> conv_desc;
    std::unique_ptr<convolution_forward::primitive_desc> conv_pd;
    std::unique_ptr<eltwise_forward::primitive_desc> relu_pd;
    std::unique_ptr<convolution_relu_forward::primitive_desc> conv_relu_pd;
    std::unique_ptr<memory> src, wgt, dst, bia;

    conv_desc = get_conv_desc(cd, with_bias);
    conv_pd = get_conv_pd(conv_desc);
    src.reset(new memory(conv_pd->src_primitive_desc()));
    wgt.reset(new memory(conv_pd->weights_primitive_desc()));
    dst.reset(new memory(conv_pd->dst_primitive_desc()));
    if (with_bias) {
      bia.reset(new memory(conv_pd->bias_primitive_desc()));
    }

    if (with_fuse) {
      conv_relu_pd = get_conv_relu_pd(conv_desc);
      if (with_bias) {
        fwd.reset(new convolution_relu_forward(*conv_relu_pd, *src, *wgt, *bia, *dst));
      } else {
        fwd.reset(new convolution_relu_forward(*conv_relu_pd, *src, *wgt, *dst));
      }
      pipeline.push_back(*fwd);
    } else {  // without fusion
      if (with_bias) {
        fwd.reset(new convolution_forward(*conv_pd, *src, *wgt, *bia, *dst));
      } else {
        fwd.reset(new convolution_forward(*conv_pd, *src, *wgt, *dst));
      }
      pipeline.push_back(*fwd);

      // add relu
      relu_pd = get_relu_pd(conv_pd->dst_primitive_desc().desc());
      fwd_relu.reset(new eltwise_forward(*relu_pd, *dst, *dst));
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

    std::cout << "Conv Relu ";
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

    std::cout << "avg time: " << (t_stop - t_start) / (double) iter << " ms" << std::endl;
}


// test conv3x3_relu + conv1x1_relu
// u8 (wei:s8, bia:s32) => relu => u8, (wei:s8, bia:s32) => s32
void test_conv3x3_1x1(test_convolution_sizes_t cd, bool with_bias = true) {
    std::unique_ptr<primitive> fwd3x3, fwd1x1;
    std::vector<primitive> pp3x3, pp1x1;  // pipeline
    std::unique_ptr<memory> src, wgt3x3, dst3x3, bia3x3, wgt1x1, bia1x1, dst;

    std::unique_ptr<convolution_forward::desc> desc3x3, desc1x1;
    std::unique_ptr<convolution_forward::primitive_desc> pdconv3x3, pdconv1x1;
    std::unique_ptr<convolution_relu_forward::primitive_desc> pdconvrelu3x3, pdconvrelu1x1;

    desc3x3 = get_conv_desc(cd, with_bias, data_traits<u8>::data_type);
    cd.ic = cd.oc; cd.ih = cd.oh; cd.iw = cd.ow;
    cd.oc = 96;
    cd.kh = 1; cd.kw = 1;
    desc1x1 = get_conv_desc(cd, with_bias, data_traits<s32>::data_type);

    pdconv3x3 = get_conv_pd(desc3x3);
    pdconv1x1 = get_conv_pd(desc1x1);
    pdconvrelu3x3 = get_conv_relu_pd(desc3x3);
    pdconvrelu1x1 = get_conv_relu_pd(desc1x1);

    src.reset(new memory(pdconv3x3->src_primitive_desc()));
    wgt3x3.reset(new memory(pdconv3x3->weights_primitive_desc()));
    wgt1x1.reset(new memory(pdconv1x1->weights_primitive_desc()));
    dst3x3.reset(new memory(pdconv3x3->dst_primitive_desc()));
    dst.reset(new memory(pdconv1x1->dst_primitive_desc()));
    if (with_bias) {
      bia3x3.reset(new memory(pdconv3x3->bias_primitive_desc()));
      bia1x1.reset(new memory(pdconv1x1->bias_primitive_desc()));
    }

    if (with_bias) {
      fwd3x3.reset(new convolution_relu_forward(*pdconvrelu3x3, *src, *wgt3x3, *bia3x3, *dst3x3));
    } else {
      fwd3x3.reset(new convolution_relu_forward(*pdconvrelu3x3, *src, *wgt3x3, *dst3x3));
    }
    pp3x3.push_back(*fwd3x3);

    if (with_bias) {
      fwd1x1.reset(new convolution_relu_forward(*pdconvrelu1x1, *dst3x3, *wgt1x1, *bia1x1, *dst));
    } else {
      fwd1x1.reset(new convolution_relu_forward(*pdconvrelu1x1, *dst3x3, *wgt1x1, *dst));
    }
    pp1x1.push_back(*fwd1x1);

    for (auto i = 0; i < burning_iter; ++i) {
      stream(stream::kind::eager).submit(pp3x3).wait();
      stream(stream::kind::eager).submit(pp1x1).wait();
    }

    auto get_current_ms = []() -> double {
      struct timeval time;
      gettimeofday(&time, NULL);
      return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
    };

    auto t_start = get_current_ms();
    double sum3x3 = 0;
    double sum1x1 = 0;
    
    for (auto i = 0; i < iter; ++i) {
      auto s1 = get_current_ms();
      stream(stream::kind::eager).submit(pp3x3).wait();
      auto s2 = get_current_ms();
      stream(stream::kind::eager).submit(pp1x1).wait();
      auto s3 = get_current_ms();
      sum3x3 += (s2 - s1);
      sum1x1 += (s3 - s2);
    }
    auto t_stop = get_current_ms();

    std::cout << "Conv3x3_Relu + Conv1x1_Relu ";
    if (with_bias) {
      std::cout << "with bias ";
    } else {
      std::cout << "without bias ";
    }
#ifdef ENABLE_VNNI
    std::cout << "with VNNI ";
#else
    std::cout << "without VNNI ";
#endif
    std::cout << "avg time ("
    << sum3x3 / (double)iter << " + " << sum1x1 / (double)iter << "): "
    << (t_stop - t_start) / (double) iter << " ms" << std::endl;
}

static void usage() {
  std::cout << "./test_fuse function(3x3/1x1/reorder) buring_iter(1~1000) valid_iter(1~1000) test_case_idx(0/1/2) option(0/1)"<< std::endl;
  std::cout << "function: \n 3x3: conv_relu or conv+relu \n 1x1: conv3x3_relu+conv1x1_relu" << std::endl;
  std::cout << "buring_iter: \n defalut is 10 \nvalid_iter: default is 10" << std::endl;
  std::cout << "test_case_idx: \n default is 2 \n" << std::endl;
  std::cout << "option: in 3x3 it's with_fuse; in 1x1 it's with_bias" << std::endl;
}

#define CHECK(x) \
  if (!(x)) {  \
    std::cout << "Check failed!" << std::endl;\
    usage(); assert(false);\
  }

int main(int argc, char **argv) {
    // std::cout << argv[0] << std::endl;
    std::string func_option("3x3");
    bool with_fuse = true, with_bias = true;
    int test_case_idx = 2;
    if (argc >= 2) {
      std::string in(argv[1]);
      func_option = in;
      CHECK(in == "3x3" || in == "1x1" || in == "reorder");
    }
    if (func_option == "reorder" && argc >= 3) {
      std::cout << "Warnning: \" " << argv[2] <<
        " \" and after inputs are invalid when test_reorder." << std::endl;
    }
    if (argc >= 3) {
      std::string in(argv[2]);
      burning_iter = std::stoi(in);
      CHECK(burning_iter > 0 && burning_iter <= 1000);
    }
    if (argc >= 4) {
      std::string in(argv[3]);
      iter = std::stoi(in);
      CHECK(iter > 0 && iter <= 1000);
    }
    if (argc >= 5) {
      std::string in(argv[4]);
      test_case_idx = std::stoi(in);
      CHECK(test_case_idx >= 0 && test_case_idx < 3);
    }

    if (argc >= 6) {
      std::string in(argv[5]);
      assert(in == "0" || in == "1");
      bool res = static_cast<bool>(std::stoi(in));
      if (func_option == "3x3") {
        with_fuse = res;
      } else if (func_option == "1x1") {
        with_bias = res;
      } 
    }
    
    // conv desc
    test_convolution_sizes_t cds[] = {
      { // small one
        2, 1,  // bs, gp
        12, 16, 16,  // ic, ih, iw
        32, 14, 14,  // oc, oh, ow
        3, 3,  // kh, kw
        0, 0,  // ph, pw
        1, 1   // sh, sw
      },
      {
        2, 1,  // bs, gp
        64, 128, 128,  // ic, ih, iw
        128, 128, 128,  // oc, oh, ow
        3, 3,  // kh, kw
        0, 0,  // ph, pw
        1, 1   // sh, sw
      },
      {
        2, 1,  // bs, gp
        32, 258, 258,  // ic, ih, iw
        64, 256, 256,  // oc, oh, ow
        3, 3,  // kh, kw
        0, 0,  // ph, pw
        1, 1   // sh, sw
      }
    };
    try {
        if (func_option == "3x3") {
          test_conv(cds[test_case_idx], with_fuse);
        } else if (func_option == "1x1") {
          test_conv3x3_1x1(cds[test_case_idx], with_bias);
        } else {  // reoder
          test_reorder();
        }
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
