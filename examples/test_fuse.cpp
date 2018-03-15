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
#include <assert.h>

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

template<typename dtype>
static void fscanf_data(FILE* fp, dtype* pdata) {
  printf("Error: unkown data type\n");
}

template<>
void fscanf_data<u8>(FILE* fp, u8* pdata) {
  int res = fscanf(fp, "%hhu,", pdata);
}

template<>
void fscanf_data<s8>(FILE* fp, s8* pdata) {
  s32 tmp32 = 0;
  int res = fscanf(fp, "%d,", &tmp32);
  *pdata = static_cast<s8>(tmp32);
}

template<>
void fscanf_data<s32>(FILE* fp, s32* pdata) {
  int res = fscanf(fp, "%d,", pdata);
}

template<typename dtype>
static void load_nhwc(const char* filename, dtype* pdata, const int bs, const int height,
  const int width, const int channel) {
  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: no such file %s\n", filename);
  }
  for (int n = 0; n < bs; ++n) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channel; ++c) {
          int offset = c + w*channel + h*width*channel + n*height*width*channel;
          fscanf_data<dtype>(fp, pdata + offset);
        }
        int res = fscanf(fp, "\n");
      }
    }
  }
  fclose(fp);
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

template<typename dtype>
static void save_x(const char* filename, dtype* pdata, const size_t size) {
  FILE *fp = NULL;\
  fp = fopen(filename, "w");
  for (size_t i = 0; i < size; ++i) {
     fprintf(fp, "%d,", pdata[i]);
  }
  fclose(fp);
}

// return 0 if success
template<typename dtype>
static int compare(dtype* p1, dtype* p2, const size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    if (p1[i] != p2[i]) { return -1;}
  }
  return 0;
}

// save should be always right, so only test load function
void test_load_data(test_convolution_sizes_t* cds) {
  for (int i = 0; i < 4; ++i) {
    auto& p = cds[i];
    std::string filename;
    // check src
    filename = "src_";
    filename += std::to_string(i);
    filename += ".txt";
    std::cout << filename << std::endl;
    size_t sz = p.mb * p.ih * p.iw * p.ic;
    u8* psrc1 = (u8 *)malloc(sz * sizeof(u8));
    u8* psrc2 = (u8 *)malloc(sz * sizeof(u8));
    load_nhwc<u8>(filename.c_str(), psrc1, p.mb, p.ih, p.iw, p.ic);
    filename += ".tmp";
    save_nhwc<u8>(filename.c_str(), psrc1, p.mb, p.ih, p.iw, p.ic);
    load_nhwc<u8>(filename.c_str(), psrc2, p.mb, p.ih, p.iw, p.ic);
    if (compare<u8>(psrc1, psrc2, sz) != 0) {
      printf("Test failed: load src!\n");
      break;
    }
    printf("Pass\n");
    free(psrc1);
    free(psrc2);

    for (std::string s1x1 : {"", "1x1"}) {
    // check wei
    filename = "wei";
    filename += s1x1;
    filename += "_";
    filename += std::to_string(i);
    filename += ".txt";
    std::cout << filename << std::endl;
    sz = p.oc * p.ic * p.kh * p.kw;
    s8* pwei1 = (s8 *)malloc(sz * sizeof(s8));
    s8* pwei2 = (s8 *)malloc(sz * sizeof(s8));
    load_x<s8>(filename.c_str(), pwei1, sz);
    filename += ".tmp";
    save_x<s8>(filename.c_str(), pwei1, sz);
    load_x<s8>(filename.c_str(), pwei2, sz);
    if (compare<s8>(pwei1, pwei2, sz) != 0) {
      printf("Test failed: load wei!\n");
      break;
    }
    printf("Pass\n");
    free(pwei1);
    free(pwei2);
    
    // check bias
    filename = "bia";
    filename += s1x1;
    filename += "_";
    filename += std::to_string(i);
    filename += ".txt";
    std::cout << filename << std::endl;
    sz = p.oc;
    s32* pbia1 = (s32 *)malloc(sz * sizeof(s32));
    s32* pbia2 = (s32 *)malloc(sz * sizeof(s32));
    load_x<s32>(filename.c_str(), pbia1, sz);
    filename += ".tmp";
    save_x<s32>(filename.c_str(), pbia1, sz);
    load_x<s32>(filename.c_str(), pbia2, sz);
    if (compare<s32>(pbia1, pbia2, sz) != 0) {
      printf("Test failed: load bias!\n");
      break;
    }
    printf("Pass\n");
    free(pbia1);
    free(pbia2);
    }
  }
}

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
  primitive_attr attr = mkldnn::primitive_attr();
  attr.set_int_output_round_mode(rmode);
  const int count = 1;  // number of scales
  const int mask = 0;  // multi-channel 1
  const float scale = 0.3f;
  std::vector<float> s(count, scale);
  attr.set_output_scales(mask, s);

  if (with_relu) {
    post_ops ops;
    ops.append_eltwise(1.0f, algorithm::eltwise_relu, negative_slope, 0.f);
    attr.set_post_ops(ops);
  }
   
  return std::unique_ptr<convolution_forward::primitive_desc>(
      new convolution_forward::primitive_desc(*conv_desc, attr, eng));
}

std::unique_ptr<eltwise_forward::primitive_desc> get_relu_pd(const memory::desc md) {
  auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_relu, md, 0.f, 0.f);
  return std::unique_ptr<eltwise_forward::primitive_desc>(
      new eltwise_forward::primitive_desc(relu_desc, eng));
}

void test_conv(const std::string& type, test_convolution_sizes_t* cds,
  const int id, bool with_relu, bool fuse_relu = true, bool with_bias = true) {
  test_convolution_sizes_t cd = cds[id];
  std::vector<primitive> pipeline;
  std::unique_ptr<primitive> fwd;
  std::unique_ptr<primitive> fwd_relu;
  std::unique_ptr<convolution_forward::desc> conv_desc;
  std::unique_ptr<convolution_forward::primitive_desc> conv_pd;
  std::unique_ptr<eltwise_forward::primitive_desc> relu_pd;
  std::unique_ptr<memory> src, wgt, dst, bia;

  if (type == "1x1") {
    // always change the params, the output used as input
    cd.ic = cd.oc; cd.ih = cd.oh; cd.iw = cd.ow;
    cd.oc = 96;
    cd.kh = 1; cd.kw = 1;
    cd.padh = 0; cd.padw = 0;
  }
  conv_desc = get_conv_desc(cd, with_bias);
  conv_pd = get_conv_pd(conv_desc, with_relu && fuse_relu);
  src.reset(new memory(conv_pd->src_primitive_desc()));
  wgt.reset(new memory(conv_pd->weights_primitive_desc()));
  dst.reset(new memory(conv_pd->dst_primitive_desc()));

  if (with_bias) {
    bia.reset(new memory(conv_pd->bias_primitive_desc()));
#ifdef LOAD_SAVE_DATA
    // load data
    s32* pbia = (s32*)(bia->get_data_handle());
    std::string filename;
    filename = "bia";
    filename += (type == "1x1" ? "1x1" : "");
    filename += "_";
    filename += std::to_string(id);
    filename += ".txt";
    std::cout << "Load data " << filename << std::endl;
    load_x<s32>(filename.c_str(), pbia, cd.oc);
#endif
  }

#ifdef LOAD_SAVE_DATA
  std::string filename;
  u8* psrc = (u8*)(src->get_data_handle());
  s8* pwei = (s8*)(wgt->get_data_handle());
  s32* pdst = (s32*)(dst->get_data_handle());

  // load src data
  // 1x1 src data is dst_relu_u8_i.txt
  if (type == "1x1") {
    filename = "dst_relu_u8_";
  } else {
    filename = "src_";
  }
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Load data " << filename << std::endl;
  load_nhwc<u8>(filename.c_str(), psrc, cd.mb, cd.ih, cd.iw, cd.ic);

  // load wei data
  filename = "wei";
  filename += (type == "1x1" ? "1x1" : "");
  filename += "_";
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Load data " << filename << std::endl;
  load_x<s8>(filename.c_str(), pwei, cd.oc * cd.ic * cd.kh * cd.kw);
#endif

  if (with_bias) {
    fwd.reset(new convolution_forward(*conv_pd, *src, *wgt, *bia, *dst));
  } else {
    fwd.reset(new convolution_forward(*conv_pd, *src, *wgt, *dst));
  }
  pipeline.push_back(*fwd);

  if (with_relu && !fuse_relu) {
    // add relu
    relu_pd = get_relu_pd(conv_pd->dst_primitive_desc().desc());
    fwd_relu.reset(new eltwise_forward(*relu_pd, *dst, *dst));
    pipeline.push_back(*fwd_relu);
  }
#ifdef LOAD_SAVE_DATA
  stream(stream::kind::eager).submit(pipeline).wait();
  filename = "conv";
  filename += type;
  filename += (with_relu ? (fuse_relu ? "_relu_" : "+relu_") : "");
  filename += "dst_";
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Save data " << filename << std::endl;
  save_nhwc<s32>(filename.c_str(), pdst, cd.mb, cd.oh, cd.ow, cd.oc);
#endif

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

  std::cout << "Conv" << type;
  if (with_relu) {
    if (fuse_relu) {
      std::cout << "_ReLU fused";
    } else {
      std::cout << "+ReLU non-fused";
    }
  }

  if (with_bias) {
    std::cout << " with bias";
  } else {
    std::cout << " without bias";
  }

#ifdef ENABLE_VNNI
  std::cout << " with VNNI ";
#else
  std::cout << " without VNNI ";
#endif

  std::cout << "avg time: " << (t_stop - t_start) / (double) iter << " ms" << std::endl;
}

// test conv3x3_relu + conv1x1_relu
// u8 (wei:s8, bia:s32) => relu => u8, (wei:s8, bia:s32) => s32
void test_conv3x3_1x1(test_convolution_sizes_t* cds, const int id,
    bool conv3x3_with_relu, bool conv1x1_with_relu,
    bool fuse_3x3relu, bool fuse_1x1relu,
    bool with_bias = true) {
  test_convolution_sizes_t cd = cds[id];
  std::unique_ptr<primitive> fwd3x3, fwd1x1;
  std::vector<primitive> pp3x3, pp1x1;  // pipeline
  std::unique_ptr<memory> src, wgt3x3, dst3x3, bia3x3, wgt1x1, bia1x1, dst;
  std::unique_ptr<convolution_forward::desc> desc3x3, desc1x1;
  std::unique_ptr<convolution_forward::primitive_desc> pd3x3, pd1x1;
  std::unique_ptr<primitive> relu3x3, relu1x1;
  std::unique_ptr<eltwise_forward::primitive_desc> relu3x3_pd, relu1x1_pd;

  desc3x3 = get_conv_desc(cd, with_bias, data_traits<u8>::data_type);
  pd3x3 = get_conv_pd(desc3x3, conv3x3_with_relu && fuse_3x3relu);
  src.reset(new memory(pd3x3->src_primitive_desc()));
  wgt3x3.reset(new memory(pd3x3->weights_primitive_desc()));
  dst3x3.reset(new memory(pd3x3->dst_primitive_desc()));
  if (with_bias) {
    bia3x3.reset(new memory(pd3x3->bias_primitive_desc()));
#ifdef LOAD_SAVE_DATA
    // load bia3x3 data
    std::string filename;
    s32* pbia3x3 = (s32*)(bia3x3->get_data_handle());
    filename = "bia_";
    filename += std::to_string(id);
    filename += ".txt";
    std::cout << "Load data " << filename << std::endl;
    load_x<s32>(filename.c_str(), pbia3x3, cd.oc);
#endif
  }
#ifdef LOAD_SAVE_DATA
  u8* psrc = (u8*)(src->get_data_handle());
  s8* pwei3x3 = (s8*)(wgt3x3->get_data_handle());
  u8* pdst3x3 = (u8*)(dst3x3->get_data_handle());
  std::string filename;
  // load src data
  filename = "src_";
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Load data " << filename << std::endl;
  load_nhwc<u8>(filename.c_str(), psrc, cd.mb, cd.ih, cd.iw, cd.ic);
  
  // load wei3x3 data
  filename = "wei_";
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Load data " << filename << std::endl;
  load_x<s8>(filename.c_str(), pwei3x3, cd.oc * cd.ic * cd.kh * cd.kw);
#endif

  // 1x1
  cd.ic = cd.oc; cd.ih = cd.oh; cd.iw = cd.ow;
  cd.oc = 96;
  cd.kh = 1; cd.kw = 1;
  cd.padh = 0; cd.padw = 0;
  desc1x1 = get_conv_desc(cd, with_bias, data_traits<s32>::data_type);
  pd1x1 = get_conv_pd(desc1x1, conv1x1_with_relu && fuse_1x1relu);
  wgt1x1.reset(new memory(pd1x1->weights_primitive_desc()));
  dst.reset(new memory(pd1x1->dst_primitive_desc()));
  if (with_bias) {
    bia1x1.reset(new memory(pd1x1->bias_primitive_desc()));
#ifdef LOAD_SAVE_DATA
    std::string filename;
    s32* pbia1x1 = (s32*)(bia1x1->get_data_handle());
    // load bia1x1 data
    filename = "bia1x1_";
    filename += std::to_string(id);
    filename += ".txt";
    std::cout << "Load data " << filename << std::endl;
    load_x<s32>(filename.c_str(), pbia1x1, cd.oc);
#endif
  }
  
#ifdef LOAD_SAVE_DATA
  s8* pwei1x1 = (s8*)(wgt1x1->get_data_handle());
  s32* pdst1x1 = (s32*)(dst->get_data_handle());
  // load wei1x1 data
  filename = "wei1x1_";
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Load data " << filename << std::endl;
  load_x<s8>(filename.c_str(), pwei1x1, cd.oc * cd.ic * cd.kh * cd.kw);
#endif

  if (with_bias) {
    fwd3x3.reset(new convolution_forward(*pd3x3, *src, *wgt3x3, *bia3x3, *dst3x3));
    fwd1x1.reset(new convolution_forward(*pd1x1, *dst3x3, *wgt1x1, *bia1x1, *dst));
  } else {
    fwd3x3.reset(new convolution_forward(*pd3x3, *src, *wgt3x3, *dst3x3));
    fwd1x1.reset(new convolution_forward(*pd1x1, *dst3x3, *wgt1x1, *dst));
  }
  pp3x3.push_back(*fwd3x3);
  pp1x1.push_back(*fwd1x1);

  if (conv3x3_with_relu && !fuse_3x3relu) {
    // add 3x3relu
    relu3x3_pd = get_relu_pd(pd3x3->dst_primitive_desc().desc());
    relu3x3.reset(new eltwise_forward(*relu3x3_pd, *dst3x3, *dst3x3));
    pp3x3.push_back(*relu3x3);
  }
  if (conv1x1_with_relu && !fuse_1x1relu) {
    // add 1x1relu
    relu1x1_pd = get_relu_pd(pd1x1->dst_primitive_desc().desc());
    relu1x1.reset(new eltwise_forward(*relu1x1_pd, *dst, *dst));
    pp1x1.push_back(*relu1x1);
  }

#ifdef LOAD_SAVE_DATA
  // save 3x3 relu intermated data, this data is u8 type
  stream(stream::kind::eager).submit(pp3x3).wait();
  filename = "conv3x3";
  filename += (conv3x3_with_relu ? (fuse_3x3relu ? "_relu_" : "+relu_") : "");
  filename += "dst_u8_";
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Save data " << filename << std::endl;
  save_nhwc<u8>(filename.c_str(), pdst3x3, cd.mb, cd.ih, cd.iw, cd.ic);

  // save 1x1 relu output data, s32 dtype
  stream(stream::kind::eager).submit(pp1x1).wait();
  filename = "conv1x1";
  filename += (conv1x1_with_relu ? (fuse_1x1relu ? "_relu_" : "+relu_") : "");
  filename += "dst_";
  filename += std::to_string(id);
  filename += ".txt";
  std::cout << "Save data " << filename << std::endl;
  save_nhwc<s32>(filename.c_str(), pdst1x1, cd.mb, cd.oh, cd.ow, cd.oc);
#endif

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

  std::cout << "Conv3x3";
  if (conv3x3_with_relu) {
    if (fuse_3x3relu) {
      std::cout << "_ReLU fused";
    } else {
      std::cout << "+ReLU non-fused";
    }
  }

  std::cout << " + Conv1x1";
  if (conv1x1_with_relu) {
    if (fuse_1x1relu) {
      std::cout << "_ReLU fused";
    } else {
      std::cout << "+ReLU non-fused";
    }
  }

  if (with_bias) {
    std::cout << " with bias";
  } else {
    std::cout << " without bias";
  }
    
#ifdef ENABLE_VNNI
  std::cout << " with VNNI ";
#else
  std::cout << " without VNNI ";
#endif
  std::cout << "avg time ("
  << sum3x3 / (double)iter << " + " << sum1x1 / (double)iter << "): "
  << (t_stop - t_start) / (double) iter << " ms" << std::endl;
}

static void test_concat(bool with_relu = false) {
  std::unique_ptr<primitive> fwd_concat, fwd_relu;
  std::vector<primitive> pp_concat, pp_relu;  // pipeline

  // below is input
  using dtype = s32;
  int concat_dimension = 1;
  memory::format fmt = memory::format::nhwc;
  // note: src dims always is nchw format, only data layout can be nhwc
  std::vector<memory::dims> src_dims = {
    {4, 96, 224, 224},
    {4, 256, 224, 224}};

  // cal dst dims
  int oc = src_dims[0][concat_dimension];
  assert(src_dims[0].size() == 4);
  for (size_t i = 1; i < src_dims.size(); i++) {
    assert(src_dims[0].size() == src_dims[i].size());
    for (size_t dim = 0; dim < src_dims[i].size(); ++dim) {
      if (dim == (size_t)concat_dimension) {
        oc += src_dims[i][dim];
      } else {
        assert(src_dims[i][dim] == src_dims[0][dim]);
      }
    }
  }
  memory::dims dst_dims = {src_dims[0][0], oc, src_dims[0][2], src_dims[0][3]};
  memory::data_type data_type = data_traits<dtype>::data_type;

  // allocate srcs memory
  std::vector<memory::primitive_desc> srcs_pd;
  std::vector<memory> srcs;
  for (size_t i = 0; i < src_dims.size(); ++i) {
    auto desc = memory::desc(src_dims[i], data_type, fmt);
    auto mpd = memory::primitive_desc(desc, eng);
    auto src_memory = memory(mpd);
    //const size_t sz = src_memory.get_primitive_desc().get_size() / sizeof(data_t);
    //fill_data<data_t>(sz, (data_t *)src_memory.get_data_handle());
    srcs_pd.push_back(mpd);
    srcs.push_back(src_memory);
  }

  // dst memory
  auto dst_desc = memory::desc(dst_dims, data_type, fmt);
  auto concat_pd = concat::primitive_desc(dst_desc, concat_dimension, srcs_pd);
  auto dst = memory(concat_pd.dst_primitive_desc());

  // concat
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs.size(); i++) {
      inputs.push_back(srcs[i]);
  }
  fwd_concat.reset(new concat(concat_pd, inputs, dst));
  pp_concat.clear();
  pp_concat.push_back(*fwd_concat);

  if (with_relu) {

  }

  // cal time
  for (auto i = 0; i < burning_iter; ++i) {
    stream(stream::kind::eager).submit(pp_concat).wait();
    //stream(stream::kind::eager).submit(fwd_relu).wait();
  }

  auto get_current_ms = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
  };

  auto t_start = get_current_ms();
  double sum_concat = 0;
  double sum_relu = 0;
  
  for (auto i = 0; i < iter; ++i) {
    auto s1 = get_current_ms();
    stream(stream::kind::eager).submit(pp_concat).wait();
    auto s2 = get_current_ms();
    //stream(stream::kind::eager).submit(pp1x1).wait();
    //auto s3 = get_current_ms();
    sum_concat += (s2 - s1);
    //sum_relu += (s3 - s2);
  }
  auto t_stop = get_current_ms();


  std::cout << "In";
  for (size_t i = 0; i < src_dims.size(); i++) {
    auto& dims = src_dims[i];
    printf("(%d, %d, %d, %d) ", dims[0], dims[1], dims[2], dims[3]);
  }
  printf("==> Out(%d, %d, %d, %d)\n", dst_dims[0], dst_dims[1], dst_dims[2], dst_dims[3]);


  std::cout << "Concat";
  if (with_relu) {
  }

  std::cout << "avg time ("
  << sum_concat / (double)iter << "): "
  << (t_stop - t_start) / (double) iter << " ms" << std::endl;
}

static void usage() {
  std::cout << "./test_fuse function_options test_case_idx(0/1/2/3) buring_iter(1~1000) valid_iter(1~1000)"<< std::endl;
  std::cout << "function_options:\n e.g.\n" << std::endl;
  std::cout << "3x3_relu: conv3x3 fused relu" << std::endl;
  std::cout << "3x3_relu+1x1_relu: conv3x3 fused relu and conv1x1 fused relu" << std::endl;
  std::cout << "test_case_idx: \n default is 3 \n" << std::endl;
  std::cout << "buring_iter: \n defalut is 200 \nvalid_iter: default is 200" << std::endl;
  std::cout << "with_bias: 0 or 1" << std::endl;
}

#define CHECK(x) \
  if (!(x)) {  \
    std::cout << "Check failed!" << std::endl;\
    usage(); assert(false);\
  }

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

int main(int argc, char **argv) {
  // std::cout << argv[0] << std::endl;
  std::string func_option("3x3_relu");
  bool with_bias = true;
  int test_case_idx = 3;
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
      1, 1,  // ph, pw
      1, 1   // sh, sw
    },
    {
      2, 1,  // bs, gp
      32, 258, 258,  // ic, ih, iw
      64, 256, 256,  // oc, oh, ow
      3, 3,  // kh, kw
      0, 0,  // ph, pw
      1, 1   // sh, sw
    },
    {
      50, 1,  // bs, gp
      64, 56, 56,  // ic, ih, iw
      64, 56, 56,  // oc, oh, ow
      3, 3,  // kh, kw
      1, 1,  // ph, pw
      1, 1   // sh, sw
    }
  };
  if (argc >= 2) {
    std::string in(argv[1]);
    func_option = in;
    CHECK(one_of(func_option,
      "3x3_relu+1x1_relu",
      "3x3_relu+1x1+relu",
      "3x3+relu+1x1_relu",
      "3x3+relu+1x1+relu",
      "3x3+1x1",
      "3x3",
      "1x1",
      "3x3_relu",
      "3x3+relu",
      "1x1_relu",
      "1x1+relu",
      "reorder",
      "concat",
      "test_load_data"
      ));
  }
  if (one_of(func_option, "reorder", "test_load_data") && argc >= 3) {
    std::cout << "Warnning: \" " << argv[2] <<
      " \" and after inputs are invalid when test_reorder or test_load_data." << std::endl;
  }
  
  if (argc >= 3) {
    std::string in(argv[2]);
    test_case_idx = std::stoi(in);
    CHECK(test_case_idx >= 0 && test_case_idx < (int)(sizeof(cds)/sizeof(cds[0])));
  }
  if (argc >= 4) {
    std::string in(argv[3]);
    burning_iter = std::stoi(in);
    CHECK(burning_iter > 0 && burning_iter <= 1000);
  }
  if (argc >= 5) {
    std::string in(argv[4]);
    iter = std::stoi(in);
    CHECK(iter > 0 && iter <= 1000);
  }

  if (argc >= 6) {
    std::string in(argv[5]);
    assert(in == "0" || in == "1");
    with_bias = static_cast<bool>(std::stoi(in));
  }
  if (argc >= 7) {
    std::cout << "Warnning: no more args accepted: " << argv[6] << std::endl;
    usage();
    return 0;
  }

  auto &pm = cds[test_case_idx];
  if (!(one_of(func_option, "concat", "concat_relu", "concat+relu"))) {
    printf("In(%d, %d, %d, %d) ==> Out(%d, %d, %d, %d), kernel(%d, %d)\n",
      pm.mb, pm.ic, pm.ih, pm.iw, pm.mb, pm.oc, pm.oh, pm.ow, pm.kh, pm.kw);  // params
  }
  try {
    if (func_option == "3x3") {
      test_conv("3x3", cds, test_case_idx, false, false, with_bias);
    } else if (func_option == "3x3_relu") {
      test_conv("3x3", cds, test_case_idx, true, true, with_bias);
    } else if (func_option == "3x3+relu") {
      test_conv("3x3", cds, test_case_idx, true, false, with_bias);
    } else if (func_option == "1x1") {
      test_conv("1x1", cds, test_case_idx, false, false, with_bias);
    } else if (func_option == "1x1_relu") {
      test_conv("1x1", cds, test_case_idx, true, true, with_bias);
    } else if (func_option == "1x1+relu") {
      test_conv("1x1", cds, test_case_idx, true, false, with_bias);
    } else if (func_option == "3x3+1x1") {
      test_conv3x3_1x1(cds, test_case_idx, false, false, true, true, with_bias);
    } else if (func_option == "3x3_relu+1x1_relu") {
      test_conv3x3_1x1(cds, test_case_idx, true, true, true, true, with_bias);
    } else if (func_option == "3x3+relu+1x1+relu") {
      test_conv3x3_1x1(cds, test_case_idx, true, true, false, false, with_bias);
    } else if (func_option == "3x3_relu+1x1+relu") {
      test_conv3x3_1x1(cds, test_case_idx, true, true, true, false, with_bias);
    } else if (func_option == "3x3+relu+1x1_relu") {
      test_conv3x3_1x1(cds, test_case_idx, true, true, false, true, with_bias);
    } else if (func_option == "test_load_data") {
      test_load_data(cds);
    } else if (func_option == "concat") {
      test_concat(false);
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
