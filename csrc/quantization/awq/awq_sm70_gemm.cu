/*
 * SM70 AWQ GEMM integration using TurboMind s884h kernels.
 * Adapted from LMDeploy TurboMind (Apache-2.0).
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <mutex>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind {
void unpack_awq_gemm(uint4_t* dst, const uint4_t* src, int rows, int cols, cudaStream_t st);
}  // namespace turbomind

namespace vllm {
namespace awq_sm70 {

namespace {

struct WorkspaceHolder {
  torch::Tensor barriers;
  torch::Tensor partials;
  torch::Tensor tensormaps;
  torch::Tensor flags;
  turbomind::gemm::Workspace workspace{};
};

struct GemmHolder {
  std::unique_ptr<turbomind::gemm::Gemm> gemm;
};

std::mutex workspace_mutex;
std::mutex gemm_mutex;
std::unordered_map<int, WorkspaceHolder> workspace_cache;
std::unordered_map<int, GemmHolder> gemm_cache;

WorkspaceHolder& get_workspace(int device) {
  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto it = workspace_cache.find(device);
  if (it != workspace_cache.end()) {
    return it->second;
  }

  WorkspaceHolder holder;
  auto byte_opts = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, device))
                       .dtype(torch::kUInt8);
  auto int_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device))
                      .dtype(torch::kInt32);

  holder.barriers = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kBarriersSize}, byte_opts);
  holder.partials = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kPartialsSize}, byte_opts);
  // Keep same tensormap size as TurboMind LlamaLinear.
  holder.tensormaps = torch::empty({(long long)(8192 * 128)}, byte_opts);
  holder.flags = torch::zeros({1}, int_opts);

  holder.workspace.barriers = holder.barriers.data_ptr();
  holder.workspace.barriers_size = holder.barriers.numel();
  holder.workspace.partials = holder.partials.data_ptr();
  holder.workspace.partials_size = holder.partials.numel();
  holder.workspace.tensormaps = holder.tensormaps.data_ptr();
  holder.workspace.tensormaps_size = holder.tensormaps.numel();
  holder.workspace.flags = holder.flags.data_ptr<int>();

  auto [insert_it, _] = workspace_cache.emplace(device, std::move(holder));
  return insert_it->second;
}

turbomind::gemm::Gemm& get_gemm(int device) {
  std::lock_guard<std::mutex> lock(gemm_mutex);
  auto it = gemm_cache.find(device);
  if (it != gemm_cache.end()) {
    return *it->second.gemm;
  }
  GemmHolder holder;
  holder.gemm = std::make_unique<turbomind::gemm::Gemm>();
  auto [insert_it, _] = gemm_cache.emplace(device, std::move(holder));
  return *insert_it->second.gemm;
}

void validate_awq_inputs(const torch::Tensor& qweight,
                         const torch::Tensor& scales,
                         const torch::Tensor& qzeros) {
  TORCH_CHECK(qweight.is_cuda(), "awq_sm70_prepare: qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "awq_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(qzeros.is_cuda(), "awq_sm70_prepare: qzeros must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qweight must be int32.");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qzeros must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "awq_sm70_prepare: scales must be float16.");
}

}  // namespace

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            int64_t group_size) {
  validate_awq_inputs(qweight, scales, qzeros);

  qweight = qweight.contiguous();
  scales = scales.contiguous();
  qzeros = qzeros.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1) * 8;
  const int64_t num_groups = scales.size(0);

  TORCH_CHECK(scales.size(1) == n,
              "awq_sm70_prepare: scales shape mismatch.");
  TORCH_CHECK(qzeros.size(0) == num_groups,
              "awq_sm70_prepare: qzeros group mismatch.");
  TORCH_CHECK(qzeros.size(1) * 8 == n,
              "awq_sm70_prepare: qzeros shape mismatch.");
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0,
              "awq_sm70_prepare: K and N must be multiples of 8.");
  TORCH_CHECK(k % num_groups == 0,
              "awq_sm70_prepare: input dim must be divisible by groups.");

  if (group_size <= 0) {
    group_size = k / num_groups;
  }
  TORCH_CHECK(k / num_groups == group_size,
              "awq_sm70_prepare: group_size mismatch with scales.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_sm70_prepare: SM70 AWQ supports group_size=32/64/128, got ",
              group_size, ".");

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_sm70_prepare: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto packed_weight = torch::empty_like(qweight);
  turbomind::unpack_awq_gemm(
      reinterpret_cast<turbomind::uint4_t*>(packed_weight.data_ptr<int>()),
      reinterpret_cast<const turbomind::uint4_t*>(qweight.data_ptr<int>()),
      static_cast<int>(k), static_cast<int>(n), stream);

  auto u16_opts = torch::TensorOptions()
                      .device(qweight.device())
                      .dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const turbomind::uint4_t*>(
          packed_weight.data_ptr<int>()),
      tmp_u16.numel(), stream);

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };

  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty_like(qweight);
  TORCH_CHECK(
      conv_w->Convert(tmp_u16_conv.data_ptr(),
                      w_desc,
                      tm_weight.data_ptr(),
                      k_desc,
                      stream) == 0,
      "awq_sm70_prepare: weight conversion failed.");

  // Unpack AWQ zeros using PyTorch tensor ops (matches lmdeploy's Python
  // approach).  The C++ unpack_awq_gemm() requires rows%8==0 which fails
  // when num_groups < 8 (e.g. Qwen3-30B-A3B w2: K=768, num_groups=6).
  const int awq_order[] = {0, 4, 1, 5, 2, 6, 3, 7};
  std::vector<torch::Tensor> zslices;
  auto zz = qzeros;
  for (int i = 0; i < 8; ++i) {
    zslices.push_back((zz & 0xF).to(torch::kUInt8));
    zz = zz.__rshift__(4);
  }
  std::vector<torch::Tensor> zordered;
  for (int i = 0; i < 8; ++i) {
    zordered.push_back(zslices[awq_order[i]]);
  }
  auto zeros_half = torch::stack(zordered, -1)
                        .reshape({num_groups, n})
                        .to(torch::kFloat16);

  auto fused = torch::empty({num_groups, n * 2},
                            torch::TensorOptions()
                                .device(scales.device())
                                .dtype(torch::kFloat16));
  turbomind::fuse_scales_and_zeros(
      reinterpret_cast<half*>(fused.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(zeros_half.data_ptr<at::Half>()),
      scales.numel(), stream);

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty(
      {num_groups, n},
      torch::TensorOptions()
          .device(scales.device())
          .dtype(torch::kInt32));
  TORCH_CHECK(
      conv_s->Convert(fused.data_ptr(),
                      s_desc,
                      tm_scales.data_ptr(),
                      q_desc,
                      stream) == 0,
      "awq_sm70_prepare: scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

torch::Tensor awq_gemm_sm70(torch::Tensor in_feats,
                            torch::Tensor tm_weight,
                            torch::Tensor tm_scales,
                            int64_t group_size,
                            int64_t k_ld,
                            int64_t q_ld) {
  TORCH_CHECK(in_feats.is_cuda(), "awq_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "awq_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "awq_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt32,
              "awq_gemm_sm70: weight must be int32.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kInt32,
              "awq_gemm_sm70: scales must be int32.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1) * 8;
  TORCH_CHECK(tm_weight.size(0) == k,
              "awq_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(k % group_size == 0,
              "awq_gemm_sm70: input dim must be divisible by group size.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "awq_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n,
              "awq_gemm_sm70: scale shape mismatch.");

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  const int64_t num_groups_raw = k / group_size;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups_raw),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(q_ld);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(n),
  };

  auto options = torch::TensorOptions()
                     .dtype(in_feats.dtype())
                     .device(in_feats.device());
  auto out = torch::empty({m, n}, options);

  turbomind::gemm::Operation op{};
  op.dispatch = turbomind::gemm::DispatchPolicy::kDefault;
  op.epilogue = turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op,
                          1.f,
                          in_feats.data_ptr(),
                          desc_A,
                          nullptr,
                          desc_U,
                          tm_weight.data_ptr(),
                          desc_B,
                          tm_scales.data_ptr(),
                          desc_V,
                          0.f,
                          out.data_ptr(),
                          desc_D,
                          out.data_ptr(),
                          desc_D,
                          workspace_holder.workspace,
                          stream);
  TORCH_CHECK(ec == 0, "awq_gemm_sm70: TurboMind GEMM failed.");
  return out;
}

}  // namespace awq_sm70
}  // namespace vllm

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            torch::Tensor _zeros,
                                            int64_t group_size) {
  return vllm::awq_sm70::awq_sm70_prepare(
      _kernel, _scaling_factors, _zeros, group_size);
}

torch::Tensor awq_gemm_sm70(torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors,
                            int64_t group_size,
                            int64_t k_ld,
                            int64_t q_ld) {
  return vllm::awq_sm70::awq_gemm_sm70(
      _in_feats, _kernel, _scaling_factors, group_size, k_ld, q_ld);
}

// ---------------------------------------------------------------------------
// MoE batched GEMM support
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> awq_moe_build_strided_ptrs(
    torch::Tensor tm_weights,   // [E, ...]  stacked TM weights
    torch::Tensor tm_scales,    // [E, ...]  stacked TM scales
    int64_t k_ld,
    int64_t q_ld,
    int64_t num_experts) {
  TORCH_CHECK(tm_weights.is_cuda(), "awq_moe_build_strided_ptrs: weights must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "awq_moe_build_strided_ptrs: scales must be CUDA.");
  TORCH_CHECK(num_experts > 0, "awq_moe_build_strided_ptrs: num_experts must be > 0.");
  TORCH_CHECK(tm_weights.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: weights dim0 != num_experts.");
  TORCH_CHECK(tm_scales.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: scales dim0 != num_experts.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tm_weights));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Build {ptr, stride} pairs for each expert
  std::vector<std::pair<void*, int>> w_ptrs;
  std::vector<std::pair<void*, int>> s_ptrs;
  w_ptrs.reserve(num_experts);
  s_ptrs.reserve(num_experts);

  const int64_t w_expert_stride = tm_weights.stride(0) * tm_weights.element_size();
  const int64_t s_expert_stride = tm_scales.stride(0) * tm_scales.element_size();
  char* w_base = static_cast<char*>(tm_weights.data_ptr());
  char* s_base = static_cast<char*>(tm_scales.data_ptr());

  for (int64_t e = 0; e < num_experts; ++e) {
    w_ptrs.emplace_back(w_base + e * w_expert_stride, static_cast<int>(k_ld));
    s_ptrs.emplace_back(s_base + e * s_expert_stride, static_cast<int>(q_ld));
  }

  // MakeStridedPtrs allocates GPU memory via cudaMallocAsync
  void* w_gpu = turbomind::gemm::MakeStridedPtrs(w_ptrs, stream);
  void* s_gpu = turbomind::gemm::MakeStridedPtrs(s_ptrs, stream);

  // Wrap in torch tensors for lifetime management.
  // StridedPtr is 16 bytes (__align__(16): void* ptr + int stride + padding).
  const int64_t buf_bytes = num_experts * 16;
  auto opts = torch::TensorOptions()
                  .device(tm_weights.device())
                  .dtype(torch::kUInt8);

  // Copy into torch-managed tensors so cudaFree of the original is safe.
  auto w_tensor = torch::empty({buf_bytes}, opts);
  auto s_tensor = torch::empty({buf_bytes}, opts);
  cudaMemcpyAsync(w_tensor.data_ptr(), w_gpu, buf_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(s_tensor.data_ptr(), s_gpu, buf_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(w_gpu, stream);
  cudaFreeAsync(s_gpu, stream);

  return {w_tensor, s_tensor};
}

torch::Tensor awq_moe_gemm_sm70(
    torch::Tensor sorted_input,     // [total_tokens, K] float16
    torch::Tensor expert_offsets,    // [num_experts + 1] int32
    torch::Tensor strided_ptrs_w,   // [num_experts * 16] uint8 (StridedPtr array)
    torch::Tensor strided_ptrs_s,   // [num_experts * 16] uint8 (StridedPtr array)
    int64_t num_experts,
    int64_t k,
    int64_t n,
    int64_t group_size) {
  TORCH_CHECK(sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
              "awq_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.scalar_type() == torch::kInt32,
              "awq_moe_gemm_sm70: expert_offsets must be CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda() && strided_ptrs_s.is_cuda(),
              "awq_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "awq_moe_gemm_sm70: invalid dimensions.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t total_tokens = sorted_input.size(0);
  auto out = torch::empty({total_tokens, n},
      torch::TensorOptions().dtype(sorted_input.dtype()).device(sorted_input.device()));

  if (total_tokens == 0) return out;

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_moe_gemm_sm70: no compatible TurboMind converters.");

  // desc_A: input activations with offsets (kBlocked mode)
  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::MatrixLayout desc_U{};

  // desc_B: weights via StridedPtr (ld=0 triggers StridedPtr resolution)
  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf, order_w,
      static_cast<int>(n), static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = 0;  // StridedPtr mode
  desc_B.num = static_cast<int>(num_experts);

  // desc_V: scales via StridedPtr
  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  const int64_t num_groups_raw = k / group_size;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32, order_s,
      static_cast<int>(n), static_cast<int>(num_groups_raw),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = 0;  // StridedPtr mode
  desc_V.num = static_cast<int>(num_experts);

  // desc_D: output with offsets (same as A)
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(n),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation op{};
  op.dispatch = turbomind::gemm::DispatchPolicy::kDefault;
  op.epilogue = turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = vllm::awq_sm70::get_workspace(device);
  auto& gemm = vllm::awq_sm70::get_gemm(device);

  const int ec = gemm.Run(op, 1.f,
      sorted_input.data_ptr(), desc_A,
      nullptr, desc_U,
      strided_ptrs_w.data_ptr(), desc_B,
      strided_ptrs_s.data_ptr(), desc_V,
      0.f,
      out.data_ptr(), desc_D,
      out.data_ptr(), desc_D,
      workspace_holder.workspace, stream);

  TORCH_CHECK(ec == 0, "awq_moe_gemm_sm70: TurboMind batched GEMM failed (ec=",
              ec, ").");
  return out;
}
