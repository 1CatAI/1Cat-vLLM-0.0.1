# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SM70 AWQ MoE method using TurboMind s884h GEMM kernels.

Pre-allocates all intermediate buffers (lmdeploy-style) for CUDA graph
compatibility. Uses batched Gemm::Run via StridedPtr arrays, zero CUDA syncs.
Falls back to per-expert loop if batched GEMM is unavailable.
"""

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.linear import set_weight_attrs

logger = init_logger(__name__)

# Max tokens for pre-allocated buffers. Must cover the largest CUDA graph
# capture size to avoid repeated reallocation during graph capture.
# 8192 covers typical max_num_seqs * max_num_batched_tokens scenarios.
_DEFAULT_MAX_TOKENS = 8192


class AWQSM70MoEMethod(FusedMoEMethodBase):
    """AWQ MoE method for SM70 (V100) using TurboMind GEMM kernels.

    Only supports group_size=32/64/128, float16, 4-bit weights.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        if weight_bits != 4:
            raise ValueError(
                f"AWQSM70MoEMethod only supports 4-bit, got {weight_bits}."
            )
        if group_size not in (32, 64, 128):
            raise ValueError(
                f"AWQSM70MoEMethod supports group_size=32/64/128, got {group_size}."
            )
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // weight_bits  # 8

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )
        extra_weight_attrs.pop("intermediate_size_full", None)

        w13_qweight = Parameter(
            torch.empty(num_experts, hidden_size,
                        2 * intermediate_size_per_partition // self.pack_factor,
                        dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(num_experts, intermediate_size_per_partition,
                        hidden_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        w13_scales = Parameter(
            torch.empty(num_experts, num_groups_w13,
                        intermediate_size_per_partition * 2, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = Parameter(
            torch.empty(num_experts, num_groups_w13,
                        2 * intermediate_size_per_partition // self.pack_factor,
                        dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(num_experts, num_groups_w2,
                        hidden_size // self.pack_factor, dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert AWQ weights to TurboMind format and pre-allocate buffers."""
        num_experts = layer.w13_qweight.shape[0]
        device = layer.w13_qweight.device

        # --- Prepare TurboMind weights per expert ---
        w13_tm_weights, w13_tm_scales, w13_meta = [], [], []
        w2_tm_weights, w2_tm_scales, w2_meta = [], [], []

        for e in range(num_experts):
            r13 = ops.awq_sm70_prepare(
                layer.w13_qweight[e], layer.w13_scales[e],
                layer.w13_qzeros[e], self.group_size)
            w13_tm_weights.append(r13[0])
            w13_tm_scales.append(r13[1])
            w13_meta.append(r13[2])

            r2 = ops.awq_sm70_prepare(
                layer.w2_qweight[e], layer.w2_scales[e],
                layer.w2_qzeros[e], self.group_size)
            w2_tm_weights.append(r2[0])
            w2_tm_scales.append(r2[1])
            w2_meta.append(r2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False)
        layer.w13_tm_scales = Parameter(
            torch.stack(w13_tm_scales), requires_grad=False)
        layer.w2_tm_weight = Parameter(
            torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(
            torch.stack(w2_tm_scales), requires_grad=False)

        # Cache meta as CPU ints (zero-cost at inference)
        layer.w13_meta_list = [
            (int(w13_meta[i][0].item()), int(w13_meta[i][1].item()))
            for i in range(num_experts)
        ]
        layer.w2_meta_list = [
            (int(w2_meta[i][0].item()), int(w2_meta[i][1].item()))
            for i in range(num_experts)
        ]
        layer.sm70_num_experts = num_experts

        # Dimensions for batched GEMM
        layer.sm70_w13_k_dim = layer.w13_tm_weight.shape[1]
        layer.sm70_w13_n_dim = layer.w13_tm_weight.shape[2] * 8
        layer.sm70_w2_k_dim = layer.w2_tm_weight.shape[1]
        layer.sm70_w2_n_dim = layer.w2_tm_weight.shape[2] * 8
        intermediate_size = layer.sm70_w2_k_dim
        hidden_size = layer.sm70_w13_k_dim
        layer.sm70_intermediate_size = intermediate_size

        # --- Build StridedPtr arrays for batched GEMM ---
        w13_k_ld, w13_q_ld = layer.w13_meta_list[0]
        w2_k_ld, w2_q_ld = layer.w2_meta_list[0]
        try:
            w13_ptrs = ops.awq_moe_build_strided_ptrs(
                layer.w13_tm_weight, layer.w13_tm_scales,
                w13_k_ld, w13_q_ld, num_experts)
            w2_ptrs = ops.awq_moe_build_strided_ptrs(
                layer.w2_tm_weight, layer.w2_tm_scales,
                w2_k_ld, w2_q_ld, num_experts)
            layer.w13_strided_ptrs_w = Parameter(
                w13_ptrs[0], requires_grad=False)
            layer.w13_strided_ptrs_s = Parameter(
                w13_ptrs[1], requires_grad=False)
            layer.w2_strided_ptrs_w = Parameter(
                w2_ptrs[0], requires_grad=False)
            layer.w2_strided_ptrs_s = Parameter(
                w2_ptrs[1], requires_grad=False)
            layer.sm70_batched_ready = True
            logger.info("SM70 MoE: batched GEMM enabled (%d experts)",
                        num_experts)
        except Exception as e:
            layer.sm70_batched_ready = False
            logger.warning("SM70 MoE: batched GEMM unavailable (%s), "
                           "using per-expert loop fallback.", e)

        # --- Pre-allocate buffers (lmdeploy-style, CUDA graph safe) ---
        top_k = self.moe.experts_per_token
        max_slots = _DEFAULT_MAX_TOKENS * top_k
        layer._buf_max_slots = max_slots
        layer._buf_top_k = top_k
        layer._buf_expert_counts = torch.empty(
            num_experts, dtype=torch.int32, device=device)
        layer._buf_expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device)
        layer._buf_intermediate = torch.empty(
            max_slots, intermediate_size, dtype=torch.float16, device=device)
        layer._buf_ones = torch.ones(
            max_slots, dtype=torch.int32, device=device)
        layer._buf_output = torch.empty(
            _DEFAULT_MAX_TOKENS, hidden_size, dtype=torch.float16,
            device=device)

        # Free original weights
        del layer.w13_qweight, layer.w13_scales, layer.w13_qzeros
        del layer.w2_qweight, layer.w2_scales, layer.w2_qzeros

    def _ensure_buffers(self, layer: torch.nn.Module,
                        total_slots: int, num_tokens: int):
        """Grow pre-allocated buffers if needed (rare, outside CUDA graph)."""
        device = layer._buf_expert_counts.device
        if total_slots > layer._buf_max_slots:
            layer._buf_max_slots = total_slots
            layer._buf_intermediate = torch.empty(
                total_slots, layer.sm70_intermediate_size,
                dtype=torch.float16, device=device)
            layer._buf_ones = torch.ones(
                total_slots, dtype=torch.int32, device=device)
            logger.warning("SM70 MoE: grew buffers to %d slots", total_slots)
        if num_tokens > layer._buf_output.shape[0]:
            hidden_size = layer.sm70_w13_k_dim
            layer._buf_output = torch.empty(
                num_tokens, hidden_size, dtype=torch.float16, device=device)
            logger.warning("SM70 MoE: grew output buf to %d tokens",
                           num_tokens)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """MoE forward: batched GEMM (preferred) or sorted-loop fallback."""
        if getattr(layer, "sm70_batched_ready", False):
            return self._apply_batched(layer, x, topk_weights, topk_ids)
        return self._apply_sorted_loop(layer, x, topk_weights, topk_ids)

    def _sort_tokens_by_expert(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ):
        """Sort tokens by expert (all GPU, no sync, CUDA graph safe).

        Uses scatter_add_ instead of torch.bincount for graph compatibility.
        Uses pre-allocated expert_counts/offsets buffers.
        """
        top_k = topk_ids.shape[1]
        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1)
        total_slots = flat_ids.shape[0]

        # Sort by expert ID
        sorted_expert_ids, sorted_order = flat_ids.long().sort(stable=True)
        sorted_token_origin = sorted_order // top_k
        sorted_weights = flat_weights[sorted_order]
        sorted_input = x[sorted_token_origin]

        # Expert offsets via scatter_add_ (CUDA graph compatible,
        # replaces torch.bincount which is not compilable)
        expert_counts = layer._buf_expert_counts
        expert_counts.zero_()
        expert_counts.scatter_add_(
            0, sorted_expert_ids.int(), layer._buf_ones[:total_slots])

        expert_offsets = layer._buf_expert_offsets
        expert_offsets.zero_()
        torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

        return (sorted_input, sorted_token_origin, sorted_weights,
                expert_offsets)

    def _apply_batched(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Batched GEMM via TurboMind. Zero CUDA syncs, graph safe."""
        num_tokens, hidden_size = x.shape
        top_k = topk_ids.shape[1]
        num_experts = layer.sm70_num_experts
        total_slots = num_tokens * top_k

        self._ensure_buffers(layer, total_slots, num_tokens)
        output = layer._buf_output[:num_tokens]
        output.zero_()
        if total_slots == 0:
            return output

        (sorted_input, sorted_token_origin, sorted_weights,
         expert_offsets) = self._sort_tokens_by_expert(
            layer, x, topk_weights, topk_ids, num_experts)

        # Batched w13 GEMM (gate+up fused) — single Gemm::Run
        gate_up = ops.awq_moe_gemm_sm70(
            sorted_input, expert_offsets,
            layer.w13_strided_ptrs_w, layer.w13_strided_ptrs_s,
            num_experts, layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim, self.group_size)

        # Fused SiLU-and-mul into pre-allocated buffer
        intermediate = layer._buf_intermediate[:total_slots]
        torch.ops._C.silu_and_mul(intermediate, gate_up)

        # Batched w2 GEMM (down projection) — single Gemm::Run
        sorted_output = ops.awq_moe_gemm_sm70(
            intermediate, expert_offsets,
            layer.w2_strided_ptrs_w, layer.w2_strided_ptrs_s,
            num_experts, layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim, self.group_size)

        # Weighted scatter-add back
        weighted = sorted_output * sorted_weights.unsqueeze(1).to(x.dtype)
        output.index_add_(0, sorted_token_origin, weighted)
        return output

    def _apply_sorted_loop(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: sort + single CPU sync + per-expert GEMM loop."""
        num_tokens, hidden_size = x.shape
        top_k = topk_ids.shape[1]
        num_experts = layer.sm70_num_experts
        total_slots = num_tokens * top_k

        self._ensure_buffers(layer, total_slots, num_tokens)
        output = layer._buf_output[:num_tokens]
        output.zero_()
        if total_slots == 0:
            return output

        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1)
        token_origin = (
            torch.arange(num_tokens, device=x.device, dtype=torch.int64)
            .unsqueeze(1).expand(num_tokens, top_k).reshape(-1))

        sorted_order = torch.argsort(flat_ids.long(), stable=True)
        sorted_token_origin = token_origin[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        sorted_input = x[sorted_token_origin]

        sorted_expert_ids = flat_ids[sorted_order]
        expert_counts = torch.bincount(
            sorted_expert_ids.long(), minlength=num_experts)
        expert_offsets = torch.zeros(
            num_experts + 1, dtype=torch.int64, device=x.device)
        torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

        # Single CUDA sync — read offsets to CPU
        h_offsets = expert_offsets.cpu().tolist()

        sorted_output = torch.empty(
            total_slots, hidden_size, dtype=x.dtype, device=x.device)

        for e in range(num_experts):
            start, end = h_offsets[e], h_offsets[e + 1]
            if start == end:
                continue
            expert_input = sorted_input[start:end]

            w13_k_ld, w13_q_ld = layer.w13_meta_list[e]
            gate_up = ops.awq_gemm_sm70(
                expert_input, layer.w13_tm_weight[e],
                layer.w13_tm_scales[e], self.group_size,
                w13_k_ld, w13_q_ld)

            intermediate = torch.empty(
                end - start, gate_up.shape[1] // 2,
                dtype=x.dtype, device=x.device)
            torch.ops._C.silu_and_mul(intermediate, gate_up)

            w2_k_ld, w2_q_ld = layer.w2_meta_list[e]
            sorted_output[start:end] = ops.awq_gemm_sm70(
                intermediate, layer.w2_tm_weight[e],
                layer.w2_tm_scales[e], self.group_size,
                w2_k_ld, w2_q_ld)

        weighted = sorted_output * sorted_weights.unsqueeze(1).to(x.dtype)
        output.index_add_(0, sorted_token_origin, weighted)
        return output

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None
