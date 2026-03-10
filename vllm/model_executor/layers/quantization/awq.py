# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Union

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_name(self) -> "QuantizationMethods":
        return "awq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # SM70 uses TurboMind s884h kernels; SM75+ uses the native AWQ kernel.
        return 70

    @staticmethod
    def _is_sm70_available() -> bool:
        """Check if current CUDA device is SM70 (V100).

        SM70 needs TurboMind kernels for MoE since Marlin requires SM75+.
        """
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap[0] == 7 and cap[1] == 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # SM70 (V100): use TurboMind GEMM kernels for MoE,
            # since Marlin requires SM75+.
            if self._is_sm70_available():
                # SM70 (V100): TurboMind s884h kernels require:
                #   K % 8 == 0, N % 8 == 0, K % group_size == 0
                # No requirement on (K/group_size) % 8.
                moe_cfg = layer.moe_config
                hidden = moe_cfg.hidden_dim
                inter = moe_cfg.intermediate_size_per_partition
                gs = self.group_size
                sm70_compatible = (
                    gs in (32, 64, 128)
                    and hidden % gs == 0
                    and inter % gs == 0
                    and hidden % 8 == 0
                    and inter % 8 == 0
                )
                if sm70_compatible:
                    from .awq_sm70_moe import AWQSM70MoEMethod

                    return AWQSM70MoEMethod(
                        weight_bits=self.weight_bits,
                        group_size=self.group_size,
                        zero_point=self.zero_point,
                        moe=moe_cfg,
                    )
                else:
                    logger.warning_once(
                        f"Layer '{prefix}' MoE dimensions incompatible "
                        "with SM70 TurboMind kernels "
                        f"(hidden={hidden}, inter={inter}, "
                        f"group_size={gs}). "
                        "Falling back to MoeWNA16 kernels."
                    )
                    from .moe_wna16 import MoeWNA16Config

                    config = {
                        "quant_method": "awq",
                        "bits": self.weight_bits,
                        "group_size": self.group_size,
                        "zero_point": self.zero_point,
                        "lm_head": False,
                        "modules_to_not_convert": self.modules_to_not_convert,
                    }
                    return MoeWNA16Config.from_config(
                        config
                    ).get_quant_method(layer, prefix)

            # Lazy import to avoid circular import.
            from .awq_marlin import AWQMarlinConfig
            from .utils.marlin_utils import check_moe_marlin_supports_layer

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                config = {
                    "quant_method": "awq",
                    "bits": self.weight_bits,
                    "group_size": self.group_size,
                    "zero_point": self.zero_point,
                    "lm_head": False,
                    "modules_to_not_convert": self.modules_to_not_convert,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix
                )
            marlin_compatible_config_dict = {
                "quant_method": "awq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "lm_head": False,
                "modules_to_not_convert": self.modules_to_not_convert,
            }
            awq_marlin_config = AWQMarlinConfig.from_config(
                marlin_compatible_config_dict
            )
            return awq_marlin_config.get_quant_method(layer, prefix)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def maybe_update_config(self, model_name: str, revision: str | None = None):
        if self.modules_to_not_convert:
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        layers = {param_name.rsplit(".", 1)[0] for param_name in metadata}
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_to_not_convert = list(layers - quant_layers)


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # SM70: eagerly prepare TurboMind weights at load time
        # (not lazily in apply()) so torch.compile can trace the forward.
        if (
            layer.qweight.is_cuda
            and hasattr(torch.ops._C, "awq_sm70_prepare")
            and self.quant_config.group_size in (32, 64, 128)
        ):
            cap = torch.cuda.get_device_capability(layer.qweight.device)
            if cap[0] == 7 and cap[1] == 0:
                tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
                    layer.qweight, layer.scales, layer.qzeros,
                    self.quant_config.group_size,
                )
                layer._awq_sm70_weight = tm_weight
                layer._awq_sm70_scales = tm_scales
                layer._awq_sm70_k_ld = int(meta[0])
                layer._awq_sm70_q_ld = int(meta[1])
                layer._awq_sm70_prepared = True
                # Free original AWQ tensors once TM tensors are ready.
                # This significantly lowers residency for large dense models
                # (e.g. Qwen3.5-27B) and helps SM70 single-GPU startup.
                layer.qweight = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int32, device=tm_weight.device),
                    requires_grad=False,
                )
                layer.qzeros = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int32, device=tm_weight.device),
                    requires_grad=False,
                )
                layer.scales = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.float16, device=tm_weight.device),
                    requires_grad=False,
                )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = x.shape[-1]
        reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

        if getattr(layer, "_awq_sm70_prepared", False):
            out_shape = x.shape[:-1] + (layer._awq_sm70_weight.shape[-1] * 8,)
            out = ops.awq_gemm_sm70(
                reshaped_x,
                layer._awq_sm70_weight,
                layer._awq_sm70_scales,
                group_size,
                layer._awq_sm70_k_ld,
                layer._awq_sm70_q_ld,
            )
        elif FP16_MATMUL_HEURISTIC_CONDITION:
            out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
