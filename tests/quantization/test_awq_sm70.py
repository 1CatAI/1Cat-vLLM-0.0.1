# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.awq import (
    AWQConfig,
    _should_use_awq_sm70,
)
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig


def test_awq_min_capability_sm70():
    assert AWQConfig.get_min_capability() == 70
    assert AWQMarlinConfig.get_min_capability() == 75


def test_awq_sm70_dispatch_cpu():
    x = torch.zeros(1, 1)
    assert _should_use_awq_sm70(x) is False
