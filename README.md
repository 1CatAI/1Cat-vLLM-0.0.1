# vLLM SM70 AWQ Branch

An experimental vLLM fork for **Tesla V100 / SM70** that enables
**AWQ 4-bit inference on Volta GPUs**.

Upstream vLLM AWQ kernels require **SM75+** in the default path. This branch
integrates **lmdeploy TurboMind SM70 WMMA kernels** and a set of SM70-specific
runtime fixes so that V100 can serve modern AWQ models, including dense,
MoE, and selected multimodal models.

## What this branch adds

- AWQ 4-bit support for **SM70 / Tesla V100**
- Dense and MoE AWQ execution paths on V100
- Reuse of SM70 AWQ kernels for selected compressed-tensors MoE paths
- SM70-specific Triton attention and MLA/GDN runtime fixes
- Compatibility with `torch.compile` and CUDA graphs
- OpenAI-compatible API serving through standard vLLM entrypoints

This repository is based on vLLM and keeps the original project structure and
serving APIs wherever possible.

## Validated stack

The commands in this README were validated on the following setup:

- OS: `Ubuntu 24.04.4 LTS`
- Python: `3.12.13`
- CUDA toolkit: `12.8`
- PyTorch: `2.9.1+cu128`
- Triton: `3.5.1`
- Driver: `570.211.01`
- GPU: `4 x Tesla V100-SXM2-16GB`

## Key ideas

This branch mainly changes two areas:

1. **AWQ on SM70**
   - Integrates TurboMind SM70 native `8x8x4` half-precision MMA kernels.
   - Extends vLLM AWQ quantized layers so V100 can run AWQ models that
     normally require Turing or newer.

2. **SM70 runtime fixes**
   - Triton attention tuning for Volta
   - MLA/GDN compatibility work
   - CUDA graph and `torch.compile` adjustments for this hardware class

## Current focus

This branch is primarily aimed at:

- Tesla V100 / SM70 users
- AWQ 4-bit deployment
- Qwen3 / Qwen3.5 style models
- Small-memory multi-GPU serving setups such as `4 x 16GB`

## Important runtime notes

- The **first real request is not representative** of steady-state speed.
  On V100, it is normal for the first request to spend **1 to 3 minutes**
  compiling kernels, building graphs, and warming up execution paths.
- On the validated `4 x V100 16GB` machine, the most stable runtime settings
  were:
  - `VLLM_DISABLE_PYNCCL=1`
  - `--disable-custom-all-reduce`
  - `--gpu-memory-utilization 0.80`
- For `Qwen3.5-35B-A3B-AWQ` on `4 x 16GB`, `gpu_memory_utilization=0.92`
  could start the server but was not stable for the first real request.
- Text-only serving on small cards benefits from:
  - `--skip-mm-profiling`
  - `--limit-mm-per-prompt '{"image":0,"video":0}'`

## Quick start

### 1. Install CUDA 12.8

Use the official NVIDIA repository on Ubuntu 24.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

If the machine also has CUDA 13.x installed, force build-time CUDA to 12.8:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
nvcc -V
```

### 2. Create the conda environment

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -y -n vllm-py312 python=3.12
conda activate vllm-py312
python -m pip install --upgrade pip setuptools wheel
```

### 3. Prepare the `lmdeploy` source dependency

This branch currently builds the SM70 AWQ kernels against selected
TurboMind sources from `lmdeploy`.

Initialize the pinned `lmdeploy` submodule before building:

```bash
cd /path/to/vllm

git submodule update --init --recursive lmdeploy
```

Notes:

- The submodule points to the `1CatAI/lmdeploy` fork that already contains the
  SM70 compatibility changes required by this branch.
- It is pinned to fork commit `159b0ab1db2d2496b57dbb19dd43f0d0db1c4257` for
  reproducible builds.
- If you cloned the repository without submodules, run the command above before
  building.

### 4. Install dependencies

This repository uses `requirements/` instead of a single `requirements.txt`.

```bash
cd /path/to/vllm

python -m pip install -r requirements/build.txt
python -m pip install -r requirements/cuda.txt
python -m pip install -r requirements/common.txt
```

### 5. Build from source

The validated release-style build used **24 CMake compile jobs**.

```bash
cd /path/to/vllm
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate vllm-py312

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24
export NVCC_THREADS=1

python -m pip install -e . --no-build-isolation
```

`setup.py` divides `MAX_JOBS` by `NVCC_THREADS` when CUDA builds are enabled.
If you want a true `cmake --build -j24`, keep `MAX_JOBS=24` and leave
`NVCC_THREADS=1` (or unset it).

### 6. Verify the environment

```bash
python - <<'PY'
import torch, triton, vllm, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
PY
```

### 7. Build a precompiled wheel

If you want a redistributable precompiled package instead of an editable source
install, build a wheel with the same 24-job settings:

```bash
cd /path/to/vllm
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate vllm-py312

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24
export NVCC_THREADS=1

python -m pip wheel . --no-build-isolation -w dist
```

This command writes both the main `vllm-*.whl` and its dependency wheels into
`dist/`, so `dist/` can be used as a local wheelhouse on another machine.

Install the wheel on another machine from that local wheelhouse:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -y -n vllm-py312 python=3.12
conda activate vllm-py312

python -m pip install --no-index \
  --find-links /path/to/vllm/dist \
  /path/to/vllm/dist/vllm-*.whl
```

The wheel already contains the compiled extension, so runtime installation from
the wheel does not need the `lmdeploy` source tree.

## Recommended server settings for V100 16GB

Use this baseline unless you already know you need something else:

```bash
export VLLM_DISABLE_PYNCCL=1
```

Recommended flags:

```text
--attention-backend TRITON_ATTN
--disable-custom-all-reduce
--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'
--gpu-memory-utilization 0.80
--max-model-len 512
--max-num-seqs 1
--max-num-batched-tokens 128
```

For **text-only** mode on small cards, also use:

```text
--skip-mm-profiling
--limit-mm-per-prompt '{"image":0,"video":0}'
```

## Launch examples

### Qwen3.5-27B-AWQ, TP2, text-only

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_DISABLE_PYNCCL=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-27B-AWQ \
  --served-model-name qwen35-27b \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 128 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --disable-custom-all-reduce \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 127.0.0.1 \
  --port 8201
```

### Qwen3.5-27B-AWQ, TP4, text-only

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_DISABLE_PYNCCL=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-27B-AWQ \
  --served-model-name qwen35-27b \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 128 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --disable-custom-all-reduce \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 127.0.0.1 \
  --port 8200
```

### Qwen3.5-35B-A3B-AWQ, TP4, text-only

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_DISABLE_PYNCCL=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-35B-A3B-AWQ \
  --served-model-name qwen35-35b \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 128 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --disable-custom-all-reduce \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 127.0.0.1 \
  --port 8200
```

### Qwen3.5-27B-AWQ, TP4, vision-enabled

This model includes a `vision_config`. To enable image understanding, do not
start the server in text-only mode.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_DISABLE_PYNCCL=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-27B-AWQ \
  --served-model-name qwen35-27b \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --allowed-local-media-path /path/to/media \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 127.0.0.1 \
  --port 8000
```

Notes:

- Do **not** pass `--skip-mm-profiling` for the vision-enabled setup.
- Do **not** set `--limit-mm-per-prompt '{"image":0,"video":0}'`.
- On `4 x V100 16GB`, start conservatively with `4096 / 512` before trying
  larger context lengths.

## OpenAI-compatible requests

### Text-only request

```bash
curl http://127.0.0.1:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-27b",
    "messages": [{"role":"user","content":"你好，简单介绍一下你自己。"}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": false
  }'
```

### Image request

Remote image:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-27b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {"type": "image_url", "image_url": {"url": "https://example.com/test.jpg"}}
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

Local image:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-27b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {"type": "image_url", "image_url": {"url": "file:///path/to/test.jpg"}}
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

## Benchmark method

The validated throughput numbers below used:

- OpenAI-compatible `/v1/chat/completions`
- streaming responses
- 1 warmup request
- 3 measured requests
- prompt length about `128` tokens
- output length `32` tokens

Again: **ignore the first warmup request** when judging throughput.

## Hot-run results on 4 x V100 16GB

| Model | TP | Avg TTFT | Avg Decode TPS | Avg Total TPS |
|------|---:|----------:|----------------:|--------------:|
| Qwen3.5-27B-AWQ | 2 | 169.67 ms | 50.73 | 40.99 |
| Qwen3.5-27B-AWQ | 4 | 197.93 ms | 65.76 | 47.82 |
| Qwen3.5-35B-A3B-AWQ | 4 | 216.92 ms | 81.19 | 53.45 |

## Known limits

- This branch is optimized for **SM70 / Tesla V100**, not for all hardware.
- Results above were obtained on **4 x 16GB** cards with conservative settings.
- Large context lengths and multimodal features will increase memory pressure
  quickly on 16GB GPUs.
- `Qwen3.5-35B-A3B-AWQ` on `4 x V100 16GB` was not stable with
  `gpu_memory_utilization=0.92`.
- If you want maximum context length, first confirm that your memory budget is
  sufficient for the model and modality mix.

## Repository notes

- The original upstream project is **vLLM**.
- This branch focuses on **SM70 AWQ support and V100-oriented runtime tuning**.
- Development notes and machine-specific reproduction details are kept in:
  - [RUN_V100_CUDA128.md](RUN_V100_CUDA128.md)
  - [CLAUDE.md](CLAUDE.md)

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)

## License

This repository follows the upstream vLLM license model. See [LICENSE](LICENSE).
