# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## What this repo is
CosyVoice is a text-to-speech (TTS) system where an LLM generates discrete “speech tokens” from text (+ optional prompt text/audio), then a token-to-mel model and a vocoder generate waveform audio.

## Setup (local dev)
### Clone (submodules required)
This repo depends on `third_party/Matcha-TTS` as a git submodule.

```sh
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If submodules are missing:
git submodule update --init --recursive
```

### Python deps
The canonical dependency set is `requirements.txt`.

```sh
# (example using conda)
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt
```

### PYTHONPATH (Matcha)
Some entrypoints add the Matcha submodule to `sys.path` at runtime (e.g. `example.py`, `webui.py`), but if you’re importing `cosyvoice` from your own scripts you may need to add it.

PowerShell example:
```powershell
$env:PYTHONPATH = "$PWD\third_party\Matcha-TTS;$env:PYTHONPATH"
```

## Common commands
### Run a quick smoke test (inference)
Runs the sample inference script (writes `.wav` files to the repo root).

```sh
python example.py
```

### Run the Gradio web UI
`webui.py` starts a Gradio app and uses `AutoModel` under the hood.

```sh
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

### vLLM inference example
`vllm_example.py` demonstrates loading vLLM integration (CosyVoice2/3) and running repeated inference.

```sh
python vllm_example.py
```

### FastAPI server (streaming audio bytes)
Server:
```sh
python runtime/python/fastapi/server.py --port 50000 --model_dir iic/CosyVoice2-0.5B
```
Client (saves `demo.wav`):
```sh
python runtime/python/fastapi/client.py --host 127.0.0.1 --port 50000 --mode sft --tts_wav demo.wav
```

### gRPC server
Server:
```sh
python runtime/python/grpc/server.py --port 50000 --model_dir iic/CosyVoice2-0.5B
```
Client (saves `demo.wav`):
```sh
python runtime/python/grpc/client.py --host 127.0.0.1 --port 50000 --mode sft --tts_wav demo.wav
```

### Docker (deployment / runtime)
Python runtime image (see `runtime/python/Dockerfile`):
```sh
docker build -t cosyvoice:v1.0 runtime/python
```

TensorRT-LLM + Triton runtime (see `runtime/triton_trtllm/`):
```sh
cd runtime/triton_trtllm
docker compose up -d
```

### Lint (matches CI)
CI runs flake8 with specific pins and flags (see `.github/workflows/lint.yml`). Locally:

```sh
python -m pip install flake8==3.8.2 flake8-bugbear flake8-comprehensions flake8-executable flake8-pyi==20.5.0 mccabe pycodestyle==2.6.0 pyflakes==2.2.0
flake8 --max-line-length 180 --ignore B006,B008,B905,C408,E402,E731,E741,W503,W504,F401,F403,F405,F722,F841 --exclude ./third_party/,./runtime/python/grpc/cosyvoice_pb2*py
```

### Tests
There is no repo-level pytest/unittest test suite wired up in this repository. Use `python example.py` (or the FastAPI/gRPC clients) as a practical smoke test for changes.

## High-level architecture (big picture)
### Main inference entrypoints
- `cosyvoice/cli/cosyvoice.py`
  - `AutoModel(model_dir=...)` selects CosyVoice v1/v2/v3 based on which YAML exists in the model directory (`cosyvoice.yaml`, `cosyvoice2.yaml`, `cosyvoice3.yaml`).
  - Public APIs: `inference_sft`, `inference_zero_shot`, `inference_cross_lingual`, `inference_instruct` (v1), `inference_instruct2` (v2/v3), `inference_vc`.

### Inference data flow (end-to-end)
1. **Text normalization + chunking**
   - `CosyVoiceFrontEnd.text_normalize()` in `cosyvoice/cli/frontend.py` optionally uses `ttsfrd` (if installed) or `wetext`, then splits long text into manageable segments for the LLM.
2. **Tokenization + prompt feature extraction**
   - Text tokens: `CosyVoiceFrontEnd._extract_text_token()`
   - Prompt audio (when provided) yields:
     - speaker embedding via ONNX `campplus.onnx`
     - prompt speech tokens via ONNX `speech_tokenizer_v{1,2,3}.onnx`
     - prompt acoustic features via `feat_extractor` (24 kHz path)
3. **LLM generates speech tokens**
   - Orchestrated by `CosyVoiceModel.llm_job()` / `CosyVoice2Model.llm_job()` in `cosyvoice/cli/model.py`.
   - Core LLM implementations live in `cosyvoice/llm/llm.py`.
4. **Token → mel (flow matching / diffusion-style)**
   - `cosyvoice/flow/flow.py` implements token→mel inference (v1 uses cached flow; v2/v3 have causal/streaming support).
5. **Mel → waveform (vocoder / HiFiGAN)**
   - The vocoder path is wired as `hift` in `cosyvoice/cli/model.py`. HiFiGAN components are in `cosyvoice/hifigan/`.
6. **Streaming**
   - `CosyVoiceModel.tts(..., stream=True)` yields audio chunks while the LLM is still producing tokens.
   - v1 uses token overlap + mel overlap + waveform overlap windows for smooth stitching.
   - v2/v3 streaming uses fixed hop sizes and model-side lookahead (`pre_lookahead_len`) in the causal flow.

### Model packaging expectations
A `model_dir` is either a local folder under `pretrained_models/` or a ModelScope repo id; `AutoModel` will download via `modelscope.snapshot_download` when `model_dir` is not a local path.

Model directories are expected to contain:
- a YAML config (`cosyvoice*.yaml`)
- `llm.pt`, `flow.pt`, `hift.pt`
- ONNX artifacts used by the frontend (e.g. `campplus.onnx`, `speech_tokenizer_v*.onnx`)
- optional acceleration artifacts:
  - TorchScript zips for parts of the graph (`*.zip`)
  - TensorRT plans for the flow decoder estimator (`flow.decoder.estimator.*.plan`)

### Deployment wrappers
- `runtime/python/fastapi/server.py` and `runtime/python/grpc/server.py` are thin wrappers around `AutoModel` that stream raw int16 PCM bytes back to clients.
- `runtime/triton_trtllm/` provides a higher-performance Triton + TensorRT-LLM deployment path (Docker Compose driven) and includes a staged `run.sh` workflow.

### Training & data pipeline (where to look)
- Training entrypoints are under `cosyvoice/bin/` (e.g. `train.py`, `export_onnx.py`, `export_jit.py`, `average_model.py`).
- Dataset code lives under `cosyvoice/dataset/` (`dataset.py`, `processor.py`).
- Example training/inference recipes live under `examples/` (e.g. `examples/grpo/cosyvoice2/`).
