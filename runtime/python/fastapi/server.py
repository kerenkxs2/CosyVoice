# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import io
import wave
import json
import argparse
import logging
from typing import Optional, Dict, Any

logging.getLogger('matplotlib').setLevel(logging.WARNING)

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, '..', '..', '..'))

sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


class OpenAITTSRequest(BaseModel):
    # OpenAI-compatible TTS request shape (subset)
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = 'wav'  # wav|pcm
    speed: Optional[float] = 1.0


# Defaults used by /v1/audio/speech (set in __main__)
default_prompt_text: str = ''
default_spk_id: str = ''
default_prompt_speech_16k = None

# Optional voice presets (set in __main__): voice_id -> {"text": str, "speech_16k": Tensor}
voice_prompts: Dict[str, Dict[str, Any]] = {}


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


def collect_pcm16le(model_output) -> bytes:
    chunks = []
    for i in model_output:
        chunks.append((i['tts_speech'].numpy().flatten() * (2 ** 15)).astype(np.int16))
    if not chunks:
        return b''
    return np.concatenate(chunks, axis=0).tobytes()


def pcm16le_to_wav_bytes(pcm16le: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16le)
    return buf.getvalue()


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/v1/audio/voices")
async def list_voices():
    return {
        "voices": sorted(list(voice_prompts.keys())),
    }


@app.post("/v1/audio/speech")
async def openai_audio_speech(req: OpenAITTSRequest):
    # OpenAI-style TTS API used by some clients.
    # For CosyVoice2/3, we drive TTS via zero-shot using a preset prompt wav+text.
    tts_text = req.input
    speed = float(req.speed or 1.0)

    if hasattr(cosyvoice, 'inference_zero_shot'):
        if default_prompt_speech_16k is None:
            raise RuntimeError('default_prompt_speech_16k is not set; run server.py directly or provide --default_prompt_wav')

        prompt_text = default_prompt_text
        prompt_speech_16k = default_prompt_speech_16k
        if req.voice and req.voice in voice_prompts:
            prompt_text = (voice_prompts[req.voice].get('text') or '').strip() or prompt_text
            prompt_speech_16k = voice_prompts[req.voice].get('speech_16k') or prompt_speech_16k

        # If we don't have a transcript for the prompt audio, cross-lingual does not require prompt_text.
        if (not prompt_text) and hasattr(cosyvoice, 'inference_cross_lingual'):
            model_output = cosyvoice.inference_cross_lingual(
                tts_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        else:
            model_output = cosyvoice.inference_zero_shot(
                tts_text,
                prompt_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
    else:
        # Fall back to SFT if available (requires preset voice).
        model_output = cosyvoice.inference_sft(tts_text, default_spk_id, stream=False, speed=speed)

    pcm16le = collect_pcm16le(model_output)

    fmt = (req.response_format or 'wav').lower()
    if fmt == 'pcm':
        return Response(content=pcm16le, media_type='application/octet-stream')

    wav_bytes = pcm16le_to_wav_bytes(pcm16le, cosyvoice.sample_rate)
    return Response(content=wav_bytes, media_type='audio/wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--default_prompt_wav',
                        type=str,
                        default=os.path.join(REPO_ROOT, 'asset', 'zero_shot_prompt.wav'),
                        help='Default prompt wav to use for /v1/audio/speech (16k+; <=30s recommended)')
    parser.add_argument('--voices_manifest',
                        type=str,
                        default=os.path.join(REPO_ROOT, 'voice_presets', 'voices.json'),
                        help='Optional JSON mapping of voice_id -> {wav, text} for /v1/audio/speech voice selection')
    parser.add_argument('--default_prompt_text',
                        type=str,
                        default='希望你以后能够做的比我还好呦。',
                        help='Transcript of the default prompt audio')
    parser.add_argument('--default_spk_id',
                        type=str,
                        default='中文女',
                        help='Fallback preset speaker id for models that support SFT')

    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    # Globals used by the OpenAI-compatible endpoint
    default_prompt_text = args.default_prompt_text
    default_spk_id = args.default_spk_id
    default_prompt_speech_16k = load_wav(args.default_prompt_wav, 16000)

    # Optional voice presets
    try:
        if os.path.exists(args.voices_manifest):
            with open(args.voices_manifest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for voice_id, spec in data.items():
                    if not isinstance(spec, dict):
                        continue
                    wav = spec.get('wav') or spec.get('path')
                    if not wav:
                        continue
                    wav_path = wav if os.path.isabs(wav) else os.path.join(REPO_ROOT, wav)
                    if not os.path.exists(wav_path):
                        continue
                    voice_prompts[str(voice_id)] = {
                        'text': spec.get('text', '') or default_prompt_text,
                        'speech_16k': load_wav(wav_path, 16000),
                    }
    except Exception as e:
        logging.warning(f'failed to load voices_manifest: {e}')

    uvicorn.run(app, host="0.0.0.0", port=args.port)
