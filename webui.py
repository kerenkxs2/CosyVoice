# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import argparse
import json
import glob
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa


def _concat_audio_chunks(chunks):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    # chunks are 1D float arrays
    return np.concatenate(chunks, axis=0)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# NOTE: available modes depend on the selected model.
# - Preset voice (SFT) requires speaker presets (spk2info.pt inside model_dir).
# - Instruction control differs between CosyVoice v1 (inference_instruct) and v2/v3 (inference_instruct2).
PRESET_MODE = '预训练音色'
ZERO_SHOT_MODE = '3s极速复刻'
CROSS_LINGUAL_MODE = '跨语种复刻'
INSTRUCT_MODE = '自然语言控制'

instruct_dict = {
    PRESET_MODE: '1. Select a preset voice\n2. Click Generate Audio',
    ZERO_SHOT_MODE: '1. Choose a prompt audio file (or record one). Keep it under 30s. If both are provided, the uploaded file is used first.\n2. Enter the prompt text\n3. Click Generate Audio',
    CROSS_LINGUAL_MODE: '1. Choose a prompt audio file (or record one). Keep it under 30s. If both are provided, the uploaded file is used first.\n2. Click Generate Audio',
    INSTRUCT_MODE: '1. Provide a prompt audio file (or record one)\n2. Enter the instruction text\n3. Click Generate Audio',
}

stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict.get(mode_checkbox_group, '')


def on_prompt_voice_change(prompt_voice):
    if not prompt_voice:
        return gr.update()
    spec = voice_presets.get(prompt_voice)
    if not spec:
        return gr.update()
    txt = spec.get('text') or ''
    if txt == '':
        return gr.update()
    return gr.update(value=txt)


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_voice, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    elif prompt_voice and prompt_voice in voice_presets and os.path.exists(voice_presets[prompt_voice]['wav']):
        prompt_wav = voice_presets[prompt_voice]['wav']
        if prompt_text == '':
            prompt_text = voice_presets[prompt_voice].get('text', '') or ''
    else:
        prompt_wav = None

    # Instruction control differs by model version.
    # - CosyVoice v1: inference_instruct(tts_text, spk_id, instruct_text)
    # - CosyVoice v2/v3: inference_instruct2(tts_text, instruct_text, prompt_wav)
    if mode_checkbox_group == INSTRUCT_MODE:
        if instruct_text == '':
            gr.Warning('You are using Instruction mode. Please enter the instruction text.')
            yield (cosyvoice.sample_rate, default_data)
        if hasattr(cosyvoice, 'inference_instruct2'):
            if prompt_wav is None:
                gr.Warning('You are using Instruction mode. Please provide prompt audio.')
                yield (cosyvoice.sample_rate, default_data)
            if prompt_text != '':
                gr.Info('You are using Instruction mode. Prompt text will be ignored.')
        else:
            # v1 instruct relies on preset voice
            if prompt_wav is not None or prompt_text != '':
                gr.Info('You are using Instruction mode. Prompt audio / prompt text will be ignored.')

    # Cross-lingual clone
    if mode_checkbox_group == CROSS_LINGUAL_MODE:
        if instruct_text != '':
            gr.Info('You are using Cross-lingual mode. Instruction text will be ignored.')
        if prompt_wav is None:
            gr.Warning('You are using Cross-lingual mode. Please provide prompt audio.')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('You are using Cross-lingual mode. Ensure the synthesis text and prompt text are in different languages.')

    # zero_shot / cross_lingual require prompt audio
    if mode_checkbox_group in [ZERO_SHOT_MODE, CROSS_LINGUAL_MODE]:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty. Did you forget to provide prompt audio?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sample rate {} is below {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)

    # Preset voice (SFT) requires a non-empty selection
    if mode_checkbox_group == PRESET_MODE:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Preset Voice mode. Prompt text / prompt audio / instruction text will be ignored.')
        if not sft_dropdown:
            gr.Warning('No preset voices are available for this model.')
            yield (cosyvoice.sample_rate, default_data)

    # Quick clone requires prompt_text
    if mode_checkbox_group == ZERO_SHOT_MODE:
        if prompt_text == '':
            gr.Warning('Prompt text is empty. Did you forget to enter prompt text?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using 3s Quick Clone mode. Preset voice / instruction text will be ignored.')

    if mode_checkbox_group == PRESET_MODE:
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == ZERO_SHOT_MODE:
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == CROSS_LINGUAL_MODE:
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == INSTRUCT_MODE and hasattr(cosyvoice, 'inference_instruct2'):
        logging.info('get instruct2 inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def _generate_simple(tts_text, voice, seed, speed):
    # Minimal UI path: voice is a preset prompt wav (+ optional transcript).
    if not voice or voice not in voice_presets:
        gr.Warning('Select a voice.')
        return (cosyvoice.sample_rate, default_data)

    spec = voice_presets[voice]
    prompt_wav = spec.get('wav')
    prompt_text = (spec.get('text') or '').strip()
    if not prompt_wav or not os.path.exists(prompt_wav):
        gr.Warning('Voice preset wav not found.')
        return (cosyvoice.sample_rate, default_data)

    # Non-streaming: avoids ffmpeg/pydub dependency for ADTS conversion.
    set_all_random_seed(seed)
    chunks = []

    # If we don't have a transcript for the prompt audio, fall back to cross-lingual,
    # which does not require prompt_text.
    if prompt_text == '' and hasattr(cosyvoice, 'inference_cross_lingual'):
        for out in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=False, speed=speed):
            chunks.append(out['tts_speech'].numpy().flatten())
    else:
        for out in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, speed=speed):
            chunks.append(out['tts_speech'].numpy().flatten())

    audio = _concat_audio_chunks(chunks)
    return (cosyvoice.sample_rate, audio)


def main():
    with gr.Blocks() as demo:
        gr.Markdown(f"### CosyVoice (model: `{args.model_dir}`)")

        if args.simple:
            # Minimal "type text + pick voice" UI
            tts_text = gr.Textbox(label="Text", lines=2, value="Hello! This is a demo of CosyVoice text-to-speech.")
            voice_choices = sorted(list(voice_presets.keys()))
            voice = gr.Dropdown(choices=voice_choices, label='Voice', value=(voice_choices[0] if voice_choices else None))
            speed = gr.Number(value=1, label="Speed", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Row():
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random seed")
            generate_button = gr.Button("Generate")
            audio_output = gr.Audio(label="Audio", autoplay=True, streaming=False)

            seed_button.click(generate_seed, inputs=[], outputs=seed)
            generate_button.click(_generate_simple, inputs=[tts_text, voice, seed, speed], outputs=[audio_output])
        else:
            # Full demo UI
            gr.Markdown("#### Enter the text to synthesize, choose an inference mode, and follow the steps below")

            tts_text = gr.Textbox(label="Synthesis text", lines=1, value="Hello! This is a demo of CosyVoice text-to-speech.")

            available_choices = []
            if supports_preset:
                available_choices.append(('Preset voice', PRESET_MODE))
            available_choices.extend([
                ('3s quick clone', ZERO_SHOT_MODE),
                ('Cross-lingual clone', CROSS_LINGUAL_MODE),
            ])
            if supports_instruct:
                available_choices.append(('Instruction control', INSTRUCT_MODE))

            default_mode = available_choices[0][1] if len(available_choices) else ZERO_SHOT_MODE

            with gr.Row():
                mode_checkbox_group = gr.Radio(
                    choices=available_choices,
                    label='Inference mode',
                    value=default_mode,
                )
                instruction_text = gr.Text(label="Steps", value=instruct_dict.get(default_mode, ''), scale=0.5)
                sft_dropdown = gr.Dropdown(choices=sft_spk, label='Preset voice', value=(sft_spk[0] if supports_preset else None), scale=0.25, visible=supports_preset)
                stream = gr.Radio(choices=stream_mode_list, label='Streaming', value=stream_mode_list[0][1])
                speed = gr.Number(value=1, label="Speed (non-streaming only)", minimum=0.5, maximum=2.0, step=0.1)
                with gr.Column(scale=0.25):
                    seed_button = gr.Button(value="\U0001F3B2")
                    seed = gr.Number(value=0, label="Random seed")

            with gr.Row():
                prompt_voice = gr.Dropdown(
                    choices=voice_preset_choices,
                    label='Prompt voice preset',
                    value=(voice_preset_choices[0] if len(voice_preset_choices) else ''),
                    visible=(len(voice_preset_choices) > 1),
                )
                prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Prompt audio file (sample rate >= 16 kHz)')
                prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio')
            prompt_text = gr.Textbox(label="Prompt text", lines=1, placeholder="Enter prompt text (should match the prompt audio).", value='')
            instruct_text = gr.Textbox(label="Instruction text", lines=1, placeholder="Enter instruction text.", value='')

            generate_button = gr.Button("Generate Audio")

            # NOTE: streaming=True triggers ffmpeg/pydub on Windows; keep it off by default.
            audio_output = gr.Audio(label="Generated audio", autoplay=True, streaming=False)

            seed_button.click(generate_seed, inputs=[], outputs=seed)
            prompt_voice.change(fn=on_prompt_voice_change, inputs=[prompt_voice], outputs=[prompt_text])
            generate_button.click(generate_audio,
                                  inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_voice, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                          seed, stream, speed],
                                  outputs=[audio_output])
            mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])

    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--simple',
                        action='store_true',
                        help='Run a minimal UI (text + voice dropdown only)')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    def _load_voice_presets():
        presets = {}
        preset_dir = os.path.join(ROOT_DIR, 'voice_presets')
        manifest = os.path.join(preset_dir, 'voices.json')
        if os.path.exists(manifest):
            try:
                with open(manifest, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for name, spec in data.items():
                        if not isinstance(spec, dict):
                            continue
                        wav = spec.get('wav') or spec.get('path')
                        if not wav:
                            continue
                        wav_path = wav if os.path.isabs(wav) else os.path.join(ROOT_DIR, wav)
                        if os.path.exists(wav_path):
                            presets[str(name)] = {'wav': wav_path, 'text': spec.get('text', '') or ''}
            except Exception as e:
                logging.warning(f'failed to load voice presets manifest: {e}')

        # Also accept loose wavs dropped into voice_presets/
        for wav_path in glob.glob(os.path.join(preset_dir, '*.wav')):
            name = os.path.splitext(os.path.basename(wav_path))[0]
            presets.setdefault(name, {'wav': wav_path, 'text': ''})

        return presets

    voice_presets = _load_voice_presets()
    voice_preset_choices = [''] + sorted(voice_presets.keys())

    sft_spk = cosyvoice.list_available_spks()
    supports_preset = len([s for s in sft_spk if s]) > 0
    supports_instruct = hasattr(cosyvoice, 'inference_instruct') or hasattr(cosyvoice, 'inference_instruct2')

    if not supports_preset:
        # Keep dropdown stable, but it will be hidden when supports_preset is False.
        sft_spk = ['']

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
