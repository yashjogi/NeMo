# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

import copy
import json
import math
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
# from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.streaming_utils import *
from nemo.utils import logging

can_gpu = torch.cuda.is_available()


def prepare_for_streaming(buffer_len, samples, chunk_len_in_secs=0.16):
    # WER calculation
    # Collect all buffers from the audio file
    sampbuffer = np.zeros([buffer_len], dtype=np.float32)
    sample_rate = 16000
    chunk_len = int(sample_rate*chunk_len_in_secs)
    chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
    buffer_list = []
    for chunk in chunk_reader:
        sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
        sampbuffer[-chunk_len:] = chunk
        buffer_list.append(np.array(sampbuffer))

    return buffer_list

def streaming_vad_before_asr(buffer_list, vad_model):
    vad_buffer_len_in_secs=0.63
    chunk_len_in_secs = 0.16
    vad_decoder = VadChunkBufferDecoder(vad_model, 
                                        chunk_len_in_secs=chunk_len_in_secs, 
                                        vad_buffer_len_in_secs=vad_buffer_len_in_secs, 
                                        patience=25)
    threshold=0.4
    streaming_vad_output, speech_segments = vad_decoder.transcribe_buffers(buffer_list, plot=False,threshold=threshold)
    
    speech_segments = [list(i) for i in speech_segments]
    speech_segments.sort(key=lambda x: x[0])

    total_duration_asr = 0
    for i in range(len(speech_segments)): 
        speech_segment = speech_segments[i]
        total_duration_asr += (speech_segment[1] - speech_segment[0]+ 0.16*4)

    return speech_segments, total_duration_asr, streaming_vad_output


def get_wer_feat(mfst, asr, frame_len, tokens_per_chunk, delay, preprocessor_cfg, model_stride_in_secs, device, 
                vad_before_asr, vad_buffer_len=None, vad_model=None):
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    with open(mfst, "r") as mfst_f:
        for l in mfst_f:
            asr.reset()
            row = json.loads(l.strip())
            if vad_before_asr:
                 # ugly way and will need refactor
                samples = get_samples(row['audio_filepath'])
                buffer_list = prepare_for_streaming(vad_buffer_len, samples)
            
                speech_segments, total_duration_asr, streaming_vad_output = streaming_vad_before_asr(buffer_list, vad_model)
                print("Speech_segments:", speech_segments)
                print("Total duration after VAD to be inferred by ASR: ", total_duration_asr)

                final_hyp = " "
                for i in range(len(speech_segments)): 
                    asr.reset()
                    speech_segment = speech_segments[i] 
                    offset = speech_segment[0] -0.16 * 4
                    duration = speech_segment[1] - speech_segment[0]+ 0.16*4

                    asr.read_audio_file(row['audio_filepath'], offset, duration, delay, model_stride_in_secs)
                    hyp = asr.transcribe(tokens_per_chunk, delay) + " "
                    # there should be some better method to merge the hyps of segments.
                    final_hyp += hyp

                final_hyp = final_hyp[1:-1]
                print(final_hyp)
                hyps.append(final_hyp)
                refs.append(row['text'])

            else:
                asr.read_audio_file(row['audio_filepath'], offset=0, duration=None, 
                                    delay=delay, model_stride_in_secs=model_stride_in_secs)
                hyp = asr.transcribe(tokens_per_chunk, delay)
                print("hyp is", hyp)
                hyps.append(hyp)
                refs.append(row['text'])


    wer = word_error_rate(hypotheses=hyps, references=refs)
    print(wer)
    return hyps, refs, wer



def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to asr model .nemo file",
    )
    parser.add_argument(
        "--vad_model", type=str, required=False, help="Path to asr model .nemo file",
    )
    parser.add_argument("--test_manifest", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--total_buffer_in_secs",
        type=float,
        default=4.0,
        help="Length of buffer (chunk + left and right padding) in seconds ",
    )
    parser.add_argument("--chunk_len_in_ms", type=int, default=1600, help="Chunk length in milliseconds")
    parser.add_argument("--output_path", type=str, help="path to output file", default=None)
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models",
    )


    vad_before_asr = False

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    if args.vad_model:
        if args.vad_model.endswith('.nemo'):
            logging.info(f"Using local ASR model from {args.vad_model}")
            vad_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.vad_model)
        elif args.vad_model.endswith('.ckpt'):
            logging.info(f"Using local ASR model from {args.vad_model}")
            vad_model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(args.vad_model)
        else:
            logging.info(f"Using NGC cloud ASR model {args.vad_model}")
            vad_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.vad_model)

    sample_rate = 16000
    chunk_len_in_secs = 0.16
    if vad_before_asr and args.vad_model:
        vad_buffer_len_in_secs = 0.63
        
        vad_buffer_len = int(sample_rate * vad_buffer_len_in_secs)
        chunk_len = int(sample_rate*chunk_len_in_secs)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        vad_model.eval()
        vad_model = vad_model.to(vad_model.device)
        

    cfg = copy.deepcopy(asr_model._cfg)
 
    OmegaConf.set_struct(cfg.preprocessor, False)
    

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

    # note VAD model itself is none normalized
    
    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)


    feature_stride = cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * args.model_stride
    total_buffer = args.total_buffer_in_secs

    chunk_len = args.chunk_len_in_ms / 1000

    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)

    print("tokens_per_chunk, mid_delay", tokens_per_chunk, mid_delay)

    frame_asr = FrameBatchASR(
        asr_model=asr_model, 
        frame_len=chunk_len, 
        total_buffer=args.total_buffer_in_secs, 
        batch_size=args.batch_size,
    )
    if vad_before_asr:
        hyps, refs, wer = get_wer_feat(
            args.test_manifest,
            frame_asr,
            chunk_len,
            tokens_per_chunk,
            mid_delay,
            cfg.preprocessor,
            model_stride_in_secs,
            asr_model.device,
            vad_before_asr,
            vad_buffer_len,
            vad_model
        )
    else:
         hyps, refs, wer = get_wer_feat(
            args.test_manifest,
            frame_asr,
            chunk_len,
            tokens_per_chunk,
            mid_delay,
            cfg.preprocessor,
            model_stride_in_secs,
            asr_model.device,
            vad_before_asr,
        )
    logging.info(f"WER is {round(wer, 2)} when decoded with a delay of {round(mid_delay*model_stride_in_secs, 2)}s")

    if args.output_path is not None:

        fname = (
            os.path.splitext(os.path.basename(args.asr_model))[0]
            + "_"
            + os.path.splitext(os.path.basename(args.test_manifest))[0]
            + "_"
            + str(args.chunk_len_in_ms)
            + "_"
            + str(int(total_buffer * 1000))
            + ".json"
        )
        hyp_json = os.path.join(args.output_path, fname)
        os.makedirs(args.output_path, exist_ok=True)
        with open(hyp_json, "w") as out_f:
            for i, hyp in enumerate(hyps):
                record = {
                    "pred_text": hyp,
                    "text": refs[i],
                    "wer": round(word_error_rate(hypotheses=[hyp], references=[refs[i]]) * 100, 2),
                }
                out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
