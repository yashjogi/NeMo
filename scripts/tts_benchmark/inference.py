# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import json
import pprint
from collections import UserList


import torch
import tqdm
import time
import numpy as np
from torch import nn
from scipy.stats import norm

from nemo.collections.tts.models import MixerTTSModel


def parse_args() -> argparse.Namespace:
    """Parses args from CLI."""
    parser = argparse.ArgumentParser(description='Mixer-TTS Benchmark')
    parser.add_argument('--manifest-path', type=str, required=True)
    parser.add_argument('--mixer-ckpt-path', type=str, required=True)
    parser.add_argument('--without-matching', action='store_true', default=False)
    parser.add_argument('--torchscript', action='store_true', default=False)
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--n-repeats', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cudnn-benchmark', action='store_true', default=False)
    return parser.parse_args()


def make_data(manifest_path, n_samples=None):
    """Makes data source and returns batching functor and total number of samples."""

    data = []
    with open(manifest_path, 'r') as f:
        for line in f.readlines():
            line_data = json.loads(line)
            data.append(dict(raw_text=line_data['text']))

    data = data[:n_samples] if n_samples is not None else data
    data.sort(key=lambda d: len(d['raw_text']), reverse=True)  # Bigger samples are more important.

    data = {k: [s[k] for s in data] for k in data[0]}
    raw_text_data = data['raw_text']
    total_samples = len(raw_text_data)

    def batching(batch_size):
        """<batch size> => <batch generator>"""
        for i in range(0, len(raw_text_data), batch_size):
            yield raw_text_data[i : i + batch_size]

    return batching, total_samples


def load_and_setup_mixer(ckpt_path: str, torchscript: bool = False) -> nn.Module:
    """Loads and setup Mixer-TTS model."""

    model = MixerTTSModel.load_from_checkpoint(ckpt_path)

    if torchscript:
        model = torch.jit.script(model)

    model.eval()

    return model


class MeasureTime(UserList):
    """Convenient class for time measurement."""

    def __init__(self, *args, cuda=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)


def main():
    """Launches TTS benchmark."""

    args = parse_args()

    batching, total_samples = make_data(args.manifest_path, args.n_samples)

    mixer = load_and_setup_mixer(args.mixer_ckpt_path, args.torchscript)
    mixer.to(args.device)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark  # noqa
    for _ in tqdm.tqdm(range(args.warmup_steps), desc='warmup'):
        with torch.no_grad():
            _ = mixer.generate_spectrogram(
                raw_texts=next(batching(args.batch_size)),
                without_matching=args.without_matching,
            )

    sample_rate = mixer.cfg.train_ds.dataset.sample_rate
    hop_length = mixer.cfg.train_ds.dataset.hop_length
    gen_measures = MeasureTime(cuda=(args.device != 'cpu'))
    all_letters, all_frames = 0, 0
    all_utterances, all_samples = 0, 0
    for _ in tqdm.trange(args.n_repeats, desc='repeats'):
        for raw_text in tqdm.tqdm(
            iterable=batching(args.batch_size),
            total=(total_samples // args.batch_size) + int(total_samples % args.batch_size),
            desc='batches',
        ):
            with torch.no_grad(), gen_measures:
                mel = mixer.generate_spectrogram(
                    raw_texts=raw_text,
                    without_matching=args.without_matching,
                )

            all_letters += sum(len(t) for t in raw_text)  # <raw text length>
            # TODO(stasbel): Actually, this need to be more precise as samples are of different length?
            all_frames += mel.size(0) * mel.size(1)  # <batch size> * <mel length>

            all_utterances += len(raw_text)  # <batch size>
            # TODO(stasbel): Same problem as above?
            # <batch size> * <mel length> * <hop length> = <batch size> * <audio length>
            all_samples += mel.size(0) * mel.size(1) * hop_length

    gm = np.sort(np.asarray(gen_measures))
    results = {
        'avg_letters/s': all_letters / gm.sum(),
        'avg_frames/s': all_frames / gm.sum(),
        'avg_latency': gm.mean(),
        'all_samples': all_samples,
        'all_utterances': all_utterances,
        'avg_RTF': all_samples / (all_utterances * gm.mean() * sample_rate),
        '90%_latency': gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std(),
        '95%_latency': gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std(),
        '99%_latency': gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std(),
    }
    pprint.pprint(results)


if __name__ == '__main__':
    main()
