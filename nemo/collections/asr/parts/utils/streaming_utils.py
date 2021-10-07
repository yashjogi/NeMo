# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import copy

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import LengthsType, NeuralType


class AudioFeatureIterator(IterableDataset):
    def __init__(self, samples, frame_len, preprocessor, device):
        self._samples = samples
        self._frame_len = frame_len
        self._start = 0
        self.output = True
        self.count = 0
        timestep_duration = preprocessor._cfg['window_stride']
        self._feature_frame_len = frame_len / timestep_duration
        audio_signal = torch.from_numpy(self._samples).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([self._samples.shape[0]]).to(device)
        self._features, self._features_len = preprocessor(input_signal=audio_signal, length=audio_signal_len,)
        self._features = self._features.squeeze()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._feature_frame_len)
        if last <= self._features_len[0]:
            frame = self._features[:, self._start : last].cpu()
            self._start = last
        else:
            frame = np.zeros([self._features.shape[0], int(self._feature_frame_len)], dtype='float32')
            samp_len = self._features_len[0] - self._start
            frame[:, 0:samp_len] = self._features[:, self._start : self._features_len[0]].cpu()
            self.output = False
        self.count += 1
        return frame


def speech_collate_fn(batch):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal = []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths


# simple data layer to pass buffered frames of audio samples
class AudioBuffersDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._buf_count == len(self.signal):
            raise StopIteration
        self._buf_count += 1
        return (
            torch.as_tensor(self.signal[self._buf_count - 1], dtype=torch.float32),
            torch.as_tensor(self.signal_shape[1], dtype=torch.int64),
        )

    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1


def get_samples(audio_file, target_sr=16000):
    with sf.SoundFile(audio_file, 'r') as f:
        dtype = 'int16'
        sample_rate = f.samplerate
        samples = f.read(dtype=dtype)
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
        samples = samples.astype('float32') / 32768
        samples = samples.transpose()
        return samples


class FeatureFrameBufferer:
    """
    Class to append each feature frame to a buffer and return
    an array of buffers.
    """

    def __init__(self, asr_model, frame_len=1.6, batch_size=4, total_buffer=4.0):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
        self.asr_model = asr_model
        self.sr = asr_model._cfg.sample_rate
        self.frame_len = frame_len
        timestep_duration = asr_model._cfg.preprocessor.window_stride
        self.n_frame_len = int(frame_len / timestep_duration)

        total_buffer_len = int(total_buffer / timestep_duration)
        # print("total_buffer_len", total_buffer_len)
        self.n_feat = asr_model._cfg.preprocessor.features
        self.buffer = np.ones([self.n_feat, total_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL

        self.batch_size = batch_size

        self.signal_end = False
        self.frame_reader = None
        self.feature_buffer_len = total_buffer_len

        self.feature_buffer = (
            np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.frame_buffers = []
        self.buffered_features_size = 0
        self.reset()
        self.buffered_len = 0

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = np.ones(shape=self.buffer.shape, dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.prev_char = ''
        self.unmerged = []
        self.frame_buffers = []
        self.buffered_len = 0
        self.feature_buffer = (
            np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )

    def get_batch_frames(self):
        if self.signal_end:
            return []
        batch_frames = []
        for frame in self.frame_reader:
            batch_frames.append(np.copy(frame))
            if len(batch_frames) == self.batch_size:
                return batch_frames
        self.signal_end = True

        return batch_frames

    def get_frame_buffers(self, frames):
        # Build buffers for each frame
        self.frame_buffers = []
        for frame in frames:
            self.buffer[:, : -self.n_frame_len] = self.buffer[:, self.n_frame_len :]
            self.buffer[:, -self.n_frame_len :] = frame
            self.buffered_len += frame.shape[1]
            self.frame_buffers.append(np.copy(self.buffer))
        return self.frame_buffers

    def set_frame_reader(self, frame_reader):
        self.frame_reader = frame_reader
        self.signal_end = False

    def _update_feature_buffer(self, feat_frame):
        self.feature_buffer[:, : -feat_frame.shape[1]] = self.feature_buffer[:, feat_frame.shape[1] :]
        self.feature_buffer[:, -feat_frame.shape[1] :] = feat_frame
        self.buffered_features_size += feat_frame.shape[1]

    def get_norm_consts_per_frame(self, batch_frames):
        norm_consts = []
        for i, frame in enumerate(batch_frames):
            self._update_feature_buffer(frame)
            mean_from_buffer = np.mean(self.feature_buffer, axis=1)
            stdev_from_buffer = np.std(self.feature_buffer, axis=1)
            norm_consts.append((mean_from_buffer.reshape(self.n_feat, 1), stdev_from_buffer.reshape(self.n_feat, 1)))
        return norm_consts

    def normalize_frame_buffers(self, frame_buffers, norm_consts):
        CONSTANT = 1e-5
        for i, frame_buffer in enumerate(frame_buffers):
            frame_buffers[i] = (frame_buffer - norm_consts[i][0]) / (norm_consts[i][1] + CONSTANT)

    def get_buffers_batch(self, normalize=True):
        batch_frames = self.get_batch_frames()

        while len(batch_frames) > 0:
            frame_buffers = self.get_frame_buffers(batch_frames)
            if len(frame_buffers) == 0:
                continue
            if normalize:
                norm_consts = self.get_norm_consts_per_frame(batch_frames)
                self.normalize_frame_buffers(frame_buffers, norm_consts)
            return frame_buffers
        return []


# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameBatchASR:
    """
    class for streaming frame-based ASR use reset() method to reset FrameASR's
    state call transcribe(frame) to do ASR on contiguous signal's frames
    """

    def __init__(
        self, asr_model, vad_model, frame_len=1.6, total_buffer=4.0, vad_context_window=0.63, batch_size=4,
    ):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''

        self.frame_bufferer = FeatureFrameBufferer(
            asr_model=asr_model, frame_len=frame_len, batch_size=batch_size, total_buffer=total_buffer
        )
        self.vad_frame_bufferer = FeatureFrameBufferer(
            asr_model=vad_model, frame_len=frame_len, batch_size=batch_size, total_buffer=vad_context_window
        )

        self.asr_model = asr_model
        self.vad_model = vad_model

        self.batch_size = batch_size
        self.all_logits = []
        self.all_preds = []
        self.all_vad_preds = []

        self.unmerged = []

        self.blank_id = len(asr_model.decoder.vocabulary)
        self.tokenizer = asr_model.tokenizer
        self.toks_unmerged = []
        self.frame_buffers = []
        self.reset()
        cfg = copy.deepcopy(asr_model._cfg)
        vad_cfg = copy.deepcopy(vad_model._cfg)
        self.frame_len = frame_len
        OmegaConf.set_struct(cfg.preprocessor, False)
        OmegaConf.set_struct(vad_cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        self.raw_preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self.raw_preprocessor.to(asr_model.device)

        self.vad_preprocessor = EncDecClassificationModel.from_config_dict(vad_cfg.preprocessor)
        self.vad_preprocessor.to(vad_model.device)

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        self.prev_char = ''
        self.unmerged = []
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=self.batch_size, collate_fn=speech_collate_fn)

        self.vad_data_layer = AudioBuffersDataLayer()
        self.vad_data_loader = DataLoader(self.vad_data_layer, batch_size=self.batch_size, collate_fn=speech_collate_fn)

        self.all_logits = []
        self.all_preds = []
        self.toks_unmerged = []

        self.vad_pred_buffer = [0] * 25

        self.frame_buffers = []
        self.frame_bufferer.reset()

        self.vad_frame_buffers = []
        self.vad_frame_bufferer.reset()


    def read_audio_file(self, audio_filepath: str, delay, model_stride_in_secs):
        samples = get_samples(audio_filepath)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        vad_frame_reader = AudioFeatureIterator(samples, self.frame_len, self.vad_preprocessor, self.vad_model.device)
        self.set_frame_reader(frame_reader, None)
        self.set_frame_reader(None, vad_frame_reader)

    def set_frame_reader(self, frame_reader, vad_frame_reader):
        # Fei todo This is super ugly. change this 
        if frame_reader:
            self.frame_bufferer.set_frame_reader(frame_reader)
        if vad_frame_reader:
            self.vad_frame_bufferer.set_frame_reader(vad_frame_reader)

    @torch.no_grad()
    def infer_logits(self):
        frame_buffers = self.frame_bufferer.get_buffers_batch()
        vad_frame_buffers = self.vad_frame_bufferer.get_buffers_batch(normalize=False)

        while len(frame_buffers) > 0:
            self.frame_buffers += frame_buffers[:]
            self.data_layer.set_signal(frame_buffers[:])


            self.vad_frame_buffers += vad_frame_buffers[:]
            self.vad_data_layer.set_signal(vad_frame_buffers[:])

            self._get_batch_preds()

            frame_buffers = self.frame_bufferer.get_buffers_batch()
            vad_frame_buffers = self.vad_frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self):
        device = self.asr_model.device


        for (batch, vad_batch) in zip(iter(self.data_loader), iter(self.vad_data_loader)):
            
            feat_signal, feat_signal_len = batch
            vad_feat_signal, vad_feat_signal_len = vad_batch

            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            vad_feat_signal, vad_feat_signal_len = vad_feat_signal.to(device), vad_feat_signal_len.to(device)
            # print("===", feat_signal.shape, vad_feat_signal.shape )
            
            log_probs, encoded_len, predictions = self.asr_model(
                processed_signal=feat_signal, processed_signal_length=feat_signal_len
            )
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())

            # VAD batch inference
            logits = self.vad_model(
                processed_signal=vad_feat_signal, processed_signal_length=vad_feat_signal_len
            )
            vad_probs = torch.softmax(logits, dim=-1)
            vad_pred = vad_probs[:, 1] 
            self.all_vad_preds.extend(vad_pred)

            del log_probs
            del encoded_len
            del predictions
            del logits

    def _postprocessing_gate_vad(self, vad_pred, threshold=0.4, min_num_speech=0, last_k=25):
        """
        simpliest case. if at least ONE 1 in the buffer, this buffer is speech
        need to find better decision method, this is just an example here
        """
        self.vad_pred_buffer = self.vad_pred_buffer[1:]
        # print(vad_pred)
        if vad_pred > threshold:
            self.vad_pred_buffer.append(1)
        else:
            self.vad_pred_buffer.append(0)
            
        print(self.vad_pred_buffer)

        if 1 in self.vad_pred_buffer[-last_k:]:
             buffer_vad_decision = 1
        else:
             buffer_vad_decision = 0

        return buffer_vad_decision


    def transcribe(
        self, 
        tokens_per_chunk: int, 
        delay: int, 
        vad_gate=False, 
        threshold=0.4,
        last_k=25
    ):
        self.infer_logits()
        self.unmerged = []

        for i in range(len(self.all_preds)):
            buffer_vad_decision = self._postprocessing_gate_vad(self.all_vad_preds[i],threshold=threshold, last_k=last_k)
            if vad_gate:
                if buffer_vad_decision > 0:
                    # print("add")
                    decoded = self.all_preds[i].tolist()
                    print("decoded", decoded)
                    self.unmerged += decoded[len(decoded) - 1 - delay : len(decoded) - 1 - delay + tokens_per_chunk]
                    print("unmerged", self.unmerged)
                else:
                    # print("filtered out")
                    continue
            else:
                decoded = self.all_preds[i].tolist()
                self.unmerged += decoded[len(decoded) - 1 - delay : len(decoded) - 1 - delay + tokens_per_chunk]

        return self.greedy_merge(self.unmerged)

    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = self.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis
