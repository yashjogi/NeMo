#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pyaudio as pa
import argparse
import os, time
import nemo
import nemo.collections.asr as nemo_asr
import soundfile as sf
from pyannote.metrics.diarization import DiarizationErrorRate

from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
import librosa
import ipdb
from datetime import datetime
# from datetime import datetime as datetime_sub

### From speaker_diarize.py
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE, write_txt, get_uniq_id_from_audio_path, WER_TS
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map, perform_diarization, write_rttm2manifest, get_DER
from nemo.collections.asr.parts.utils.speaker_utils import get_contiguous_stamps, merge_stamps, labels_to_pyannote_object, rttm_to_labels, labels_to_rttmfile
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.collections.asr.models import ClusteringDiarizer
from sklearn.preprocessing import OneHotEncoder
from nemo.collections.asr.parts.utils.nmse_clustering import (
# from nmse_clustering_enhanced import (
    NMESC,
    _SpectralClustering,
    getEnhancedSpeakerCount,
    COSclustering,
    getCosAffinityMatrix,
    getAffinityGraphMat,
    getLaplacian,
    getLamdaGaplist,
    eigDecompose,
)

from nemo.core.config import hydra_runner
from nemo.utils import logging
import hydra
from typing import List, Optional, Dict
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import copy
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.utils import logging, model_utils
import torch
from torch.utils.data import DataLoader
import math

from collections import Counter
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# For streaming ASR
from nemo.core.classes import IterableDataset
from torch.utils.data import DataLoader
import math
from sklearn.manifold import TSNE


TOKEN_OFFSET = 100

import contextlib
import json
import os

import editdistance
from sklearn.model_selection import ParameterGrid

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import logging
# import nemo.scripts.asr_language_modeling.ngram_lm.kenlm_utils as kenlm_utils
from ctcdecode import OnlineCTCBeamDecoder, CTCBeamDecoder


seed_everything(42)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%2.2fms %r'%((te - ts) * 1000, method.__name__))
            pass
        return result
    return timed


def isOverlap(rangeA, rangeB):
    start1, end1 = rangeA
    start2, end2 = rangeB
    return end1 > start2 and end2 > start1

def getOverlapRange(rangeA, rangeB):
    assert isOverlap(rangeA, rangeB)
    return [ max(rangeA[0], rangeB[0]), min(rangeA[1], rangeB[1])]


def combine_overlaps(ranges):
    return reduce(
        lambda acc, el: acc[:-1:] + [(min(*acc[-1], *el), max(*acc[-1], *el))]
            if acc[-1][1] >= el[0] - 1
            else acc + [el],
        ranges[1::],
        ranges[0:1],
    )

def getMergedRanges(label_list_A, label_list_B):
    if label_list_A == [] and label_list_B != []:
        return label_list_B
    elif label_list_A != [] and label_list_B == []:
        return label_list_A
    else:
        label_list_A = [ [fl2int(x[0]), fl2int(x[1])] for x in label_list_A] 
        label_list_B = [ [fl2int(x[0]), fl2int(x[1])] for x in label_list_B] 

        combined = combine_overlaps(label_list_A + label_list_B)

        return [ [int2fl(x[0]), int2fl(x[1])] for x in combined ]

def getSubRangeList(target_range: List, source_list: List) -> List:
    if target_range == []:
        return []
    else:
        out_range_list = []
        for s_range in source_list:
            if isOverlap(s_range, target_range):
                ovl_range = getOverlapRange(s_range, target_range)
                out_range_list.append(ovl_range)
        return out_range_list 

def fl2int(x):
    return int(x*100)

def int2fl(x):
    return round(float(x/100.0), 2)


def getVADfromRTTM(rttm_fullpath):
    out_list = []
    with open(rttm_fullpath, 'r') as rttm_stamp:
        rttm_stamp_list = rttm_stamp.readlines()
        for line in rttm_stamp_list:
            stt = float(line.split()[3])
            end = float(line.split()[4]) + stt
            out_list.append([stt, end])
    return out_list

def get_partial_ref_labels(pred_labels, ref_labels):
    last_pred_time = float(pred_labels[-1].split()[1])
    ref_labels_out = []
    for label in ref_labels:
        start, end, speaker = label.split()
        start, end = float(start), float(end)
        if last_pred_time <= start:
            pass
        elif start < last_pred_time <= end:
            label = f"{start} {last_pred_time} {speaker}"
            ref_labels_out.append(label) 
        elif end < last_pred_time:
            ref_labels_out.append(label) 
    return ref_labels_out 

def read_wav(audio_file):
    with sf.SoundFile(audio_file, 'r') as f:
        sample_rate = f.samplerate
        samples = f.read(dtype='float32')
    samples = samples.transpose()
    return sample_rate, samples


def load_ASR_model(ASR_model_name):
    # Preserve a copy of the full config
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(ASR_model_name)
    cfg = copy.deepcopy(asr_model._cfg)
    print(OmegaConf.to_yaml(cfg))

    # Make config overwrite-able
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    # cfg.preprocessor.normalize = normalization

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)
    
    # Set model to inference mode
    asr_model.eval();
    asr_model = asr_model.to(asr_model.device)

    return cfg, asr_model
from nemo.core.classes import IterableDataset

def speech_collate_fn(batch):
    """collate batch of audio sig, audio len
    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """

    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
   
    
    audio_signal= []
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

# simple data layer to pass audio signal
class AudioBuffersDataLayer(IterableDataset):
    def __init__(self):
        super().__init__()

        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._buf_count == len(self.signal) :
            raise StopIteration
        self._buf_count +=1
        return torch.as_tensor(self.signal[self._buf_count-1], dtype=torch.float32), \
               torch.as_tensor(self.signal_shape[0], dtype=torch.int64)
        
    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1

class AudioChunkIterator():
    def __init__(self, samples, chunk_len_in_secs, sample_rate):
        self._samples = samples
        self._chunk_len = chunk_len_in_secs*sample_rate
        self._start = 0
        self.output=True
   
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start: last]
            self._start = last
        else:
            chunk = np.zeros([int(self._chunk_len)], dtype='float32')
            samp_len = len(self._samples) - self._start
            chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            self.output = False
   
        return chunk

class ChunkBufferDecoder:

    def __init__(self,asr_model, stride, chunk_len_in_secs=1, buffer_len_in_secs=3):
        self.asr_model = asr_model
        self.asr_model.eval()
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.all_logprobs = []
        self.chunk_len = chunk_len_in_secs
        self.buffer_len = buffer_len_in_secs
        assert(chunk_len_in_secs<=buffer_len_in_secs)
        
        feature_stride = asr_model._cfg.preprocessor['window_stride']
        self.model_stride_in_secs = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(self.chunk_len / self.model_stride_in_secs)
        self.blank_id = len(asr_model.decoder.vocabulary)
        self.plot=False

        self.stride = stride
        
    @torch.no_grad()    
    def transcribe_buffers(self, buffer_start, buffers, merge=True, plot=False):
        self.plot = plot
        self.buffers = buffers
        self.buffer_start = buffer_start
        self.data_layer.set_signal(buffers[:])
        self._get_batch_preds()      
        return self.decode_final(merge)
    
    def _get_batch_preds(self):
        device = self.asr_model.device
        for batch in iter(self.data_loader):

            audio_signal, audio_signal_len = batch

            audio_signal, audio_signal_len = audio_signal.to(device), audio_signal_len.to(device)
            probs, encoded_len, predictions = self.asr_model(input_signal=audio_signal, input_signal_length=audio_signal_len)
            # log_probs, encoded_len, predictions = self.asr_model(input_signal=audio_signal, input_signal_length=audio_signal_len, logprobs=True)
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
            self.all_logprobs.append(probs)
    
    def decode_final(self, merge=True, extra=0):
        self.unmerged = []
        self.toks_unmerged = []
        self.part_logprobs = []
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len - self.chunk_len) / 2) / self.model_stride_in_secs)

        decoded_frames = []
        all_toks = []
        for pred in self.all_preds:
            ids, toks = self._greedy_decoder(pred, self.asr_model.tokenizer)
            decoded_frames.append(ids)
            all_toks.append(toks)

        for idx, decoded in enumerate(decoded_frames):
            _stt, _end = len(decoded) - 1 - delay, len(decoded) - 1 - delay + self.n_tokens_per_chunk
            self.unmerged += decoded[len(decoded) - 1 - delay:len(decoded) - 1 - delay + self.n_tokens_per_chunk]
            self.part_logprobs.append(self.all_logprobs[idx][0, _stt:_end, :])
            # self.part_logprobs.append(self.all_logprobs[idx][0, :, :])
        self.unmerged_logprobs = torch.cat(self.part_logprobs, 0)
        if not merge:
            return self.unmerged
        return self._greedy_merge_with_ts(self.unmerged, self.buffer_start)
    
    def _greedy_decoder(self, preds, tokenizer):
        s = []
        ids = []
        for i in range(preds.shape[0]):
            if preds[i] == self.blank_id:
                s.append("_")
            else:
                pred = preds[i]
                s.append(tokenizer.ids_to_tokens([pred.item()])[0])
            ids.append(preds[i])
        return ids, s
    
    def _greedy_merge_with_ts(self, preds, buffer_start, ROUND=4):
        char_ts = [] 
        self.time_stride = self.stride*0.01
        decoded_prediction = []
        previous = self.blank_id
        unk = '⁇'
        for idx, p in enumerate(preds):
            ppp = p
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p.item())
                char_ts.append(round(buffer_start + idx*self.time_stride, ROUND))
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        hypothesis = hypothesis.replace(unk, '')
        word_ts = self.get_ts_from_decoded_prediction(decoded_prediction, hypothesis, char_ts)
        decoded_list = self.asr_model.tokenizer.ids_to_tokens(decoded_prediction)
        return hypothesis, word_ts
        
    def get_ts_from_decoded_prediction(self, decoded_prediction, hypothesis, char_ts):
        decoded_char_list = self.asr_model.tokenizer.ids_to_tokens(decoded_prediction)
        stt_idx, end_idx = 0, len(decoded_char_list)-1 
        space = '▁'
        word_ts = []
        word_open_flag = False
        for idx, ch in enumerate(decoded_char_list):
            if idx != end_idx and (space == ch and space in decoded_char_list[idx+1]):
                continue

            if (idx == stt_idx or space == decoded_char_list[idx-1] or (space in ch and len(ch) > 1)) and (ch != space):
                _stt = char_ts[idx]
                word_open_flag = True

            if word_open_flag and ch != space and (idx == end_idx or space in decoded_char_list[idx+1]):
                _end = round(char_ts[idx] + self.time_stride, 2)
                word_open_flag = False
                word_ts.append([_stt, _end])
        try:
            assert len(hypothesis.split()) == len(word_ts), "Hypothesis does not match word time stamp."
        except:
            ipdb.set_trace()
        return word_ts

    
    
def callback_sim(asr, uniq_key, buffer_counter, sdata, frame_count, time_info, status):
    start_time = time.time()
    asr.buffer_counter = buffer_counter
    sampled_seg_sig = sdata[asr.CHUNK_SIZE*(asr.buffer_counter):asr.CHUNK_SIZE*(asr.buffer_counter+1)]
    asr.uniq_id = uniq_key
    asr.signal = sdata
    words, timestamps, pred_diar_labels = asr.transcribe(sampled_seg_sig)
    if asr.buffer_start >= 0 and (pred_diar_labels != [] and pred_diar_labels != None):
        asr._update_word_and_word_ts(words, timestamps)
        string_out = asr._get_speaker_label_per_word(uniq_key, asr.word_seq, asr.word_ts_seq, pred_diar_labels)
        write_txt(f"{asr.diar._out_dir}/print_script.sh", string_out.strip())
        
    ETA = time.time()-start_time 
    if asr.diar.params['force_real_time']:
        assert ETA < asr.frame_len, "The process has failed to be run in real-time."
        time.sleep(1.0 - ETA*1.0)

class OnlineClusteringDiarizer(ClusteringDiarizer, ASR_DIAR_OFFLINE):
    def __init__(self, cfg: DictConfig, params: Dict):
        super().__init__(cfg)
        
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        
        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg
        self.params = params
        self._out_dir = self._cfg.diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        self._speaker_manifest_path = self._cfg.diarizer.speaker_embeddings.oracle_vad_manifest
        self.AUDIO_RTTM_MAP = None
        # self.paths2audio_files = self._cfg.diarizer.paths2audio_files
        
        self.paths2session_audio_files = []
        self.all_hypothesis = []
        self.all_reference = []
        self.out_rttm_dir = None

        self.embed_seg_len = self._cfg.diarizer.speaker_embeddings.window_length_in_sec
        self.embed_seg_hop = self._cfg.diarizer.speaker_embeddings.shift_length_in_sec
        self.max_num_speakers = 8
        self._current_buffer_segment_count = 64
        self._history_buffer_segment_count = 64
        self.MINIMUM_CLUS_BUFFER_SIZE = 10
        self.MINIMUM_HIST_BUFFER_SIZE = 32
        self._minimum_segments_per_buffer = int(self._history_buffer_segment_count/self.max_num_speakers)
        self.segment_abs_time_range_list = []
        self.segment_raw_audio_list = []
        self.Y_fullhist = []
        self.use_online_mat_reduction = True
        self.history_embedding_buffer_emb = np.array([])
        self.history_embedding_buffer_label = np.array([])
        self.history_buffer_seg_start = None
        self.history_buffer_seg_end = None
        self.old_history_buffer_seg_end = None
        self.last_emb_in_length = -float('inf')
        self.frame_index = None
        self.index_dict = {'max_embed_count': 0}
        self.cumulative_speaker_count = {}
        self.embedding_count_history = []
        self.p_value_hist = []

        self.online_diar_buffer_segment_quantity = params['online_history_buffer_segment_quantity']
        self.online_history_buffer_segment_quantity = params['online_diar_buffer_segment_quantity']
        self.enhanced_count_thres = params['enhanced_count_thres']
        self.max_num_speaker = params['max_num_speaker']
        self.oracle_num_speakers = None

        self.diar_eval_count = 0
        self.DER_csv_list = []
        self.der_dict = {}
        self.der_stat_dict = {"avg_DER":0, "avg_CER":0, "max_DER":0, "max_CER":0, "cum_DER":0, "cum_CER":0}
        self.color_palette = {'speaker_0': '\033[1;32m',
                              'speaker_1': '\033[1;34m',
                              'speaker_2': '\033[1;30m',
                              'speaker_3': '\033[1;31m',
                              'speaker_4': '\033[1;35m',
                              'speaker_5': '\033[1;36m',
                              'speaker_6': '\033[1;37m',
                              'speaker_7': '\033[1;30m',
                              'speaker_8': '\033[1;33m',
                              'speaker_9': '\033[0;34m',
                              'white': '\033[0;37m'}
    
    @property 
    def online_diar_buffer_segment_quantity(self, value):
        return self._current_buffer_segment_count

    @online_diar_buffer_segment_quantity.setter
    def online_diar_buffer_segment_quantity(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert value >= self.MINIMUM_CLUS_BUFFER_SIZE, f"Online diarization clustering buffer should be bigger than {self.MINIMUM_CLUS_BUFFER_SIZE}"
        self._current_buffer_segment_count = value # How many segments we want to use as clustering buffer
    
    @property 
    def online_history_buffer_segment_quantity(self, value):
        return self._current_buffer_segment_count

    @online_history_buffer_segment_quantity.setter
    def online_history_buffer_segment_quantity(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert value >= self.MINIMUM_HIST_BUFFER_SIZE, f"Online diarization history buffer should be bigger than {self.MINIMUM_HIST_BUFFER_SIZE}"
        self._history_buffer_segment_count = value # How many segments we want to use as history buffer

    def prepare_diarization(self, paths2audio_files: List[str] = None, batch_size: int = 1):
        """
        """
        # if paths2audio_files:
        # else:
        # ipdb.set_trace()
        if self._cfg.diarizer.paths2audio_files not in [None, 'None']:
            # self.paths2audio_files = paths2audio_files
        # else:
            # if self._cfg.diarizer.paths2audio_files is None:
                # raise ValueError("Pass path2audio files either through config or to diarize method")
            # else:
            self.paths2audio_files = self._cfg.diarizer.paths2audio_files

            if type(self.paths2audio_files) is str and os.path.isfile(self.paths2audio_files):
                paths2audio_files = []
                with open(self.paths2audio_files, 'r') as path2file:
                    for audiofile in path2file.readlines():
                        audiofile = audiofile.strip()
                        paths2audio_files.append(audiofile)

            elif type(self.paths2audio_files) in [list, ListConfig]:
                paths2audio_files = list(self.paths2audio_files)
            else:
                raise ValueError("paths2audio_files must be of type list or path to file containing audio files")

            self.paths2session_audio_files= paths2audio_files

        if self._cfg.diarizer.path2groundtruth_rttm_files not in [None, 'None']:
            self.AUDIO_RTTM_MAP = audio_rttm_map(paths2audio_files, self._cfg.diarizer.path2groundtruth_rttm_files)
        else:
            self.AUDIO_RTTM_MAP = {}

        # self._extract_embeddings(self._speaker_manifest_path)
        self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(self.out_rttm_dir, exist_ok=True)
    
    def getMergeQuantity(self, new_emb_n, before_cluster_labels):
        """
        Determine which embeddings we need to reduce or merge in history buffer.
        We want to merge or remove the embedding in the bigger cluster first.
        At the same time, we keep the minimum number of embedding per cluster
        with the variable named self._minimum_segments_per_buffer.
        The while loop creates a numpy array emb_n_per_cluster.
        that tells us how many embeddings we should remove/merge per cluster.

        Args:
            new_emb_n: (int)
                the quantity of the newly obtained embedding from the stream.

            before_cluster_labels: (np.array)
                the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
        """
        targeted_total_n = new_emb_n
        count_dict = Counter(before_cluster_labels)
        spk_freq_count = np.bincount(before_cluster_labels)
        class_vol = copy.deepcopy(spk_freq_count)
        emb_n_per_cluster = np.zeros_like(class_vol).astype(int)
        arg_max_spk_freq = np.argsort(spk_freq_count)[::-1]
        count = 0
        while np.sum(emb_n_per_cluster) < new_emb_n:
            recurr_idx = np.mod(count, len(count_dict))
            curr_idx = arg_max_spk_freq[recurr_idx]
            margin = (spk_freq_count[curr_idx] - emb_n_per_cluster[curr_idx]) - self._minimum_segments_per_buffer
            if margin > 0:
                target_number = min(margin, new_emb_n)
                emb_n_per_cluster[curr_idx] += target_number
                new_emb_n -= target_number
            count += 1
        assert sum(emb_n_per_cluster) == targeted_total_n, "emb_n_per_cluster does not match with targeted number new_emb_n."
        return emb_n_per_cluster

    def reduce_emb(self, cmat, tick2d, emb_ndx, cluster_labels, method='avg'):
        LI, RI = tick2d[0, :], tick2d[1, :]
        LI_argdx = tick2d[0].argsort()

        if method == 'drop':
            cmat_sym = cmat + cmat.T
            clus_score = np.vstack((np.sum(cmat_sym[LI], axis=1), np.sum(cmat_sym[RI], axis=1)))
            selected_dx = np.argmax(clus_score, axis=0)
            emb_idx = np.choose(selected_dx, tick2d)
            result_emb = emb_ndx[emb_idx, :]
        elif method == 'avg':
            LI, RI = LI[LI_argdx], RI[LI_argdx]
            result_emb = 0.5*(emb_ndx[LI, :] + emb_ndx[RI, :])
        else:
            raise ValueError(f'Method {method} does not exist. Abort.')
        merged_cluster_labels = cluster_labels[np.array(list(set(LI)))]
        bypass_ndx = np.array(list(set(range(emb_ndx.shape[0])) - set(list(LI)+list(RI)) ) )
        if len(bypass_ndx) > 0:
            result_emb = np.vstack((emb_ndx[bypass_ndx], result_emb))  
            merged_cluster_labels = np.hstack((cluster_labels[bypass_ndx], merged_cluster_labels))
        return result_emb, LI, merged_cluster_labels
    

    def reduceEmbedding(self, emb_in, mat):
        history_n, current_n = self._history_buffer_segment_count, self._current_buffer_segment_count
        add_new_emb_to_history = True

        # print("[Streaming diarization with history buffer]: emb_in.shape:", emb_in.shape)
        if len(self.history_embedding_buffer_emb) > 0:
            if emb_in.shape[0] <= self.index_dict['max_embed_count']:
                # If the number of embeddings is decreased compared to the last trial,
                # then skip embedding merging.
                add_new_emb_to_history = False
                hist_curr_boundary = self.history_buffer_seg_end
            else:
                # Since there are new embeddings, we push the same amount (new_emb_n) 
                # of old embeddings to the history buffer.
                # We should also update self.history_buffer_seg_end which is a pointer.
                hist_curr_boundary = emb_in.shape[0] - self._current_buffer_segment_count
                _stt = self.history_buffer_seg_end # The old history-current boundary
                _end = hist_curr_boundary # The new history-current boundary
                new_emb_n = _end - _stt
                assert new_emb_n > 0, "new_emb_n cannot be 0 or a negative number."
                update_to_history_emb = emb_in[_stt:_end]
                update_to_history_label = self.Y_fullhist[_stt:_end]
                emb = np.vstack((self.history_embedding_buffer_emb, update_to_history_emb))
                before_cluster_labels = np.hstack((self.history_embedding_buffer_label, update_to_history_label))
                self.history_buffer_seg_end = hist_curr_boundary
        else:
            # This else statement is for the very first diarization loop.
            # This is the very first reduction frame.
            hist_curr_boundary = emb_in.shape[0] - self._current_buffer_segment_count
            new_emb_n = mat.shape[0] - (self._current_buffer_segment_count + self._history_buffer_segment_count)
            emb = emb_in[:hist_curr_boundary]
            before_cluster_labels = self.Y_fullhist[:hist_curr_boundary]
            self.history_buffer_seg_end = hist_curr_boundary
       
        # Update the history/current_buffer boundary cursor
        total_emb, total_cluster_labels = [], []
        
        if add_new_emb_to_history:
            class_target_vol = self.getMergeQuantity(new_emb_n, before_cluster_labels)
            
            # Merge the segments in the history buffer
            for spk_idx, N in enumerate(list(class_target_vol)):
                ndx = np.where(before_cluster_labels == spk_idx)[0]
                if N <= 0:
                    result_emb = emb[ndx]
                    merged_cluster_labels = before_cluster_labels[ndx]
                else:
                    cmat = np.tril(mat[:,ndx][ndx,:])
                    tick2d = self.getIndecesForEmbeddingReduction(cmat, ndx, N)
                    spk_cluster_labels, emb_ndx = before_cluster_labels[ndx], emb[ndx]
                    result_emb, tick_sum, merged_cluster_labels = self.reduce_emb(cmat, tick2d, emb_ndx, spk_cluster_labels, method='avg')
                    assert (ndx.shape[0] - N) == result_emb.shape[0], ipdb.set_trace()
                total_emb.append(result_emb)
                total_cluster_labels.append(merged_cluster_labels)
        
            self.history_embedding_buffer_emb = np.vstack(total_emb)
            self.history_embedding_buffer_label = np.hstack(total_cluster_labels)
            assert self.history_embedding_buffer_emb.shape[0] == history_n, ipdb.set_trace()
        else:
            total_emb.append(self.history_embedding_buffer_emb)
            total_cluster_labels.append(self.history_embedding_buffer_label)

        # Add the current buffer
        total_emb.append(emb_in[hist_curr_boundary:])
        total_cluster_labels.append(self.Y_fullhist[hist_curr_boundary:])

        history_and_current_emb = np.vstack(total_emb)
        history_and_current_labels = np.hstack(total_cluster_labels)
        assert history_and_current_emb.shape[0] <= (history_n + current_n), ipdb.set_trace()
        
        self.last_emb_in_length = emb_in.shape[0]
        return history_and_current_emb, history_and_current_labels, current_n, add_new_emb_to_history
    
    def getIndecesForEmbeddingReduction(self, cmat, ndx, N):
        """
        Get indeces of the embeddings we want to merge or drop.

        Args:
            cmat: (np.array)
            ndx: (np.array)
            N: (int)

        Output:
            tick2d: (numpy.array)
        """
        comb_limit = int(ndx.shape[0]/2)
        assert N <= comb_limit, f" N is {N}: {N} is bigger than comb_limit -{comb_limit}"
        idx2d = np.unravel_index(np.argsort(cmat, axis=None)[::-1], cmat.shape)
        num_of_lower_half = int((cmat.shape[0]**2 - cmat.shape[0])/2)
        idx2d = (idx2d[0][:num_of_lower_half], idx2d[1][:num_of_lower_half])
        cdx, left_set, right_set, total_set = 0, [], [], []
        while len(left_set) <  N and len(right_set) < N:
            Ldx, Rdx = idx2d[0][cdx], idx2d[1][cdx] 
            if (not Ldx in total_set) and (not Rdx in total_set):
                left_set.append(Ldx)
                right_set.append(Rdx)
                total_set = left_set + right_set
            cdx += 1
        tick2d = np.array([left_set, right_set])
        return tick2d
    
    @timeit
    def getReducedMat(self, mat, emb):
        margin_seg_n = mat.shape[0] - (self._current_buffer_segment_count + self._history_buffer_segment_count)
        if margin_seg_n > 0:
            mat = 0.5*(mat + mat.T)
            np.fill_diagonal(mat, 0)
            merged_emb, cluster_labels, current_n, add_new = self.reduceEmbedding(emb, mat)
        else:
            merged_emb = emb
            current_n = self._current_buffer_segment_count
            cluster_labels, add_new = None, True
        isOnline = (len(self.history_embedding_buffer_emb) != 0)
        return merged_emb, cluster_labels, add_new, isOnline
    
    def online_eval_diarization(self, pred_labels, rttm_file, ROUND=2):
        pred_diar_labels, ref_labels_list = [], []
        all_hypotheses, all_references = [], []

        if os.path.exists(rttm_file):
            ref_labels_total = rttm_to_labels(rttm_file)
            ref_labels = get_partial_ref_labels(pred_labels, ref_labels_total)
            reference = labels_to_pyannote_object(ref_labels)
            all_references.append(reference)
        else:
            raise ValueError("No reference RTTM file provided.")

        pred_diar_labels.append(pred_labels)

        self.der_stat_dict['ref_n_spk'] = self.get_num_of_spk_from_labels(ref_labels)
        self.der_stat_dict['est_n_spk'] = self.get_num_of_spk_from_labels(pred_labels)
        hypothesis = labels_to_pyannote_object(pred_labels)
        if ref_labels == [] and pred_labels != []:
            logging.info(
                "Streaming Diar [{}][frame-  {}th  ]:".format(
                    self.uniq_key, self.frame_index
                )
            )
            DER, CER, FA, MISS = 100.0, 0.0, 100.0, 0.0
            der_dict, der_stat_dict = self.get_stat_DER(DER, CER, FA, MISS)
            return der_dict, der_stat_dict
        else:
            all_hypotheses.append(hypothesis)
            try:
                DER, CER, FA, MISS, _= get_DER(all_references, all_hypotheses)
            except:
                DER, CER, FA, MISS = 100.0, 0.0, 100.0, 0.0
            logging.info(
                "Streaming Diar [{}][frame-    {}th    ]: DER:{:.4f} MISS:{:.4f} FA:{:.4f}, CER:{:.4f}".format(
                    self.uniq_key, self.frame_index, DER, MISS, FA, CER
                )
            )

            der_dict, der_stat_dict = self.get_stat_DER(DER, CER, FA, MISS)
            return der_dict, der_stat_dict
    
    def get_stat_DER(self, DER, CER, FA, MISS, ROUND=2):
        der_dict = {"DER": round(100*DER, ROUND), 
                     "CER": round(100*CER, ROUND), 
                     "FA":  round(100*FA, ROUND), 
                     "MISS": round(100*MISS, ROUND)}
        self.diar_eval_count += 1
        self.der_stat_dict['cum_DER'] += DER
        self.der_stat_dict['cum_CER'] += CER
        self.der_stat_dict['avg_DER'] = round(100*self.der_stat_dict['cum_DER']/self.diar_eval_count, ROUND)
        self.der_stat_dict['avg_CER'] = round(100*self.der_stat_dict['cum_CER']/self.diar_eval_count, ROUND)
        self.der_stat_dict['max_DER'] = round(max(der_dict['DER'], self.der_stat_dict['max_DER']), ROUND)
        self.der_stat_dict['max_CER'] = round(max(der_dict['CER'], self.der_stat_dict['max_CER']), ROUND)
        return der_dict, self.der_stat_dict
    
    def print_time_colored(self, string_out, speaker, start_point, end_point, params):
        if params['color']:
            color = self.color_palette[speaker]
        else:
            color = ''

        datetime_offset = 16 * 3600
        if float(start_point) > 3600:
            time_str = "%H:%M:%S.%f"
        else:
            time_str = "%M:%S.%f"
        start_point_str = datetime.fromtimestamp(float(start_point) - datetime_offset).strftime(time_str)[:-4]
        end_point_str = datetime.fromtimestamp(float(end_point) - datetime_offset).strftime(time_str)[:-4]
        if params['print_time']:
            strd = "\n{}[{} - {}] {}: ".format(color, start_point_str, end_point_str, speaker)
        else:
            strd = "\n{}[{}]: ".format(color, speaker)
        
        if params['print_transcript']:
            print(strd, end=" ")
        return string_out + strd
    
    @staticmethod
    def print_word_colored(string_out, word, params, space=" "):
        word = word.strip()
        if params['print_transcript']:
            print(word, end=" ")
        return string_out + word + " "
    

    def OnlineCOSclustering(self, key, emb, oracle_num_speakers=None, max_num_speaker=8, enhanced_count_thres=80, min_samples_for_NMESC=6, fixed_thres=None, cuda=False):
        """
        Online clustering method for speaker diarization based on cosine similarity.

        Parameters:
            key: (str)
                A unique ID for each speaker

            emb: (numpy array)
                Speaker embedding extracted from an embedding extractor

            oracle_num_speaker: (int or None)
                Oracle number of speakers if known else None

            max_num_speaker: (int)
                Maximum number of clusters to consider for each session

            min_samples: (int)
                Minimum number of samples required for NME clustering, this avoids
                zero p_neighbour_lists. Default of 6 is selected since (1/rp_threshold) >= 4
                when max_rp_threshold = 0.25. Thus, NME analysis is skipped for matrices
                smaller than (min_samples)x(min_samples).
        Returns:
            Y: (List[int])
                Speaker label for each segment.
        """
        mat = getCosAffinityMatrix(emb)
        org_mat = copy.deepcopy(mat)
        emb, reduced_labels, add_new, isOnline = self.getReducedMat(mat, emb)
        
        self.index_dict[self.frame_index] = (org_mat.shape[0], self.history_buffer_seg_end)
        self.index_dict['max_embed_count'] = max(org_mat.shape[0], self.index_dict['max_embed_count'])

        if emb.shape[0] == 1:
            return np.array([0])
        elif emb.shape[0] <= max(enhanced_count_thres, min_samples_for_NMESC) and oracle_num_speakers is None:
            est_num_of_spk_enhanced = getEnhancedSpeakerCount(key, emb, cuda, random_test_count=5, anchor_spk_n=3, anchor_sample_n=10, sigma=100)
        else:
            est_num_of_spk_enhanced = None

        if oracle_num_speakers:
            max_num_speaker = oracle_num_speakers

        mat = getCosAffinityMatrix(emb)
        nmesc = NMESC(
            mat,
            max_num_speaker=max_num_speaker,
            max_rp_threshold=0.25,
            sparse_search=True,
            sparse_search_volume=10,
            fixed_thres=None,
            NME_mat_size=300,
            cuda=cuda,
        )

        if emb.shape[0] > min_samples_for_NMESC:
            est_num_of_spk, p_hat_value = self.estNumOfSpeakers(nmesc, isOnline)
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            est_num_of_spk, g_p = nmesc.getEigRatio(int(mat.shape[0]/2))
            affinity_mat = mat
        
        if oracle_num_speakers:
            est_num_of_spk = oracle_num_speakers
        elif est_num_of_spk_enhanced:
            est_num_of_spk = est_num_of_spk_enhanced

        spectral_model = _SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda)
        Y = spectral_model.predict(affinity_mat)
        Y_out = self.matchLabels(org_mat, Y, isOnline, add_new)
        return Y_out
    
    def estNumOfSpeakers(self, nmesc, isOnline):
        """
        To save the running time, the p-value is only estimated in the beginning of the session.
        After switching to online mode, the system uses the most common estimated p-value.
        Args:
            nmesc: (NMESC)
                nmesc instance.
            isOnline: (bool)
                Indicates whether the system is running on online mode or not.

        Returns:
            est_num_of_spk: (int)
                The estimated number of speakers.
            p_hat_value: (int)
                The estimated p-value from NMESC method.
        """
        if isOnline:
            p_hat_value =  max(self.p_value_hist, key = self.p_value_hist.count)
            est_num_of_spk, g_p = nmesc.getEigRatio(p_hat_value)
        else:
            est_num_of_spk, p_hat_value, best_g_p_value = nmesc.NMEanalysis()
            self.p_value_hist.append(p_hat_value)
        return est_num_of_spk, p_hat_value

    
    def matchLabels(self, org_mat, Y, isOnline, add_new):
        if isOnline:
            # Online clustering mode with history buffer
            update_point = self._history_buffer_segment_count
            Y_matched = self.matchNewOldclusterLabels(self.Y_fullhist[self.history_buffer_seg_end:], Y, with_history=True)
            if add_new:
                assert Y_matched[update_point:].shape[0] == self._current_buffer_segment_count, "Update point sync is not correct."
                Y_out = np.hstack((self.Y_fullhist[:self.history_buffer_seg_end], Y_matched[update_point:]))
                self.Y_fullhist = Y_out
            else:
                # Do not update cumulative labels since there are no new segments.
                Y_out = self.Y_fullhist[:org_mat.shape[0]]
            assert len(Y_out) == org_mat.shape[0], ipdb.set_trace()
        else:
            # Regular offline clustering
            if len(self.Y_fullhist) == 0:
                Y_out = Y
            else:
                Y_out = self.matchNewOldclusterLabels(self.Y_fullhist, Y, with_history=False)
            self.Y_fullhist = Y_out
        return Y_out

    @timeit
    def matchNewOldclusterLabels(self, cum_labels, Y, with_history=True):
        """
        Run Hungarian algorithm (linear sum assignment) to find the best permuation mapping between
        the cumulated labels in history and the new clustering output labels.

        Args:
            cum_labels (np.array):
                Cumulated diarization labels. This will be concatenated with history embedding speaker label
                then compared with the predicted label Y.

            Y (np.array):
                Contains predicted labels for reduced history embeddings concatenated with the predicted label.
                Permutation is not matched yet.

        Returns:
            mapping_array[Y] (np.array):
                An output numpy array where the input Y is mapped with mapping_array.

        """
        spk_count = max(len(set(cum_labels)), len(set(Y)))
        P_raw = np.hstack((self.history_embedding_buffer_label, cum_labels)).astype(int)
        Q_raw = Y.astype(int)
        U_set = set(P_raw) | set(Q_raw)
        min_len = min(P_raw.shape[0], Q_raw.shape[0])
        P, Q = P_raw[:min_len], Q_raw[:min_len]
        PuQ = (set(P) | set(Q))
        PiQ = (set(P) & set(Q))
        PmQ, QmP =  set(P) - set(Q),  set(Q) - set(P)
        
        if len(PiQ) == 0:
            # In this case, the label is totally flipped (0<->1)
            # without any commom labels.
            # This should be differentiated from the second case.
            pass
        elif with_history and (len(PmQ) > 0 or len(QmP) > 0):
            # Keep only the common speaker labels.
            # This is mainly for the newly added speakers
            # from the labels in Y.
            keyQ = ~np.zeros_like(Q).astype(bool)
            keyP = ~np.zeros_like(P).astype(bool)
            for spk in list(QmP):
                keyQ[Q == spk] = False
            for spk in list(PmQ):
                keyP[P == spk] = False
            common_key = keyP*keyQ
            if all(~common_key) != True:
                P, Q = P[common_key], Q[common_key]
            elif all(~common_key) == True:
                P, Q = P, Q

        if len(U_set) == 1:
            # When two speaker vectors are exactly the same: No need to encode.
            col_ind = np.array([0, 0])
            mapping_array = col_ind
            return mapping_array[Y]
        else:
            # Use one-hot encodding to find the best match.
            enc = OneHotEncoder(handle_unknown='ignore') 
            all_spks_labels = [[x] for x in range(len(U_set))]
            # all_spks_labels = [[x] for x in range(len(PuQ))]
            enc.fit(all_spks_labels)
            enc_P = enc.transform(P.reshape(-1, 1)).toarray()
            enc_Q = enc.transform(Q.reshape(-1, 1)).toarray()
            stacked = np.hstack((enc_P, enc_Q))
            cost = -1*linear_kernel(stacked.T)[spk_count:, :spk_count]
            row_ind, col_ind = linear_sum_assignment(cost)

            # If number of are speakers in each vector is not the same
            mapping_array = np.arange(len(U_set)).astype(int)
            for x in range(col_ind.shape[0]):
                if x in (set(PmQ) | set(QmP)):
                    mapping_array[x] = x
                else:
                    mapping_array[x] = col_ind[x]
            return mapping_array[Y]

# simple data layer to pass audio signal
class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

class ASR_DIAR_ONLINE:
    def __init__(self, 
                 diar, 
                 params, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.params = params
        self.sr = self.params['sample_rate']
        self.frame_len = float(self.params['frame_len'])
        self.frame_overlap = float(self.params['frame_overlap'])
        self.n_frame_len = int(self.frame_len * self.sr)
        self.n_frame_overlap = int(self.frame_overlap * self.sr)
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.CHUNK_SIZE = int(self.frame_len*self.sr)
        self._load_ASR_model(params)
        if self.params['use_lm_for_asr']:
            self.lm_model, self.ctc_decode = self._load_LM_model()

        # For diarization
        self.diar = diar
        self.n_embed_seg_len = int(self.sr * self.diar.embed_seg_len)
        self.n_embed_seg_hop = int(self.sr * self.diar.embed_seg_hop)
        
        self.embs_array = None
        self.frame_index = 0
        self.Y_fullhist = []
        
        # minimun width to consider non-speech activity 
        self.nonspeech_threshold = params['speech_detection_threshold']
        self.overlap_frames_count = int(self.n_frame_overlap/self.sr)
        self.segment_raw_audio_list = []
        self.segment_abs_time_range_list = []
        self.cumulative_speech_labels = []

        self.buffer_start = None
        self.frame_start = 0
        self.rttm_file_path = []
        self.word_seq = []
        self.word_ts_seq = []
        self.merged_cluster_labels = []
        self.offline_logits = None
        self.debug_mode = False
        self.online_diar_label_update_sec = 30
        self.streaming_buffer_list = []
        self.reset()
    
    def _load_ASR_model(self, params):
        self.device = f'cuda:{self.params["device"]}' if torch.cuda.is_available() else 'cpu'
        if 'citrinet' in  params['ASR_model_name']:
            self.asr_stride = 8
            self.asr_delay_sec = 0.12
            self.new_asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(params['ASR_model_name'], map_location=self.device)
        elif 'conformer' in params['ASR_model_name']:
            self.asr_stride = 4
            self.asr_delay_sec = 0.06
            self.new_asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(params['ASR_model_name'], map_location=self.device)
        else:
            raise ValueError(f"{params['ASR_model_name']} is not compatible with the streaming launcher.")
        self.new_asr_model = self.new_asr_model.to(self.device)
        self.time_stride = 0.01 * self.asr_stride
        # self.params['offset'] = -1 * self.asr_delay_sec
        self.params['offset'] = 0
        self.params['time_stride'] = self.asr_stride
        self.buffer_list = []

    def _load_LM_model(self):
        vocab_org = self.new_asr_model.decoder.vocabulary
        TOKEN_OFFSET = 100
        encoding_level = kenlm_utils.SUPPORTED_MODELS.get(type(self.new_asr_model).__name__, None)
        # lm_path="/home/taejinp/Downloads/LM/4gram_big.arpa"
        lm_path="/home/taejinp/Downloads/LM/generic_en_lang_model_medium-r20190501.arpa"
        # lm_path="/home/taejinp/Downloads/LM/generic_en_lang_model_large-r20190501.arpa"
        # lm_path=None
        if encoding_level == "subword":
            vocab = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab_org))]
        
        beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
            vocab=vocab,
            beam_width=2,
            alpha=1.0,
            beta=0.0,
            cutoff_prob=1.0,
            cutoff_top_n=200,
            num_cpus=1,
            lm_path=lm_path,
            input_tensor=False,
        )
        # num_cpus=max(os.cpu_count(), 1),
        return beam_search_lm, ctc_decode

    def _convert_to_torch_var(self, audio_signal):
        audio_signal = torch.stack(audio_signal).float().to(self.new_asr_model.device)
        audio_signal_lens = torch.from_numpy(np.array([self.n_embed_seg_len for k in range(audio_signal.shape[0])])).to(self.new_asr_model.device)
        return audio_signal, audio_signal_lens

    def _process_cluster_labels(self, segment_ranges, cluster_labels):
        assert len(cluster_labels) == len(segment_ranges)
        lines = []
        for idx, label in enumerate(cluster_labels):
            tag = 'speaker_' + str(label)
            lines.append(f"{segment_ranges[idx][0]} {segment_ranges[idx][1]} {tag}")
        cont_lines = get_contiguous_stamps(lines)
        string_labels = merge_stamps(cont_lines)
        return string_labels

    def _update_word_and_word_ts(self, words, word_timetamps):
        if word_timetamps != []:
            # Remove
            if self.word_ts_seq != []:

                bcursor = len(self.word_ts_seq)-1
                while bcursor:
                    if self.frame_start > self.word_ts_seq[bcursor][0]:
                        break 
                    bcursor -= 1
                del self.word_seq[bcursor:]
                del self.word_ts_seq[bcursor:]
            

            cursor = len(word_timetamps)-1
            while cursor:
                if self.frame_start > word_timetamps[cursor][0]:
                    break 
                cursor -= 1
            
            # Add
            if self.word_ts_seq == []:
                self.word_seq.extend(words)
                self.word_ts_seq.extend(word_timetamps)
            else:
                self.word_seq.extend(words[cursor:])
                self.word_ts_seq.extend(word_timetamps[cursor:])

    def _get_word_ts(self, text, timestamps, end_stamp):
        if text.strip() == '':
            _trans_words, word_timetamps, _spaces = [], [], []
        elif len(text.split()) == 1:
            _trans_words = [text]
            word_timetamps = [[timestamps[0], end_stamp]]
            _spaces = []
        else:
            trans, timestamps = self.diar.clean_trans_and_TS(text, timestamps)
            _spaces, _trans_words = self.diar._get_spaces(trans, timestamps)
            word_timetamps_middle = [[_spaces[k][1], _spaces[k + 1][0]] for k in range(len(_spaces) - 1)]
            word_timetamps = [[timestamps[0], _spaces[0][0]]] + word_timetamps_middle + [[_spaces[-1][1], end_stamp]]
        
        assert len(_trans_words) == len(word_timetamps)
        self.word_seq.extend(_trans_words)
        self.word_ts_seq.extend(word_timetamps)
  
    @torch.no_grad()
    def _run_embedding_extractor(self, audio_signal):
        self.diar._speaker_model.to(self.device)
        self.diar._speaker_model.eval()
        torch_audio_signal, torch_audio_signal_lens = self._convert_to_torch_var(audio_signal)
        _, torch_embs = self.diar._speaker_model.forward(input_signal=torch_audio_signal, 
                                                         input_signal_length=torch_audio_signal_lens)
        return torch_embs

    def _get_speaker_embeddings(self, embs_array, audio_signal, segment_ranges, online_extraction=True):
        torch.manual_seed(0)
        if online_extraction:
            hop = self.diar._cfg.diarizer.speaker_embeddings.shift_length_in_sec
            if embs_array is None:
                target_segment_count = len(segment_ranges)
                stt, end = 0, len(segment_ranges)
            else:
                target_segment_count = int(min(np.ceil((2*self.frame_overlap + self.frame_len)/hop), len(segment_ranges)))
                stt, end = len(segment_ranges)-target_segment_count, len(segment_ranges)
            torch_embs = self._run_embedding_extractor(audio_signal[stt:end])
            if embs_array is None:
                embs_array = torch_embs.cpu().numpy()
            else:
                embs_array = np.vstack((embs_array[:stt,:], torch_embs.cpu().numpy()))
            assert len(segment_ranges) == embs_array.shape[0], "Segment ranges and embs_array shapes do not match."
            
        else:
            torch_embs = self._run_embedding_extractor(audio_signal)
            embs_array = torch_embs.cpu().numpy()
        return embs_array

    @timeit
    def _online_diarization(self, audio_signal, segment_ranges):
        self.embs_array = self._get_speaker_embeddings(self.embs_array, audio_signal, segment_ranges)

        if self.debug_mode:
            _diarization_function = COSclustering
        else:
            # _diarization_function = COSclustering
            _diarization_function = self.diar.OnlineCOSclustering

        cluster_labels = _diarization_function(
            None, 
            self.embs_array, 
            oracle_num_speakers=self.diar.oracle_num_speakers,
            enhanced_count_thres=self.diar.enhanced_count_thres, 
            max_num_speaker=self.diar.max_num_speaker, 
            cuda=True,
        )
        # print("Est num of speakers: ", len(set(cluster_labels)))
        assert len(cluster_labels) == self.embs_array.shape[0]

        string_labels = self._process_cluster_labels(segment_ranges, cluster_labels)
        return string_labels

    @staticmethod 
    def get_mapped_speaker(speaker_mapping, speaker):
        if speaker in speaker_mapping:
            new_speaker = speaker_mapping[speaker]
        else:
            new_speaker = speaker
        return new_speaker

    def _get_ASR_based_VAD_timestamps(self, logits, use_offset_time=True):
        blanks = self._get_silence_timestamps(logits, symbol_idx = 28, state_symbol='blank')
        non_speech = list(filter(lambda x:x[1] - x[0] > self.nonspeech_threshold, blanks))
        if use_offset_time:
            offset_sec = int(self.frame_index - 2*self.overlap_frames_count)
        else:
            offset_sec = 0
        speech_labels = self._get_speech_labels(logits, non_speech, offset_sec)
        return speech_labels

    def _get_silence_timestamps(self, probs, symbol_idx, state_symbol):
        spaces = []
        idx_state = 0
        state = ''
        
        if np.argmax(probs[0]) == symbol_idx:
            state = state_symbol

        for idx in range(1, probs.shape[0]):
            current_char_idx = np.argmax(probs[idx])
            if state == state_symbol and current_char_idx != 0 and current_char_idx != symbol_idx:
                spaces.append([idx_state, idx-1])
                state = ''
            if state == '':
                if current_char_idx == symbol_idx:
                    state = state_symbol
                    idx_state = idx

        if state == state_symbol:
            spaces.append([idx_state, len(probs)-1])
       
        return spaces
   
    def _get_speech_labels(self, probs, non_speech, offset_sec, ROUND=2):
        frame_offset =  float((offset_sec + self.params['offset'])/self.time_stride)
        speech_labels = []
        
        if non_speech == []: 
            start = (0 + frame_offset)*self.time_stride
            end = (len(probs) -1 + frame_offset)*self.time_stride
            start, end = round(start, ROUND), round(end, ROUND)
            if start != end:
                speech_labels.append([start, end])

        else:
            start = frame_offset * self.time_stride
            first_end = (non_speech[0][0]+frame_offset)*self.time_stride
            start, first_end = round(start, ROUND), round(first_end, ROUND)
            if start != first_end:
                speech_labels.append([start, first_end])

            if len(non_speech) > 1:
                for idx in range(len(non_speech)-1):
                    start = (non_speech[idx][1] + frame_offset)*self.time_stride
                    end = (non_speech[idx+1][0] + frame_offset)*self.time_stride
                    start, end = round(start, ROUND), round(end, ROUND)
                    if start != end:
                        speech_labels.append([start, end])
            
            last_start = (non_speech[-1][1] + frame_offset)*self.time_stride
            last_end = (len(probs) -1 + frame_offset)*self.time_stride

            last_start, last_end = round(last_start, ROUND), round(last_end, ROUND)
            if last_start != last_end:
                speech_labels.append([last_start, last_end])

        return speech_labels
    
    def _get_speaker_label_per_word(self, uniq_id, words, word_ts_list, pred_diar_labels):
        params = self.diar.params
        start_point, end_point, speaker = pred_diar_labels[0].split()
        word_pos, idx = 0, 0
        string_out = ''
        string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
        for j, word_ts_stt_end in enumerate(word_ts_list):
            word_pos = np.mean(word_ts_stt_end)
            if word_pos < float(end_point):
                string_out = self.diar.print_word(string_out, words[j], params)
            else:
                idx += 1
                idx = min(idx, len(pred_diar_labels)-1)
                start_point, end_point, speaker = pred_diar_labels[idx].split()
                string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
                string_out = self.diar.print_word(string_out, words[j], params)
            stt_sec, end_sec = self.get_timestamp_in_sec(word_ts_stt_end, params)
        
        if self.rttm_file_path:
            string_out = self._print_DER_info(uniq_id, string_out, pred_diar_labels, params)
        else:
            logging.info(
                "Streaming Diar [{}][frame-  {}th  ]:".format(
                    self.diar.uniq_key, self.frame_index
                )
            )

        return string_out 
    
    @staticmethod
    def get_timestamp_in_sec(word_ts_stt_end, params):
        stt = round(params['offset'] + word_ts_stt_end[0] * params['time_stride'], params['round_float'])
        end = round(params['offset'] + word_ts_stt_end[1] * params['time_stride'], params['round_float'])
        return stt, end
    
    
    def _print_DER_info(self, uniq_id, string_out, pred_diar_labels, params):
        if params['color']:
            # color = '\033[0;37m'
            color = '\033[0;30m'
        else:
            color = ''
        der_dict, der_stat_dict = self.diar.online_eval_diarization(pred_diar_labels, self.rttm_file_path)
        DER, FA, MISS, CER = der_dict['DER'], der_dict['FA'], der_dict['MISS'], der_dict['CER']
        string_out += f'\n{color}============================================================================='
        string_out += f'\n{color}[Session: {uniq_id}, DER:{DER:.2f}%, FA:{FA:.2f}% MISS:{MISS:.2f}% CER:{CER:.2f}%]'
        string_out += f'\n{color}[Num of Speakers (Est/Ref): {der_stat_dict["est_n_spk"]}/{der_stat_dict["ref_n_spk"]}]'
        self.diar.DER_csv_list.append(f"{self.frame_index}, {DER}, {FA}, {MISS}, {CER}\n")
        write_txt(f"{asr.diar._out_dir}/{uniq_id}.csv", ''.join(self.diar.DER_csv_list))
        return string_out

    def _decode_and_cluster(self, frame, offset=0):
        torch.manual_seed(0)
        assert len(frame)==self.n_frame_len
        self.buffer_start = round(float(self.frame_index - 2*self.overlap_frames_count), 2)
        self.buffer[:-self.n_frame_len] = copy.deepcopy(self.buffer[self.n_frame_len:])
        self.buffer[-self.n_frame_len:] = copy.deepcopy(frame)
   
        text, word_ts = self.run_ASR_model(self.buffer)

        self.diar.frame_index = self.frame_index  
        SAD_timestamps_from_ASR = self._get_speech_labels_from_decoded_prediction(word_ts)

        if self.debug_mode:
            audio_signal, audio_lengths, speech_labels_used = self._get_diar_offline_segments()
        else:
            audio_signal, audio_lengths = self._get_diar_segments(SAD_timestamps_from_ASR)

        if self.buffer_start >= 0 and audio_signal != []:
            # logging.info(f"frame {self.frame_index}th, Segment range: {audio_lengths[0][0]}s - {audio_lengths[-1][-1]}s")
            labels = self._online_diarization(audio_signal, audio_lengths)
        else:
            labels = []

        self.frame_index += 1
        return text, word_ts, labels
    
    @timeit
    def _get_speech_labels_from_decoded_prediction(self, input_word_ts):
        _, buffer_end = self._get_update_abs_time(self.buffer_start)
        speech_labels = []
        word_ts = copy.deepcopy(input_word_ts)
        if word_ts == []:
            return speech_labels
        else:
            count = len(word_ts)-1
            while count > 0:
                if len(word_ts) > 1: 
                    if word_ts[count][0] - word_ts[count-1][1] <= self.nonspeech_threshold:
                        trangeB = word_ts.pop(count)
                        trangeA = word_ts.pop(count-1)
                        word_ts.insert(count-1, [trangeA[0], trangeB[1]])
                count -= 1
        return word_ts 

    def chunk_loader(self, samples, chunk_len_in_secs, context_len_in_secs):
        sample_rate = 16000
        buffer_list = []
        buffer_len_in_secs = int(chunk_len_in_secs + 2*context_len_in_secs)
        buffer_len = sample_rate*buffer_len_in_secs
        sampbuffer = np.zeros([buffer_len], dtype=np.float32)

        chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)	
        chunk_len = int(sample_rate*chunk_len_in_secs)
        count = 0
        for chunk in chunk_reader:
            count +=1
            sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
            sampbuffer[-chunk_len:] = chunk
            buffer_list.append(np.array(sampbuffer))
        
         
        if self.streaming_buffer_list == []:
            self.streaming_buffer_list = buffer_list
        else:
            self.streaming_buffer_list.pop(0)
            self.streaming_buffer_list.append(buffer_list[-1])

        return self.streaming_buffer_list, buffer_len_in_secs
   
    def apply_LM(self, log_probs):
        TOKEN_OFFSET = 100
        log_probs = asr_decoder.unmerged_logprobs.unsqueeze(0).cpu().numpy()
        # sm_probs = torch.exp(probs)
        lp_sm = kenlm_utils.softmax(log_probs[0])
        lp_sm = torch.from_numpy(lp_sm).unsqueeze(0).cpu().numpy()
        ids_to_text_func = self.new_asr_model.tokenizer.ids_to_text
        try:
            with nemo.core.typecheck.disable_checks():
                # beams_batch = self.lm_model.forward(log_probs=log_probs, log_probs_length=None)
                beams_batch = self.lm_model.forward(log_probs=lp_sm, log_probs_length=None)
        except:
            import ipdb; ipdb.set_trace() 
        for beams_idx, beams in enumerate(beams_batch):
            for candidate_idx, candidate in enumerate(beams[:1]):
                try:
                    if ids_to_text_func is not None:
                        # For BPE encodings, need to shift by TOKEN_OFFSET to retrieve the original sub-word ids
                        pred_text = ids_to_text_func([ord(c) - TOKEN_OFFSET for c in candidate[1]])
                    else:
                        pred_text = candidate[1]
                    print(pred_text)
                except:
                    import ipdb; ipdb.set_trace() 
        print("transcription1:", pred_text) 
        print("transcription2:", transcription) 
        import ipdb; ipdb.set_trace() 
        return pred_text

    @timeit
    def run_ASR_model(self, buffer):
        buffer_delay = self.overlap_frames_count
        chunk_len_in_secs = self.frame_len
        context_len_in_secs = self.frame_overlap
        buffer_list, buffer_len_in_secs = self.chunk_loader(buffer, chunk_len_in_secs, context_len_in_secs)
        asr_decoder = ChunkBufferDecoder(self.new_asr_model, stride=self.asr_stride, chunk_len_in_secs=chunk_len_in_secs, buffer_len_in_secs=buffer_len_in_secs)
        transcription, word_ts = asr_decoder.transcribe_buffers(self.buffer_start-buffer_delay, buffer_list, plot=False)
        if self.params['use_lm_for_asr']:
            self.apply_LM(log_probs)
        words = transcription.split()
        assert len(words) == len(word_ts)
        for k in range(len(word_ts)):
            word_ts[k] = [round(word_ts[k][0]-self.asr_delay_sec,2), round(word_ts[k][1]-self.asr_delay_sec,2)]
        return words, word_ts
    
    def compensate_word_ts_list(self, word_ts, params):
        """
        Compensate the word timestamps by using VAD output.
        The length of each word is capped by params['max_word_ts_length_in_sec'].

        Args:
            word_ts_list (list):
                Contains word_ts_stt_end lists.
                word_ts_stt_end = [stt, end]
                    stt: Start of the word in sec.
                    end: End of the word in sec.
            params (dict):
                Contains the parameters for diarization and ASR decoding.

        Return:
            enhanced_word_ts_list (list):
                Contains the amended word timestamps
        """
        enhanced_word_ts_list = []
        N = len(word_ts)
        enhanced_word_ts_buffer = []
        for k, word_stt_end in enumerate(word_ts_seq_list):
            if k < N - 1:
                word_len = round(word_stt_end[1] - word_stt_end[0], 2)
                len_to_next_word = round(word_ts_seq_list[k + 1][0] - word_stt_end[0] - 0.01, 2)
                vad_est_len = len_to_next_word  # Temporary
                min_candidate = min(vad_est_len, len_to_next_word)
                fixed_word_len = max(min(params['max_word_ts_length_in_sec'], min_candidate), word_len)
                enhanced_word_ts_buffer.append([word_stt_end[0], word_stt_end[0] + fixed_word_len])
        enhanced_word_ts_list.append(enhanced_word_ts_buffer)
        return enhanced_word_ts_list 

    @timeit
    def _get_diar_segments(self, SAD_timestamps_from_ASR, ROUND=2):
        if self.buffer_start >= 0:
            cursor_for_old_segments, buffer_end = self._get_update_abs_time(self.buffer_start)
            self.frame_start = round(self.buffer_start + int(self.n_frame_overlap/self.sr), ROUND)
            frame_end = self.frame_start + self.frame_len 
            
            if self.diar.segment_raw_audio_list == [] and SAD_timestamps_from_ASR != []:
                self.buffer_init_time = self.buffer_start

                SAD_timestamps_from_ASR[0][0] = max(SAD_timestamps_from_ASR[0][0], 0.0)
                speech_labels_initial = copy.deepcopy(SAD_timestamps_from_ASR)
                
                self.cumulative_speech_labels = speech_labels_initial
                
                source_buffer = copy.deepcopy(self.buffer)
                sigs_list, sig_rangel_list = self._get_segments_from_buffer(self.buffer_start, 
                                                                            speech_labels_initial, 
                                                                            source_buffer)
                self.diar.segment_raw_audio_list = sigs_list
                self.diar.segment_abs_time_range_list = sig_rangel_list
            
            else: 
                # Remove the old segments that overlap with the new frame (self.frame_start)
                # cursor_for_old_segments is set to the onset of the t_range popped lastly.
                cursor_for_old_segments = self._get_new_cursor_for_update()
                speech_labels_for_update = self._get_speech_labels_for_update(self.buffer_start, 
                                                                              buffer_end, 
                                                                              self.frame_start,
                                                                              SAD_timestamps_from_ASR,
                                                                              cursor_for_old_segments)
                
                source_buffer = copy.deepcopy(self.buffer)
                sigs_list, sig_rangel_list = self._get_segments_from_buffer(self.buffer_start, 
                                                                            speech_labels_for_update, 
                                                                            source_buffer)


                self.diar.segment_raw_audio_list.extend(sigs_list)
                self.diar.segment_abs_time_range_list.extend(sig_rangel_list)
                
        return self.diar.segment_raw_audio_list, \
               self.diar.segment_abs_time_range_list
    
    def _get_new_cursor_for_update(self):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is set to the onset of the t_range popped lastly.
        """
        cursor_for_old_segments = self.frame_start
        while True and len(self.diar.segment_raw_audio_list) > 0:
            t_range = self.diar.segment_abs_time_range_list[-1]

            mid = np.mean(t_range)
            if self.frame_start <= t_range[1]:
                self.diar.segment_abs_time_range_list.pop()
                self.diar.segment_raw_audio_list.pop()
                cursor_for_old_segments = t_range[0]
            else:
                break
        return cursor_for_old_segments

    def _get_update_abs_time(self, buffer_start):
        new_bufflen_sec = self.n_frame_len / self.sr
        n_buffer_samples = int(len(self.buffer)/self.sr)
        total_buffer_len_sec = n_buffer_samples/self.frame_len
        buffer_end = buffer_start + total_buffer_len_sec
        return (buffer_end - new_bufflen_sec), buffer_end

    def _get_speech_labels_for_update(self, buffer_start, buffer_end, frame_start, SAD_timestamps_from_ASR, cursor_for_old_segments):
        """
        Bring the new speech labels from the current buffer. Then
        1. Concatenate the old speech labels from self.cumulative_speech_labels for the overlapped region.
            - This goes to new_speech_labels.
        2. Update the new 1 sec of speech label (speech_label_for_new_segments) to self.cumulative_speech_labels.
        3. Return the speech label from cursor_for_old_segments to buffer end.

        """
        frame_start_to_buffer_end = [frame_start, buffer_end]
        
        if cursor_for_old_segments < frame_start:
            update_overlap_range = [cursor_for_old_segments, frame_start]
        else:
            update_overlap_range = []

        new_incoming_speech_labels = getSubRangeList(target_range=frame_start_to_buffer_end, 
                                                     source_list=SAD_timestamps_from_ASR)

        update_overlap_speech_labels = getSubRangeList(target_range=update_overlap_range, 
                                                       source_list=self.cumulative_speech_labels)
        
        speech_label_for_new_segments = getMergedRanges(update_overlap_speech_labels, 
                                                             new_incoming_speech_labels) 
        
        self.cumulative_speech_labels = getMergedRanges(self.cumulative_speech_labels, 
                                                             new_incoming_speech_labels) 
        return speech_label_for_new_segments

    def _get_segments_from_buffer(self, buffer_start, speech_labels_for_update, source_buffer, ROUND=3):
        sigs_list, sig_rangel_list = [], []
        n_seglen_samples = int(self.diar.embed_seg_len*self.sr)
        n_seghop_samples = int(self.diar.embed_seg_hop*self.sr)
        
        for idx, range_t in enumerate(speech_labels_for_update):
            if range_t[0] < 0:
                continue
            sigs, sig_lens = [], []
            stt_b = int((range_t[0] - buffer_start) * self.sr)
            end_b = int((range_t[1] - buffer_start) * self.sr)
            n_dur_samples = int(end_b - stt_b)
            base = math.ceil((n_dur_samples - n_seglen_samples) / n_seghop_samples)
            slices = 1 if base < 0 else base + 1
            sigs, sig_lens = self.get_segments_from_slices(slices, 
                                                      torch.from_numpy(source_buffer[stt_b:end_b]),
                                                      n_seglen_samples,
                                                      n_seghop_samples, 
                                                      sigs, 
                                                      sig_lens)

            sigs_list.extend(sigs)
            segment_offset = range_t[0]
            for seg_idx, sig_len in enumerate(sig_lens):
                seg_len_sec = float(sig_len / self.sr)
                start_abs_sec = round(float(segment_offset + seg_idx*self.diar.embed_seg_hop), ROUND)
                end_abs_sec = round(float(segment_offset + seg_idx*self.diar.embed_seg_hop + seg_len_sec), ROUND)
                sig_rangel_list.append([start_abs_sec, end_abs_sec])
        
        assert len(sigs_list) == len(sig_rangel_list)
        return sigs_list, sig_rangel_list

    def get_segments_from_slices(self, slices, sig, slice_length, shift, audio_signal, audio_lengths):
        """create short speech segments from sclices
        Args:
            slices (int): the number of slices to be created
            slice_length (int): the lenghth of each slice
            shift (int): the amount of slice window shift
            sig (FloatTensor): the tensor that contains input signal

        Returns:
            audio_signal (list): list of sliced input signal
            audio_lengths (list): list of audio sample lengths
        """
        for slice_id in range(slices):
            start_idx = int(slice_id * shift)
            end_idx = int(start_idx + slice_length)
            signal = sig[start_idx:end_idx]
            audio_lengths.append(len(signal))
            if len(signal) < slice_length:
                signal = repeat_signal(signal, len(signal), slice_length)
            audio_signal.append(signal)
            
        return audio_signal, audio_lengths
    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        
        text, word_ts, pred_diar_labels = self._decode_and_cluster(frame, offset=self.offset)
        return text, word_ts, pred_diar_labels
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s
    
    def greedy_merge_with_ts(self, s, buffer_start, ROUND=2):
        s_merged = ''
        char_ts = [] 
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
                    char_ts.append(round(buffer_start + i*self.time_stride, 2))
        end_stamp = buffer_start + len(s)*self.time_stride
        return s_merged, char_ts, end_stamp

def get_session_list(diar_init, args):
    if args.single_audio_file_path:
        uniq_key = get_uniq_id_from_audio_path(args.single_audio_file_path)
        diar_init.AUDIO_RTTM_MAP = {uniq_key: {'audio_path': args.single_audio_file_path,
                                               'rttm_path': args.single_rttm_file_path}}
    session_list = [ x[0] for x in  diar_init.AUDIO_RTTM_MAP.items() ]
    return diar_init, session_list


if __name__ == "__main__":
    SPK_EMBED_MODEL="/disk2/ejrvs/model_comparision/titanet-l.nemo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_speaker_model", default=SPK_EMBED_MODEL, type=str, help="")
    parser.add_argument("--device", default=0, type=int, help="")
    parser.add_argument("--single_audio_file_path", default=None, type=str, help="")
    parser.add_argument("--single_rttm_file_path", default=None, type=str, help="")
    parser.add_argument("--audiofile_list_path", default=None, type=str, help="")
    parser.add_argument("--reference_rttmfile_list_path", default=None, type=str, help="")
    parser.add_argument("--diarizer_out_dir", type=str, help="")
    parser.add_argument("--color", default=True, type=str, help="")
    parser.add_argument("--print_time", default=True, type=str, help="")
    parser.add_argument("--force_real_time", default=False, type=str, help="")
    parser.add_argument("--frame_len", default=1, type=int, help="")
    parser.add_argument("--frame_overlap", default=2, type=int, help="")
    parser.add_argument("--round_float", default=2, type=int, help="")
    parser.add_argument("--window_length_in_sec", default=1.5, type=float, help="")
    parser.add_argument("--shift_length_in_sec", default=0.75, type=float, help="")
    parser.add_argument("--print_transcript", default=False, type=bool, help="")
    parser.add_argument("--use_lm_for_asr", default=False, type=bool, help="")
    parser.add_argument("--lenient_overlap_WDER", default=True, type=bool, help="")
    parser.add_argument("--speech_detection_threshold", default=100, type=int, help="")
    parser.add_argument("--ASR_model_name", default='stt_en_conformer_ctc_large', type=str, help="")
    parser.add_argument("--sample_rate", default=16000, type=int, help="")
    parser.add_argument("--online_diar_buffer_segment_quantity", default=200, type=int, help="")
    parser.add_argument("--online_history_buffer_segment_quantity", default=100, type=int, help="")
    parser.add_argument("--enhanced_count_thres", default=0, type=int, help="")
    parser.add_argument("--max_num_speaker", default=8, type=int, help="")
    args = parser.parse_args()

    overrides = [
    f"diarizer.speaker_embeddings.model_path={args.pretrained_speaker_model}",
    f"diarizer.path2groundtruth_rttm_files={args.reference_rttmfile_list_path}",
    f"diarizer.paths2audio_files={args.audiofile_list_path}",
    f"diarizer.out_dir={args.diarizer_out_dir}",
    f"diarizer.speaker_embeddings.window_length_in_sec={args.window_length_in_sec}",
    f"diarizer.speaker_embeddings.shift_length_in_sec={args.shift_length_in_sec}",
    ]
    params = vars(args)
    hydra.initialize(config_path="conf")
    cfg_diar = hydra.compose(config_name="/speaker_diarization.yaml", overrides=overrides)
    diar_init = OnlineClusteringDiarizer(cfg=cfg_diar, params=params)
    diar_init.prepare_diarization()
    diar_init, session_list = get_session_list(diar_init, args)
    for uniq_key in session_list:
        diar = OnlineClusteringDiarizer(cfg=cfg_diar, params=params)
        diar.uniq_key = uniq_key 
        diar.prepare_diarization()
        asr = ASR_DIAR_ONLINE(diar, params, offset=4)
        asr.reset()
        
        dcont = diar_init.AUDIO_RTTM_MAP[uniq_key]
        samplerate, sdata = wavfile.read(dcont['audio_path'])
        asr.rttm_file_path = dcont['rttm_path']

        filePath = f"{asr.diar._out_dir}/{uniq_key}.csv"
        if os.path.exists(filePath): 
            os.remove(filePath)
        # input("Press Enter to continue...")
        for i in range(int(np.floor(sdata.shape[0]/asr.n_frame_len))):
            callback_sim(asr, uniq_key, i, sdata, frame_count=None, time_info=None, status=None)
