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

import re
import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as pt_data
from apex.transformer import parallel_state, tensor_parallel
from omegaconf.dictconfig import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.enc_dec_model import MegatronEncDecWrapper
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import AppState, logging

from nemo.collections.common.tokenizers.bytelevel_tokenizers import ByteLevelProcessor
from nemo.collections.common.tokenizers.chinese_tokenizers import ChineseProcessor
from nemo.collections.common.tokenizers.en_ja_tokenizers import EnJaProcessor
from nemo.collections.common.tokenizers.indic_tokenizers import IndicProcessor
from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from nemo.collections.nlp.data import TarredTranslationDataset, TranslationDataset
from nemo.collections.common.data import ConcatDataset


class MegatronNMTModel(MegatronEncDecWrapper):
    """
    Training NMT models with Megatron.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        tokentype_ids=None,
        lm_labels=None,
        enc_hidden_states=None,
        output_enc_hidden=False,
    ):
        return super().forward(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
            tokentype_ids,
            lm_labels,
            enc_hidden_states,
            output_enc_hidden,
        )

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self(
            tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=labels
        )

        loss = self.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)
        # Reduced loss for logging.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            # self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step), prog_bar=True)
            self._reduced_loss_buffer = []

        return loss

    def validation_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self(
            tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=labels
        )
        loss = self.loss_func(loss_mask, output_tensor)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        
        # TODO: Add beam search decoding here and return generations.
        
        return reduced_loss

    def validation_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        self.log('val_loss', averaged_loss[0], prog_bar=True)
        # self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_batch(self, batch):
        """Build the batch."""
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch

        batch = {
            'src_ids': src_ids,
            'src_mask': src_mask,
            'tgt_ids': tgt_ids,
            'tgt_mask': tgt_mask,
            'labels': labels,
        }
        keys = ['src_ids', 'src_mask', 'tgt_ids', 'tgt_mask', 'labels']
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask'] < 0.5
        dec_mask = data_b['dec_mask'] < 0.5
        enc_dec_mask = data_b['enc_dec_mask'] < 0.5

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

    def compute_consumed_samples(self, global_step):
        app_state = AppState()
        consumed_samples = (
            global_step
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def configure_gradient_clipping(self, *args, **kwargs):
        """PTL hook to configure gradients.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        parameters = self.model.parameters()
        clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        request = batch
        response = self.complete(request)
        logging.info(f"response: {response}")
        return response

    def make_inference_attention_mask_3d(self, source_block, target_block, pad_id):
        """
        Returns a 3-dimensional (3-D) attention mask
        :param source_block: 2-D array
        :param target_block: 2-D array
        """
        mask = (target_block[:, None, :] != pad_id) * (source_block[:, :, None] != pad_id)
        return mask

    def make_inference_history_mask_3d(self, block):
        batch, length = block.shape
        arange = torch.arange(length, device=block.device)
        history_mask = (arange[None,] <= arange[:, None])[
            None,
        ]
        history_mask = history_mask.expand(batch, length, length)
        return history_mask

    def decode(self, tokens_enc, enc_mask, num_tokens_to_generate):
        encoder_hidden_states = self(
            encoder_input_ids=tokens_enc,
            decoder_input_ids=None,
            encoder_attn_mask=enc_mask,
            decoder_attn_mask=None,
            encoder_decoder_attn_mask=None,
            tokentype_ids=None,
            lm_labels=None,
            enc_hidden_states=None,
            output_enc_hidden=True,
        )
        predicted_tokens_dec = torch.LongTensor([self.tokenizer.bos_id]).unsqueeze(0).to(tokens_enc.device)

        for _ in range(num_tokens_to_generate):
            # Overwrite the decoder token since we want to predict
            enc_dec_mask = self.make_inference_attention_mask_3d(
                predicted_tokens_dec, tokens_enc, self.tokenizer.pad_id
            )
            dec_mask = self.make_inference_attention_mask_3d(
                predicted_tokens_dec, predicted_tokens_dec, self.tokenizer.pad_id
            )
            dec_mask = dec_mask * self.make_inference_history_mask_3d(predicted_tokens_dec)

            enc_dec_mask = enc_dec_mask < 0.5
            dec_mask = dec_mask < 0.5

            output_tensor, _ = self(
                encoder_input_ids=tokens_enc,
                decoder_input_ids=predicted_tokens_dec,
                encoder_attn_mask=enc_mask,
                decoder_attn_mask=dec_mask,
                encoder_decoder_attn_mask=enc_dec_mask,
                tokentype_ids=None,
                lm_labels=None,
                enc_hidden_states=encoder_hidden_states,
                output_enc_hidden=False,
            )
            output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
            log_probs, token_ids = torch.max(nn.functional.log_softmax(output_tensor, dim=-1), dim=-1)
            predicted_tokens_dec = torch.cat([predicted_tokens_dec, token_ids[:, -1].unsqueeze(1)], 1)
            if token_ids[:, -1] == self.tokenizer.eos_id:
                break

        return predicted_tokens_dec, log_probs

    def complete(self, request: Dict):
        """
            Autoregressively invokes language model in the inference mode
        Args:	
            request: Dictionary with the following fields
                * prompt: a string which text the model should complete.
                * tokens_to_generate: how many tokens to generate while doing prompt completion.
        Returns:	
            response: A python dictionary with the following fields
                * prompt: original text of the prompt
                * tokenized_prompt: list of (str) tokens from prompt
                * completion: a python dictionary with the following subfields:
                    * tokens: a list of triples (token, token_id, log_prob) comprising completion
                    * text: completion text (as a single string)
                
        """
        response = {}
        self.freeze()
        # naive greedy slow loop
        # TODO: add option for BeamSearchDecoder

        response['prompt'] = request['prompt'][0]
        response['completion'] = {}
        tokens_enc = request['masked_sample']

        response['masked_input'] = ' '.join(self.tokenizer.ids_to_tokens(tokens_enc[0]))
        enc_mask = self.make_inference_attention_mask_3d(tokens_enc, tokens_enc, self.tokenizer.pad_id)
        enc_mask = enc_mask < 0.5

        predicted_tokens_ids, log_probs = self.decode(tokens_enc, enc_mask, int(request['tokens_to_generate']))
        predicted_tokens_ids = predicted_tokens_ids.cpu().numpy()[0].tolist()
        log_probs = log_probs.cpu().numpy()[0].tolist()
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        else:
            predicted_tokens_ids = [id for id in predicted_tokens_ids if id != self.tokenizer.pad_id]
        predicted_tokens_dec = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        response['completion']['text'] = self.tokenizer.tokens_to_text(predicted_tokens_dec)
        response['completion']['tokens'] = list(zip(predicted_tokens_ids, predicted_tokens_dec, log_probs))
        self.unfreeze()
        return response

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after
    
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self.setup_validation_data(self._cfg.get('validation_ds'))

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self.setup_test_data(self._cfg.get('test_ds'))

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_eval_dataloader_from_config(cfg=val_data_config)
        # instantiate Torchmetric for each val dataloader
        if self._validation_dl is not None:
            for dataloader_idx in range(len(self._validation_dl)):
                if dataloader_idx == 0:
                    setattr(
                        self, f'val_loss', GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )
                else:
                    setattr(
                        self,
                        f'val_loss_{dataloader_idx}',
                        GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_eval_dataloader_from_config(cfg=test_data_config)
        # instantiate Torchmetric for each test dataloader
        if self._test_dl is not None:
            for dataloader_idx in range(len(self._test_dl)):
                if dataloader_idx == 0:
                    setattr(
                        self, f'test_loss', GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )
                else:
                    setattr(
                        self,
                        f'test_loss_{dataloader_idx}',
                        GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        if cfg.get("use_tarred_dataset", False):
            if cfg.get("metadata_file") is None:
                raise FileNotFoundError("Trying to use tarred data set but could not find metadata path in config.")
            metadata_file_list = cfg.get('metadata_file')
            tar_files_list = cfg.get('tar_files', None)
            if isinstance(metadata_file_list, str):
                metadata_file_list = [metadata_file_list]
            if tar_files_list is not None and isinstance(tar_files_list, str):
                tar_files_list = [tar_files_list]
            if tar_files_list is not None and len(tar_files_list) != len(metadata_file_list):
                raise ValueError('The config must have the same number of tarfile paths and metadata file paths.')

            datasets = []
            for idx, metadata_file in enumerate(metadata_file_list):
                with open(metadata_file) as metadata_reader:
                    metadata = json.load(metadata_reader)
                if tar_files_list is None:
                    tar_files = metadata.get('tar_files')
                    if tar_files is not None:
                        # update absolute path of tar files based on metadata_file path
                        valid_tar_files = []
                        metadata_basedir = os.path.abspath(os.path.dirname(metadata_file))
                        updated_fn = 0
                        for fn in tar_files:
                            # if a file does not exist, look in metadata file directory
                            if os.path.exists(fn):
                                valid_fn = fn
                            else:
                                updated_fn += 1
                                valid_fn = os.path.join(metadata_basedir, os.path.basename(fn))
                                if not os.path.exists(valid_fn):
                                    raise RuntimeError(
                                        f"File in tarred dataset is missing from absolute and relative paths {fn}"
                                    )

                            valid_tar_files.append(valid_fn)

                        tar_files = valid_tar_files

                        logging.info(f'Updated the path of {updated_fn} tarred files')
                        logging.info(f'Loading from tarred dataset {tar_files}')
                else:
                    tar_files = tar_files_list[idx]
                    if metadata.get('tar_files') is not None:
                        logging.info(
                            f'Tar file paths found in both cfg and metadata using one in cfg by default - {tar_files}'
                        )

                dataset = TarredTranslationDataset(
                    text_tar_filepaths=tar_files,
                    metadata_path=metadata_file,
                    encoder_tokenizer=self.encoder_tokenizer,
                    decoder_tokenizer=self.decoder_tokenizer,
                    shuffle_n=cfg.get("tar_shuffle_n", 100),
                    shard_strategy=cfg.get("shard_strategy", "scatter"),
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                    reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                    prepend_id=self.multilingual_ids[idx] if self.multilingual else None,
                )
                datasets.append(dataset)

            if len(datasets) > 1:
                dataset = ConcatDataset(
                    datasets=datasets,
                    sampling_technique=cfg.get('concat_sampling_technique'),
                    sampling_temperature=cfg.get('concat_sampling_temperature'),
                    sampling_probabilities=cfg.get('concat_sampling_probabilities'),
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                )
            else:
                dataset = datasets[0]
        else:
            src_file_list = cfg.src_file_name
            tgt_file_list = cfg.tgt_file_name
            if isinstance(src_file_list, str):
                src_file_list = [src_file_list]
            if isinstance(tgt_file_list, str):
                tgt_file_list = [tgt_file_list]

            if len(src_file_list) != len(tgt_file_list):
                raise ValueError('The same number of filepaths must be passed in for source and target.')

            datasets = []
            for idx, src_file in enumerate(src_file_list):
                dataset = TranslationDataset(
                    dataset_src=str(Path(src_file).expanduser()),
                    dataset_tgt=str(Path(tgt_file_list[idx]).expanduser()),
                    tokens_in_batch=cfg.tokens_in_batch,
                    clean=cfg.get("clean", False),
                    max_seq_length=cfg.get("max_seq_length", 512),
                    min_seq_length=cfg.get("min_seq_length", 1),
                    max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                    max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                    cache_ids=cfg.get("cache_ids", False),
                    cache_data_per_node=cfg.get("cache_data_per_node", False),
                    use_cache=cfg.get("use_cache", False),
                    reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                    prepend_id=self.multilingual_ids[idx] if self.multilingual else None,
                )
                dataset.batchify(self.encoder_tokenizer, self.decoder_tokenizer)
                datasets.append(dataset)

            if len(datasets) > 1:
                dataset = ConcatDataset(
                    datasets=datasets,
                    shuffle=cfg.get('shuffle'),
                    sampling_technique=cfg.get('concat_sampling_technique'),
                    sampling_temperature=cfg.get('concat_sampling_temperature'),
                    sampling_probabilities=cfg.get('concat_sampling_probabilities'),
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                )
            else:
                dataset = datasets[0]

        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=None if (cfg.get("use_tarred_dataset", False) or isinstance(dataset, ConcatDataset)) else sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    def _setup_eval_dataloader_from_config(self, cfg: DictConfig):
        src_file_name = cfg.get('src_file_name')
        tgt_file_name = cfg.get('tgt_file_name')

        if src_file_name is None or tgt_file_name is None:
            raise ValueError(
                'Validation dataloader needs both cfg.src_file_name and cfg.tgt_file_name to not be None.'
            )
        else:
            # convert src_file_name and tgt_file_name to list of strings
            if isinstance(src_file_name, str):
                src_file_list = [src_file_name]
            elif isinstance(src_file_name, ListConfig):
                src_file_list = src_file_name
            else:
                raise ValueError("cfg.src_file_name must be string or list of strings")
            if isinstance(tgt_file_name, str):
                tgt_file_list = [tgt_file_name]
            elif isinstance(tgt_file_name, ListConfig):
                tgt_file_list = tgt_file_name
            else:
                raise ValueError("cfg.tgt_file_name must be string or list of strings")
        if len(src_file_list) != len(tgt_file_list):
            raise ValueError('The same number of filepaths must be passed in for source and target validation.')

        dataloaders = []
        prepend_idx = 0
        for idx, src_file in enumerate(src_file_list):
            if self.multilingual:
                prepend_idx = idx
            dataset = TranslationDataset(
                dataset_src=str(Path(src_file).expanduser()),
                dataset_tgt=str(Path(tgt_file_list[idx]).expanduser()),
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                cache_ids=cfg.get("cache_ids", False),
                cache_data_per_node=cfg.get("cache_data_per_node", False),
                use_cache=cfg.get("use_cache", False),
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                prepend_id=self.multilingual_ids[prepend_idx] if self.multilingual else None,
            )
            dataset.batchify(self.encoder_tokenizer, self.decoder_tokenizer)

            if cfg.shuffle:
                sampler = pt_data.RandomSampler(dataset)
            else:
                sampler = pt_data.SequentialSampler(dataset)

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                sampler=sampler,
                num_workers=cfg.get("num_workers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
            dataloaders.append(dataloader)

        return dataloaders

    def setup_pre_and_post_processing_utils(self, source_lang, target_lang):
        """
        Creates source and target processor objects for input and output pre/post-processing.
        """
        self.source_processor, self.target_processor = None, None

        if self.encoder_tokenizer_library == 'byte-level':
            self.source_processor = ByteLevelProcessor()
        elif (source_lang == 'en' and target_lang == 'ja') or (source_lang == 'ja' and target_lang == 'en'):
            self.source_processor = EnJaProcessor(source_lang)
        elif source_lang == 'zh':
            self.source_processor = ChineseProcessor()
        elif source_lang == 'hi':
            self.source_processor = IndicProcessor(source_lang)
        elif source_lang is not None and source_lang not in ['ja', 'zh', 'hi']:
            self.source_processor = MosesProcessor(source_lang)
        elif source_lang == 'ignore':
            self.source_processor = None

        if self.decoder_tokenizer_library == 'byte-level':
            self.target_processor = ByteLevelProcessor()
        elif (source_lang == 'en' and target_lang == 'ja') or (source_lang == 'ja' and target_lang == 'en'):
            self.target_processor = EnJaProcessor(target_lang)
        elif target_lang == 'zh':
            self.target_processor = ChineseProcessor()
        elif target_lang == 'hi':
            self.target_processor = IndicProcessor(target_lang)
        elif target_lang is not None and target_lang not in ['ja', 'zh', 'hi']:
            self.target_processor = MosesProcessor(target_lang)
        elif target_lang == 'ignore':
            self.target_processor == None

        return self.source_processor, self.target_processor

    def list_available_models():
        pass

