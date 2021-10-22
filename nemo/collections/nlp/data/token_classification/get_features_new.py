import itertools
import multiprocessing as mp
import os
import pickle
import random
from math import ceil
from typing import Dict, List, Optional

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_label_stats, get_stats
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging


MAX_NUM_QUERIES_IN_SPLIT = 10 ** 4


class TokenizeCreateMasksClipWorker:
    def __init__(
        self,
        max_seq_length,
        tokenizer,
        ignore_start_end,
        punct_label_ids,
        capit_label_ids,
        pad_label,
        ignore_extra_tokens,
        with_label
    ):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.ignore_start_end = ignore_start_end
        self.punct_label_ids = punct_label_ids
        self.capit_label_ids = capit_label_ids
        self.pad_label = pad_label
        self.ignore_extra_tokens = ignore_extra_tokens
        self.with_label = with_label

    def maybe_clip(self, values, prepend_value):
        if len(values) > self.max_seq_length:
            return [prepend_value] + values[-self.max_seq_length + 1:]
        return values

    def __call__(self, queries, punct_labels_lines, capit_labels_lines, split_i):
        all_input_ids, all_loss_mask, all_subtokens_mask, all_input_mask, sent_lengths = [], [], [], [], []
        punct_all_labels, capit_all_labels = [], []
        for i, query in enumerate(queries):
            words = query.strip().split()
            input_ids, loss_mask, subtokens_mask = [self.tokenizer.cls_id], [1 - self.ignore_start_end], [0]
            if self.with_label:
                pad_id = self.punct_label_ids[self.pad_label]
                punct_labels = [pad_id]
                punct_query_labels = [self.punct_label_ids[lab] for lab in punct_labels_lines[i]]
                capit_labels = [pad_id]
                capit_query_labels = [self.capit_label_ids[lab] for lab in capit_labels_lines[i]]
            for j, word in enumerate(words):
                word_ids = self.tokenizer.text_to_ids(word)
                input_ids.extend(word_ids)

                loss_mask.append(1)
                loss_mask.extend([int(not self.ignore_extra_tokens)] * (len(word_ids) - 1))

                subtokens_mask.append(1)
                subtokens_mask.extend([0] * (len(word_ids) - 1))

                if self.with_label:
                    punct_labels.extend([punct_query_labels[j]] * len(word_ids))
                    capit_labels.extend([capit_query_labels[j]] * len(word_ids))

            # add eos token
            input_ids.append(self.tokenizer.sep_id)
            loss_mask.append(1 - self.ignore_start_end)
            subtokens_mask.append(0)
            sent_lengths.append(len(input_ids))

            all_input_ids.append(np.array(self.maybe_clip(input_ids, self.tokenizer.cls_id), dtype=np.int32))
            all_loss_mask.append(np.array(self.maybe_clip(loss_mask, 1 - self.ignore_start_end), dtype=bool))
            all_subtokens_mask.append(np.array(self.maybe_clip(subtokens_mask, 0), dtype=bool))
            all_input_mask.append(np.array([1] * len(all_input_ids[-1]), dtype=bool))

            if self.with_label:
                punct_labels.append(pad_id)
                punct_all_labels.append(np.array(self.maybe_clip(punct_labels, pad_id), dtype=np.int32))
                capit_labels.append(pad_id)
                capit_all_labels.append(np.array(self.maybe_clip(capit_labels, pad_id), dtype=np.int32))
        logging.info(f"Finished tokenization processing split with number {split_i}")
        return (
            all_input_ids,
            all_loss_mask,
            all_subtokens_mask,
            all_input_mask,
            sent_lengths,
            punct_all_labels,
            capit_all_labels,
        )


def tokenize_create_masks_clip_parallel(
    queries,
    max_seq_length,
    tokenizer,
    ignore_start_end,
    punct_label_ids,
    capit_label_ids,
    punct_labels_lines,
    capit_labels_lines,
    pad_label,
    ignore_extra_tokens,
    with_label,
    njobs,
):
    if njobs is None:
        njobs = mp.cpu_count()
    logging.info(f"Running tokenization with {njobs} jobs.")
    num_queries_in_split = min(len(queries) // njobs, MAX_NUM_QUERIES_IN_SPLIT)
    n_split = len(queries) // num_queries_in_split
    split_queries = (
        [queries[num_queries_in_split * i : num_queries_in_split * (i + 1)] for i in range(n_split - 1)]
        + [queries[num_queries_in_split * (n_split - 1) :]]
    )
    split_punct_labels_lines = (
        [punct_labels_lines[num_queries_in_split * i : num_queries_in_split * (i + 1)] for i in range(n_split - 1)]
        + [punct_labels_lines[num_queries_in_split * (n_split - 1) :]]
    )
    split_capit_labels_lines = (
        [capit_labels_lines[num_queries_in_split * i: num_queries_in_split * (i + 1)] for i in range(n_split - 1)]
        + [capit_labels_lines[num_queries_in_split * (n_split - 1):]]
    )
    args = list(zip(split_queries, split_punct_labels_lines, split_capit_labels_lines, range(n_split)))
    with mp.Pool(njobs) as pool:
        result = pool.starmap(
            TokenizeCreateMasksClipWorker(
                max_seq_length,
                tokenizer,
                ignore_start_end,
                punct_label_ids,
                capit_label_ids,
                pad_label,
                ignore_extra_tokens,
                with_label
            ),
            args,
        )
    return tuple(list(itertools.chain(*e)) for e in zip(*result))


def get_features(
    queries: List[str],
    max_seq_length: int,
    tokenizer: TokenizerSpec,
    punct_label_ids: dict = None,
    capit_label_ids: dict = None,
    pad_label: str = 'O',
    punct_labels_lines=None,
    capit_labels_lines=None,
    ignore_extra_tokens=False,
    ignore_start_end: Optional[bool] = False,
    njobs: Optional[int] = None,
):
    """
    Processes the data and returns features.

    Args:
        queries: text sequences
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as AutoTokenizer
        pad_label: pad value use for labels. By default, it's the neutral label.
        punct_label_ids: dict to map punctuation labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        capit_label_ids: dict to map labels to label ids. Starts
            with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        punct_labels: list of labels for every word in a sequence (str)
        capit_labels: list of labels for every word in a sequence (str)
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask

    Returns:
        all_input_ids: input ids for all tokens
        all_segment_ids: token type ids
        all_input_mask: attention mask to use for BERT model
        all_subtokens_mask: masks out all subwords besides the first one
        all_loss_mask: loss mask to mask out tokens during training
        punct_all_labels: all labels for punctuation task (ints)
        capit_all_labels: all labels for capitalization task (ints)
        punct_label_ids: label (str) to id (int) map for punctuation task
        capit_label_ids: label (str) to id (int) map for capitalization task
    """
    with_label = punct_labels_lines and capit_labels_lines
    logging.info("Start initial tokenization.")
    res = tokenize_create_masks_clip_parallel(
        queries,
        max_seq_length,
        tokenizer,
        ignore_start_end,
        punct_label_ids,
        capit_label_ids,
        punct_labels_lines,
        capit_labels_lines,
        pad_label,
        ignore_extra_tokens,
        with_label,
        njobs,
    )
    input_ids, loss_mask, subtokens_mask, input_mask, sent_lengths, punct_labels, capit_labels = res
    logging.info("Finished initial tokenization.")
    get_stats(sent_lengths)
    segment_ids = [np.zeros([inp.shape[0]], dtype=np.int8) for inp in input_ids]
    logging.info(f"Finished clipping and padding.")

    for i in range(min(len(input_ids), 5)):
        logging.info("*** Example ***")
        logging.info("i: %s" % (i))
        logging.info("subtokens: %s" % " ".join(list(map(str, input_ids[i]))))
        logging.info("loss_mask: %s" % " ".join(list(map(str, loss_mask[i]))))
        logging.info("input_mask: %s" % " ".join(list(map(str, input_mask[i]))))
        logging.info("subtokens_mask: %s" % " ".join(list(map(str, subtokens_mask[i]))))
        if with_label:
            logging.info("punct_labels: %s" % " ".join(list(map(str, punct_labels[i]))))
            logging.info("capit_labels: %s" % " ".join(list(map(str, capit_labels[i]))))

    return input_ids, segment_ids, input_mask, subtokens_mask, loss_mask, punct_labels, capit_labels


class BertPunctuationCapitalizationDatasetOld(Dataset):
    """
    Creates dataset to use during training for punctuaion and capitalization tasks with a pretrained model.
    For dataset to use during inference without labels, see BertPunctuationCapitalizationInferDataset.

    Args:
        text_file: file to sequences, each line should a sentence, no header.
        label_file: file to labels, each line corresponds to word labels for a sentence in the text_file. No header.
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as AutoTokenizer
        num_samples: number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        pad_label: pad value use for labels.
            by default, it's the neutral label.
        punct_label_ids and capit_label_ids (dict):
            dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order
            For dev set use label_ids generated during training to support
            cases when not all labels are present in the dev set.
            For training set label_ids should be None or loaded from cache
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask
        use_cache: whether to use processed data cache or not
        get_label_frequencies: whether to generate label frequencies
        punct_label_ids_file and capit_label_ids_file: name of the files to save in .nemo
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'punct_labels': NeuralType(('B', 'T'), LabelsType()),
            'capit_labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        text_file: str,
        label_file: str,
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        num_samples: int = -1,
        tokens_in_batch: int = 1024,
        pad_label: str = 'O',
        punct_label_ids: Dict[str, int] = None,
        capit_label_ids: Dict[str, int] = None,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = False,
        use_cache: bool = True,
        get_label_frequencies: bool = False,
        punct_label_ids_file: str = 'punct_label_ids.csv',
        capit_label_ids_file: str = 'capit_label_ids.csv',
        njobs: Optional[int] = None,
    ):
        """ Initializes BertPunctuationCapitalizationDataset. """

        if not (os.path.exists(text_file) and os.path.exists(label_file)):
            raise FileNotFoundError(
                f'{text_file} or {label_file} not found. The data should be splitted into 2 files: text.txt and \
                labels.txt. Each line of the text.txt file contains text sequences, where words are separated with \
                spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are \
                separated with spaces. Each line of the files should follow the format:  \
                   [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
                   [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )

        # Cache features
        data_dir = os.path.dirname(text_file)
        filename = os.path.basename(text_file)

        if not filename.endswith('.txt'):
            raise ValueError("{text_file} should have extension .txt")

        self.tokens_in_batch = tokens_in_batch
        self.tokenizer = tokenizer
        self.pad_label = pad_label
        filename = filename[:-4]
        vocab_size = getattr(self.tokenizer, "vocab_size", 0)
        features_pkl = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                filename, self.tokenizer.name, str(max_seq_length), str(vocab_size), str(num_samples)
            ),
        )

        self.punct_label_ids_file = os.path.join(data_dir, punct_label_ids_file)
        self.capit_label_ids_file = os.path.join(data_dir, capit_label_ids_file)

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        cache_files_exist = (
            os.path.exists(features_pkl)
            and os.path.exists(self.punct_label_ids_file)
            and os.path.exists(self.capit_label_ids_file)
        )
        features = None
        if master_device and not (cache_files_exist and use_cache):
            if num_samples == 0:
                raise ValueError("num_samples has to be positive", num_samples)
            logging.info(f'Processing {text_file}')
            with open(text_file, 'r') as f:
                text_lines = f.readlines()

            # Collect all possible labels
            punct_unique_labels = set()
            capit_unique_labels = set()
            punct_labels_lines = []
            capit_labels_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip().split()

                    # extract punctuation and capitalization labels
                    punct_line, capit_line = zip(*line)
                    punct_labels_lines.append(punct_line)
                    capit_labels_lines.append(capit_line)

                    punct_unique_labels.update(punct_line)
                    capit_unique_labels.update(capit_line)

            if len(punct_labels_lines) != len(text_lines):
                raise ValueError("Labels file should contain labels for every word")

            dataset = list(zip(text_lines, punct_labels_lines, capit_labels_lines))

            if num_samples > 0:
                dataset = dataset[:num_samples]

            dataset = list(zip(*dataset))
            text_lines = dataset[0]
            punct_labels_lines = dataset[1]
            capit_labels_lines = dataset[2]

            # for dev/test sets use label mapping from training set
            if punct_label_ids:
                if len(punct_label_ids) != len(punct_unique_labels):
                    logging.info(
                        'Not all labels from the specified'
                        + 'label_ids dictionary are present in the'
                        + 'current dataset. Using the provided'
                        + 'label_ids dictionary.'
                    )
                else:
                    logging.info('Using the provided label_ids dictionary.')
            else:
                logging.info(
                    'Creating a new label to label_id dictionary.'
                    + ' It\'s recommended to use label_ids generated'
                    + ' during training for dev/test sets to avoid'
                    + ' errors if some labels are not'
                    + ' present in the dev/test sets.'
                    + ' For training set label_ids should be None.'
                )

                def create_label_ids(unique_labels, pad_label=self.pad_label):
                    label_ids = {pad_label: 0}
                    if pad_label in unique_labels:
                        unique_labels.remove(pad_label)
                    for label in sorted(unique_labels):
                        label_ids[label] = len(label_ids)
                    return label_ids

                punct_label_ids = create_label_ids(punct_unique_labels)
                capit_label_ids = create_label_ids(capit_unique_labels)

            self._save_label_ids(punct_label_ids, self.punct_label_ids_file)
            self._save_label_ids(capit_label_ids, self.capit_label_ids_file)

            features = get_features(
                text_lines,
                max_seq_length,
                self.tokenizer,
                pad_label=self.pad_label,
                punct_labels_lines=punct_labels_lines,
                capit_labels_lines=capit_labels_lines,
                punct_label_ids=punct_label_ids,
                capit_label_ids=capit_label_ids,
                ignore_extra_tokens=ignore_extra_tokens,
                ignore_start_end=ignore_start_end,
                njobs=njobs,
            )

            pickle.dump(tuple(list(features) + [punct_label_ids, capit_label_ids]), open(features_pkl, "wb"))
            logging.info(f'Features saved to {features_pkl}')

        # wait until the master process writes to the processed data files
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if features is None:
            features = pickle.load(open(features_pkl, 'rb'))
            punct_label_ids, capit_label_ids = features[-2], features[-1]
            features = features[:-2]
            logging.info(f'Features restored from {features_pkl}')

        input_ids = features[0]
        segment_ids = features[1]
        input_mask = features[2]
        subtokens_mask = features[3]
        loss_mask = features[4]
        punct_labels = features[5]
        capit_labels = features[6]
        self.punct_label_ids = punct_label_ids
        self.capit_label_ids = capit_label_ids
        self.batches = self.pack_into_batches(
            input_ids, segment_ids, input_mask, subtokens_mask, loss_mask, punct_labels, capit_labels
        )

        if get_label_frequencies:
            self.punct_label_frequencies = self._calculate_label_frequencies(self.punct_all_labels, data_dir, 'punct')
            self.capit_label_frequencies = self._calculate_label_frequencies(self.capit_all_labels, data_dir, 'capit')

    def pad(self, vectors, length, value):
        result = []
        for v in vectors:
            result.append(np.concatenate([v, np.full([length - v.shape[0]], value, dtype=v.dtype)]))
        return np.stack(result)

    def pack_into_batches(
        self, input_ids, segment_ids, input_mask, subtokens_mask, loss_mask, punct_labels, capit_labels
    ):
        zipped = sorted(
            zip(input_ids, segment_ids, input_mask, subtokens_mask, loss_mask, punct_labels, capit_labels),
            key=lambda x: x[0].shape[0]
        )
        input_ids, segment_ids, input_mask, subtokens_mask, loss_mask, punct_labels, capit_labels = zip(*zipped)
        batch_beginnings, batch_sizes, batch_seq_lengths = [], [], []
        current_max_length = 0
        start = 0
        for i, inp in enumerate(input_ids):
            current_max_length = max(current_max_length, ceil(len(inp) / 8) * 8)
            if current_max_length * (i + 1 - start) > self.tokens_in_batch:
                batch_size = (i - start) // 8 * 8
                seq_length = ceil(max([len(inp) for inp in input_ids[start : start + batch_size]]) / 8) * 8
                batch_beginnings.append(start)
                batch_sizes.append(batch_size)
                batch_seq_lengths.append(seq_length)
                start = start + batch_size
                current_max_length = ceil(
                    max([len(inp) for inp in input_ids[start + batch_size : i + 1]]) / 8
                ) * 8
        if start < len(input_ids):
            seq_length = ceil(max([len(inp) for inp in input_ids[start :]]) / 8) * 8
            batch_beginnings.append(start)
            batch_sizes.append(len(input_ids) - start)
            batch_seq_lengths.append(seq_length)
        batches = []
        for start, size, length in zip(batch_beginnings, batch_sizes, batch_seq_lengths):
            batch = {
                "input_ids": self.pad(input_ids[start : start + size], length, self.tokenizer.pad_id),
                "segment_ids": self.pad(segment_ids[start : start + size], length, 0).astype(np.int32),
                "input_mask": self.pad(input_mask[start : start + size], length, False),
                "subtokens_mask": self.pad(subtokens_mask[start : start + size], length, False),
                "loss_mask": self.pad(loss_mask[start : start + size], length, False),
                "punct_labels": self.pad(
                    punct_labels[start : start + size], length, self.punct_label_ids[self.pad_label]
                ).astype(np.int64),
                "capit_labels": self.pad(
                    capit_labels[start : start + size], length, self.capit_label_ids[self.pad_label]
                ).astype(np.int64),
            }
            batches.append(batch)
        random.shuffle(batches)
        return batches

    def _calculate_label_frequencies(self, all_labels: List[int], data_dir: str, name: str) -> Dict[str, float]:
        """ Calculates labels frequencies """
        merged_labels = itertools.chain.from_iterable(all_labels)
        logging.info('Three most popular labels')
        _, label_frequencies, _ = get_label_stats(merged_labels, data_dir + '/label_count_' + name + '.tsv')
        return label_frequencies

    def _save_label_ids(self, label_ids: Dict[str, int], filename: str) -> None:
        """ Saves label ids map to a file """
        with open(filename, 'w') as out:
            labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
            out.write('\n'.join(labels))
            logging.info(f'Labels: {label_ids}')
            logging.info(f'Labels mapping saved to : {out.name}')

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return self.batches[i]