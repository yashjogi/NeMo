import itertools
import multiprocessing as mp
from typing import List, Optional

import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_stats
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
    pad_id = punct_label_ids[pad_label]
    max_seq_length = min(max_seq_length, max([len(s) for s in input_ids]))
    logging.info(f'Max length: {max_seq_length}')
    get_stats(sent_lengths)
    too_long_count = 0
    logging.info("Pad...")
    segment_ids = []
    for i, inp in enumerate(input_ids):
        if len(inp) < max_seq_length:
            extra = max_seq_length - len(inp)
            input_ids[i] = np.concatenate([inp, np.zeros([extra], dtype=np.int32)])
            loss_mask[i] = np.concatenate([loss_mask[i], np.zeros([extra], dtype=bool)])
            subtokens_mask[i] = np.concatenate([subtokens_mask[i], np.zeros([extra], dtype=bool)])
            input_mask[i] = np.concatenate([input_mask[i], np.zeros([extra], dtype=bool)])

            if with_label:
                punct_labels[i] = np.concatenate([punct_labels[i], np.full([extra], pad_id, dtype=np.int32)])
                capit_labels[i] = np.concatenate([capit_labels[i], np.full([extra], pad_id, dtype=np.int32)])
        segment_ids.append(np.zeros([max_seq_length], dtype=np.int8))
    logging.info(f"Finished clipping and padding.")
    logging.info(f'{too_long_count} are longer than {max_seq_length}')

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

    return (
        np.stack(input_ids),
        np.stack(segment_ids),
        np.stack(input_mask),
        np.stack(subtokens_mask),
        np.stack(loss_mask),
        np.stack(punct_labels),
        np.stack(capit_labels),
    )

