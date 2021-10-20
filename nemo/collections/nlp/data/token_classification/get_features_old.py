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

from typing import List, Optional

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_stats
from nemo.utils import logging


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
    all_subtokens = []
    all_loss_mask = []
    all_subtokens_mask = []
    all_segment_ids = []
    all_input_ids = []
    all_input_mask = []
    sent_lengths = []
    punct_all_labels = []
    capit_all_labels = []
    with_label = False

    if punct_labels_lines and capit_labels_lines:
        with_label = True

    for i, query in enumerate(queries):
        words = query.strip().split()

        # add bos token
        subtokens = [tokenizer.cls_token]
        loss_mask = [1 - ignore_start_end]
        subtokens_mask = [0]
        if with_label:
            pad_id = punct_label_ids[pad_label]
            punct_labels = [pad_id]
            punct_query_labels = [punct_label_ids[lab] for lab in punct_labels_lines[i]]

            capit_labels = [pad_id]
            capit_query_labels = [capit_label_ids[lab] for lab in capit_labels_lines[i]]

        for j, word in enumerate(words):
            word_tokens = tokenizer.text_to_tokens(word)
            subtokens.extend(word_tokens)

            loss_mask.append(1)
            loss_mask.extend([int(not ignore_extra_tokens)] * (len(word_tokens) - 1))

            subtokens_mask.append(1)
            subtokens_mask.extend([0] * (len(word_tokens) - 1))

            if with_label:
                punct_labels.extend([punct_query_labels[j]] * len(word_tokens))
                capit_labels.extend([capit_query_labels[j]] * len(word_tokens))

        # add eos token
        subtokens.append(tokenizer.sep_token)
        loss_mask.append(1 - ignore_start_end)
        subtokens_mask.append(0)
        sent_lengths.append(len(subtokens))
        all_subtokens.append(subtokens)
        all_loss_mask.append(loss_mask)
        all_subtokens_mask.append(subtokens_mask)
        all_input_mask.append([1] * len(subtokens))

        if with_label:
            punct_labels.append(pad_id)
            punct_all_labels.append(punct_labels)
            capit_labels.append(pad_id)
            capit_all_labels.append(capit_labels)

    max_seq_length = min(max_seq_length, max(sent_lengths))
    logging.info(f'Max length: {max_seq_length}')
    get_stats(sent_lengths)
    too_long_count = 0

    for i, subtokens in enumerate(all_subtokens):
        if len(subtokens) > max_seq_length:
            subtokens = [tokenizer.cls_token] + subtokens[-max_seq_length + 1 :]
            all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1 :]
            all_loss_mask[i] = [int(not ignore_start_end)] + all_loss_mask[i][-max_seq_length + 1 :]
            all_subtokens_mask[i] = [0] + all_subtokens_mask[i][-max_seq_length + 1 :]

            if with_label:
                punct_all_labels[i] = [pad_id] + punct_all_labels[i][-max_seq_length + 1 :]
                capit_all_labels[i] = [pad_id] + capit_all_labels[i][-max_seq_length + 1 :]
            too_long_count += 1

        all_input_ids.append(tokenizer.tokens_to_ids(subtokens))

        if len(subtokens) < max_seq_length:
            extra = max_seq_length - len(subtokens)
            all_input_ids[i] = all_input_ids[i] + [0] * extra
            all_loss_mask[i] = all_loss_mask[i] + [0] * extra
            all_subtokens_mask[i] = all_subtokens_mask[i] + [0] * extra
            all_input_mask[i] = all_input_mask[i] + [0] * extra

            if with_label:
                punct_all_labels[i] = punct_all_labels[i] + [pad_id] * extra
                capit_all_labels[i] = capit_all_labels[i] + [pad_id] * extra

        all_segment_ids.append([0] * max_seq_length)

    logging.info(f'{too_long_count} are longer than {max_seq_length}')

    for i in range(min(len(all_input_ids), 5)):
        logging.info("*** Example ***")
        logging.info("i: %s" % (i))
        logging.info("subtokens: %s" % " ".join(list(map(str, all_subtokens[i]))))
        logging.info("loss_mask: %s" % " ".join(list(map(str, all_loss_mask[i]))))
        logging.info("input_mask: %s" % " ".join(list(map(str, all_input_mask[i]))))
        logging.info("subtokens_mask: %s" % " ".join(list(map(str, all_subtokens_mask[i]))))
        if with_label:
            logging.info("punct_labels: %s" % " ".join(list(map(str, punct_all_labels[i]))))
            logging.info("capit_labels: %s" % " ".join(list(map(str, capit_all_labels[i]))))

    return (
        all_input_ids,
        all_segment_ids,
        all_input_mask,
        all_subtokens_mask,
        all_loss_mask,
        punct_all_labels,
        capit_all_labels,
    )
