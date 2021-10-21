import itertools
import multiprocessing as mp

from nemo.utils import logging


MAX_NUM_QUERIES_IN_SPLIT = 10 ** 4


class TokenizeAndCreateMasksWorker:
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

    def __call__(self, queries, punct_labels_lines, capit_labels_lines, split_i):
        all_subtokens, all_loss_mask, all_subtokens_mask, all_input_mask, sent_lengths = [], [], [], [], []
        punct_all_labels, capit_all_labels = [], []
        for i, query in enumerate(queries):
            words = query.strip().split()
            subtokens, loss_mask, subtokens_mask = [self.tokenizer.cls_token], [1 - self.ignore_start_end], [0]
            if self.with_label:
                pad_id = self.punct_label_ids[self.pad_label]
                punct_labels = [pad_id]
                punct_query_labels = [self.punct_label_ids[lab] for lab in punct_labels_lines[i]]
                capit_labels = [pad_id]
                capit_query_labels = [self.capit_label_ids[lab] for lab in capit_labels_lines[i]]
            for j, word in enumerate(words):
                word_tokens = self.tokenizer.text_to_tokens(word)
                subtokens.extend(word_tokens)

                loss_mask.append(1)
                loss_mask.extend([int(not self.ignore_extra_tokens)] * (len(word_tokens) - 1))

                subtokens_mask.append(1)
                subtokens_mask.extend([0] * (len(word_tokens) - 1))

                if self.with_label:
                    punct_labels.extend([punct_query_labels[j]] * len(word_tokens))
                    capit_labels.extend([capit_query_labels[j]] * len(word_tokens))

            # add eos token
            subtokens.append(self.tokenizer.sep_token)
            loss_mask.append(1 - self.ignore_start_end)
            subtokens_mask.append(0)
            sent_lengths.append(len(subtokens))
            all_subtokens.append(subtokens)
            all_loss_mask.append(loss_mask)
            all_subtokens_mask.append(subtokens_mask)
            all_input_mask.append([1] * len(subtokens))

            if self.with_label:
                punct_labels.append(pad_id)
                punct_all_labels.append(punct_labels)
                capit_labels.append(pad_id)
                capit_all_labels.append(capit_labels)
        logging.info(f"Finished tokenization processing split with number {self.split_i}")
        return (
            all_subtokens,
            all_loss_mask,
            all_subtokens_mask,
            all_input_mask,
            sent_lengths,
            punct_all_labels,
            capit_all_labels,
        )


# def tokenize_and_create_masks(args):
#     (
#         queries,
#         tokenizer,
#         ignore_start_end,
#         punct_label_ids,
#         capit_label_ids,
#         punct_labels_lines,
#         capit_labels_lines,
#         pad_label,
#         ignore_extra_tokens,
#         with_label,
#         split_i,
#     ) = args
#     all_subtokens, all_loss_mask, all_subtokens_mask, all_input_mask, sent_lengths = [], [], [], [], []
#     punct_all_labels, capit_all_labels = [], []
#     for i, query in enumerate(queries):
#         words = query.strip().split()
#         subtokens, loss_mask, subtokens_mask = [tokenizer.cls_token], [1 - ignore_start_end], [0]
#         if with_label:
#             pad_id = punct_label_ids[pad_label]
#             punct_labels, punct_query_labels = [pad_id], [punct_label_ids[lab] for lab in punct_labels_lines[i]]
#             capit_labels, capit_query_labels = [pad_id], [capit_label_ids[lab] for lab in capit_labels_lines[i]]
#         for j, word in enumerate(words):
#             word_tokens = tokenizer.text_to_tokens(word)
#             subtokens.extend(word_tokens)
#
#             loss_mask.append(1)
#             loss_mask.extend([int(not ignore_extra_tokens)] * (len(word_tokens) - 1))
#
#             subtokens_mask.append(1)
#             subtokens_mask.extend([0] * (len(word_tokens) - 1))
#
#             if with_label:
#                 punct_labels.extend([punct_query_labels[j]] * len(word_tokens))
#                 capit_labels.extend([capit_query_labels[j]] * len(word_tokens))
#
#         # add eos token
#         subtokens.append(tokenizer.sep_token)
#         loss_mask.append(1 - ignore_start_end)
#         subtokens_mask.append(0)
#         sent_lengths.append(len(subtokens))
#         all_subtokens.append(subtokens)
#         all_loss_mask.append(loss_mask)
#         all_subtokens_mask.append(subtokens_mask)
#         all_input_mask.append([1] * len(subtokens))
#
#         if with_label:
#             punct_labels.append(pad_id)
#             punct_all_labels.append(punct_labels)
#             capit_labels.append(pad_id)
#             capit_all_labels.append(capit_labels)
#     logging.info(f"Finished tokenization processing split with number {split_i}")
#     return (
#         all_subtokens,
#         all_loss_mask,
#         all_subtokens_mask,
#         all_input_mask,
#         sent_lengths,
#         punct_all_labels,
#         capit_all_labels,
#     )


def tokenize_and_create_masks_parallel(
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
            TokenizeAndCreateMasksWorker(
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