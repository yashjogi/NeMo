import argparse
import logging
import os
import random
import re
import string
import subprocess
from copy import deepcopy
from io import StringIO
from math import ceil
from pathlib import Path

import fasttext
import requests
from bs4 import BeautifulSoup, NavigableString

from nemo.collections.nlp.modules import get_tokenizer

import prepare_small_data_for_punctuation_capitalization_task as small


logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')


random.seed(42)


SUPPORTED_CORPUS_TYPES = ["wikipedia"]

PAGE_OPENING_NORMAL_TAG = re.compile(r'^ *<page>$')
PAGE_CLOSING_NORMAL_TAG = re.compile(r'^ *</page>$')


def preprocess_wikipedia(file_path):
    with file_path.open() as f:
        page = ""
        page_in_progress = False
        for i, line in enumerate(f):
            if '<page' in line:
                if PAGE_OPENING_NORMAL_TAG.match(line) is None:
                    logging.warning(f'Encountered an unusual page opening tag in line {i} {repr(line)}')
                page_in_progress = True
            if '</page' in line:
                if PAGE_CLOSING_NORMAL_TAG.match(line) is None:
                    logging.warning(f'Encountered an unusual page opening tag in line {i} {repr(line)}')
                if not page_in_progress:
                    logging.warning(f'Encountered closing page tag without opening tag. Line: {i}')
                elif not page:
                    logging.warning(f"Encountered a page which takes only one line. Line: {i}")
            if page_in_progress:
                page += line




def main():
    args = small.get_args(SUPPORTED_CORPUS_TYPES)
    tokenizer = get_tokenizer(args.tokenizer)
    all_docs = {}
    number_of_sentences_in_input = 0
    for corpus_type, file_path in zip(args.corpus_types, args.input_files):
        logging.info(f"Processing file {file_path}..")
        if corpus_type == SUPPORTED_CORPUS_TYPES[0]:
            file_docs, number_of_sentences_in_input = preprocess_wikipedia(file_path)
        else:
            raise ValueError(
                f"Unsupported corpus type '{corpus_type}. Supported corpus types are {SUPPORTED_CORPUS_TYPES}"
            )
    if args.size is None:
        args.size = number_of_sentences_in_input
    sentences_by_number_of_words = arrange_sentences_by_number_of_words(all_docs, args.sequence_length_range)
    if (
        sum([len(x) for x in sentences_by_number_of_words.values()])
        < args.size * args.percentage_segments_with_intact_sentences / 100
    ):
        raise ValueError(
            f"Cannot find enough segments consisting of whole sentences to build dataset with {args.size} segments "
            f"and at least {args.percentage_segments_with_intact_sentences}% segments consisting of whole sentences. "
            f"Try to reduce dataset size of parameter `--percentage_segments_with_intact_sentences"
        )
    result, number_of_words_stats, selected_by_docs = select_close_to_uniform_distribution(
        sentences_by_number_of_words, args.size, args.percentage_segments_with_intact_sentences, all_docs
    )
    for i in range(len(result)):
        result[i] = ' '.join(all_docs[result[i][0]][result[i][1] : result[i][2]])
    result += create_not_whole_sentence_segments(
        all_docs, selected_by_docs, number_of_words_stats, args.size, args.percentage_segments_with_intact_sentences,
    )
    result = list(set(result))
    random.shuffle(result)
    if args.dev_size > len(result):
        raise ValueError(f"Parameter `--dev_size={args.dev_size}` is less than size of all dataset ({len(result)})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_size = int(args.size * args.test_ratio / 100)
    if test_size > 0:
        write_dataset(
            result[:test_size],
            args.output_dir / Path("test"),
            args.create_model_input,
            args.bert_labels,
            args.autoregressive_labels,
            args.allowed_punctuation,
            args.only_first_punctuation_character_after_word_in_autoregressive,
            args.no_label_if_all_characters_are_upper_case,
        )
    if args.dev_size > 0:
        write_dataset(
            result[test_size : test_size + args.dev_size],
            args.output_dir / Path("dev"),
            args.create_model_input,
            args.bert_labels,
            args.autoregressive_labels,
            args.allowed_punctuation,
            args.only_first_punctuation_character_after_word_in_autoregressive,
            args.no_label_if_all_characters_are_upper_case,
        )
    write_dataset(
        result[test_size + args.dev_size :],
        args.output_dir / Path("train"),
        args.create_model_input,
        args.bert_labels,
        args.autoregressive_labels,
        args.allowed_punctuation,
        args.only_first_punctuation_character_after_word_in_autoregressive,
        args.no_label_if_all_characters_are_upper_case,
    )
    if args.autoregressive_labels:
        with (args.output_dir / Path("autoregressive_labels_vocab.txt")).open('w') as f:
            for c in set(''.join(result)):
                f.write(c + '\n')


if __name__ == "__main__":
    main()
