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
import numpy as np
import requests
from bs4 import BeautifulSoup, NavigableString

import nltk
from nemo.collections.nlp.modules import get_tokenizer

import prepare_small_data_for_punctuation_capitalization_task as small


logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')


random.seed(42)


SUPPORTED_CORPUS_TYPES = ["wikipedia"]

PAGE_OPENING_NORMAL_TAG = re.compile(r'^ *<page>$')
PAGE_CLOSING_NORMAL_TAG = re.compile(r'^ *</page>$')
TITLE_OF_PAGE = re.compile(r'<title>(.+)</title>')
TEXT_OF_PAGE = re.compile(r'<text>(.+)</text>', flags=re.DOTALL)
QUOTES = re.compile('"\'')
REDIRECT = re.compile('^\s*#REDIRECT +\[\[[^]]*]]')
DOUBLE_BRACES_WITH_CONTENT = re.compile('{{[^}{]*}}')
TABLE = re.compile('{|')
DOUBLE_EQUALS_SIGN_HEADERS = re.compile('\n\\s*==[^\n]+==\\s*\n')
FILE_DESCRIPTION = re.compile(r'\[\[File:\w[^]]*]]')
DOUBLE_SQUARE_BRACKETS_WITH_CONTENT = re.compile(r'\[\[([^]]*)]]')
TRIPLE_QUOTES = re.compile(r"'''([^']+)'''")
END_SECTION = re.compile(r"==\s*(?:See also|Reference|Notes|Sources|Primary sources|Secondary sources)\s*==")

MAX_NUM_CHARACTERS_IN_1_FILE = 10**9


def remove_tables(text):
    result = ""
    tables_in_progress = 0
    for i in range(len(text)):
        to_inspect = text[i : i + 2]
        if tables_in_progress == 0:
            result += text[i]
        if to_inspect == '{|':
            tables_in_progress += 1
        if to_inspect == '|}':
            tables_in_progress -= 1
    return result


def double_square_brackets_replacement(match):
    text = match.group(1)
    text = text.split('|')
    if len(text) == 1:
        return text[0]
    elif len(text) == 2:
        return text[1]
    else:
        logging.warning(f"Found double square brackets with three sections {repr(match.group(0))}")
        return text[1]


def get_wiki_text_lines(text, normalize_process, tokenizer):
    text = REDIRECT.sub('', text)
    while DOUBLE_BRACES_WITH_CONTENT.search(text) is not None:
        text = DOUBLE_BRACES_WITH_CONTENT.sub('', text)
    text = text.strip()
    if not text:
        return []
    end_section = END_SECTION.search(text)
    text = text[:end_section.span()[0]].strip()
    text = DOUBLE_EQUALS_SIGN_HEADERS.sub('\n', text)
    text = FILE_DESCRIPTION.sub('', text)
    text = remove_tables(text)
    text = TRIPLE_QUOTES.sub(r'\1', text)
    text = DOUBLE_SQUARE_BRACKETS_WITH_CONTENT.sub(double_square_brackets_replacement, text)
    outs, errs = normalize_process.communicate(text.encode('utf-8'))
    text = outs.decode('utf-8')
    if tokenizer is not None:
        text = small.remove_untokenizable_characters(text, tokenizer)
    return [sent.strip() for sent in nltk.sent_tokenize(text) if sent.strip()]


def start_normalize_process(lang):
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)
    normalize_process = subprocess.Popen(
        ["./normalize-punctuation.perl", "-l", lang], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    os.chdir(cwd)
    return normalize_process


def preprocess_wikipedia(file_path, output_dir, lang, tokenizer, sequence_length_range):
    normalize_process = start_normalize_process(lang)
    sentences_by_number_of_words = {n: [] for n in range(sequence_length_range[0], sequence_length_range[1])}
    page = ""
    page_i = 0
    output_dir = output_dir / Path(file_path.name)
    page_in_progress = False
    total_number_of_characters_in_current_file = 0
    file_i = 0
    doc_id = 0
    current_file_path = output_dir / Path(str(file_i) + '.xml')
    out_f = current_file_path.open('w')
    with file_path.open() as in_f:
        for i, line in enumerate(in_f):
            if '<page' in line:
                if PAGE_OPENING_NORMAL_TAG.match(line) is None:
                    logging.warning(f'Encountered an unusual page opening tag in line {i} {repr(line)}')
                page_in_progress = True
                start_line = i
            if page_in_progress:
                page += line
            if '</page' in line:
                if PAGE_CLOSING_NORMAL_TAG.match(line) is None:
                    logging.warning(f'Encountered an unusual page opening tag in line {i} {repr(line)}')
                if not page_in_progress:
                    logging.warning(
                        f'Encountered closing page tag without opening tag. Line number: {i}. Line {repr(line)}'
                    )
                elif not page:
                    logging.warning(f"Encountered a page which takes only one line. Line: {i}. Line {repr(line)}")
                end_line = i
                title = TITLE_OF_PAGE.search(page)
                if title is None:
                    logging.warning(f"Title of page {page_i} from line {start_line} to {end_line} is not found.")
                    title = None
                else:
                    title = title.group(1)
                text = TEXT_OF_PAGE.search(page)
                if text is None:
                    logging.warning(
                        f"Text tag is not found on a page {page_i} from line {start_line} to {end_line} is not found. "
                        f"Skipping page.."
                    )
                else:
                    text = get_wiki_text_lines(text.group(1), normalize_process, tokenizer)
                    if text:
                        opening_doc_tag = f'<doc docid="{doc_id}"'
                        if title is not None:
                            opening_doc_tag += f'title="{QUOTES.sub("", title)}">'
                        else:
                            opening_doc_tag += '>'
                        file_text = '\n'.join([opening_doc_tag] + text + ['</doc>']) + '\n'
                        out_f.write(file_text)
                        for k, v in small.arrange_sentences_by_number_of_words_in_1_doc(
                                text, sequence_length_range, [file_i, doc_id]
                        ):
                            sentences_by_number_of_words[k] += v
                        doc_id += 1
                        total_number_of_characters_in_current_file += len(file_text)
                        if total_number_of_characters_in_current_file > MAX_NUM_CHARACTERS_IN_1_FILE:
                            out_f.close()
                            logging.info(f"Finished filling file {current_file_path}")
                            file_i += 1
                            current_file_path = output_dir / Path(str(file_i) + '.xml')
                            logging.info(f"Filling file {current_file_path}")
                            out_f = current_file_path.open('w')
                            total_number_of_characters_in_current_file = 0
                page = ""
                page_i += 1
                page_in_progress = False
    if total_number_of_characters_in_current_file:
        out_f.close()
        logging.info(f"Finished filling file {current_file_path}")
    return sentences_by_number_of_words


def main():
    args = small.get_args(SUPPORTED_CORPUS_TYPES)
    tokenizer = get_tokenizer(args.tokenizer)
    all_docs = {}
    number_of_sentences_in_input = 0
    sentences_by_number_of_words = {n: [] for n in range(args.sequence_length_range[0], args.sequence_length_range[1])}
    for corpus_type, file_path in zip(args.corpus_types, args.input_files):
        logging.info(f"Processing file {file_path}..")
        if corpus_type == SUPPORTED_CORPUS_TYPES[0]:
            for k, v in preprocess_wikipedia(
                file_path, args.output_dir, args.lang, tokenizer, args.sequence_length_range
            ):
                sentences_by_number_of_words[k] += v
        else:
            raise ValueError(
                f"Unsupported corpus type '{corpus_type}. Supported corpus types are {SUPPORTED_CORPUS_TYPES}"
            )
    if args.size is None:
        args.size = number_of_sentences_in_input
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
