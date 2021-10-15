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
from tqdm import tqdm

from nemo.collections.nlp.modules import get_tokenizer


logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')


random.seed(42)


EUROPARL_LINE = re.compile(r"^(.+)(ep(?:-[0-9]{2}){3}(?:-[0-9]{3})?)")
NEWS_COMMENTARY_LOCATION_LINE = re.compile(r"^[A-Z0-9 ]+ – ")
# For counting number of words in a sentence
WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION = re.compile(r"\W*\b\w+(?:[-.]\w+)*(?:'\w+)?\b\W*")
# For splitting text into words and punctuation
WORD = re.compile(r"(\w+'\w+|\w+(?:[./]\w+)*|\b-?\d+(?:\.\d+)*)")
NOT_WORD_CHARACTERS = re.compile(r"[^\w%/$@#°]")
WORD_CHARACTER = re.compile(r"\w")
SPACE_DUP = re.compile(r" {2,}")
STRIP_START = re.compile(r"^\W+")
STRIP_END = re.compile(r"\W+$")
SUPPORTED_CORPUS_TYPES = ["europarl", "news-commentary", "TED", "rapid"]
SENTENCE_ENDINGS = ".?!\";:,)"
SUPPORTED_BERT_PUNCTUATION = set("!,.:;?")
NUM_LENGTH_REMOVED_EXAMPLES = 3
MIN_NUM_CHARACTERS_IN_SENTENCE_FOR_DIGIT_ALPHA_QUOTIENT_APPLICATION = 64
MAX_DIGIT_ALPHA_QUOTIENT = 0.5
MIN_NUM_CHARACTERS_IN_SENTENCE_FOR_UPPERCASE_LIMIT = 32
MAX_UPPER_CASE_QUOTIENT = 1.0
MAX_UPPERCASE_START_LEN = 4
DIGITS = set(string.digits)
ASCII_LETTERS = set(string.ascii_letters)
MORE_THAN_3_DOTS = re.compile(r'\.{4,}')
MORE_THAN_3_SPACE_DOTS = re.compile(r'( \.){3,}')
MORE_THAN_15_DOTS = re.compile(r'\.{15,}')
DOT_SPACE_MULTIDOT = re.compile(r'\. \.{3}$')
MORE_THAN_10_HYPHENS = re.compile(r'-{10,}')
DOT_DIGIT_5 = re.compile(r'(\.\d+){4}')
ABBR_STRING = re.compile(r'^[A-Z][a-z]?\.$')
REF = re.compile(r'^\(\d+(?:\.\d+)?\)$')
SE = re.compile(r'^S\.E\.')
DIGIT = re.compile(r'\d')
ROMAN_NUMERAL = re.compile(
    r'(M{1,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?'
    r'|(CM|CD|DC{0,3}|C{1,3})(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?'
    r'|(XC|XL|LX{0,3}|X{1,3})(IX|IV|V?I{0,3})?'
    r'|(IX|IV|VI{0,3}|I{1,3}))',
    flags=re.I,
)
HTTP = re.compile(r'https?://', flags=re.I)
BRACKETS_AND_CONTENT = re.compile(r'\(.*\)')
SPACING_CHARACTERS_TO_REPLACE = re.compile(
    '[\t\v\r' + r'\u00a0\u1680\u1803\u202f\u205f\u3000\ufeff' + ''.join([chr(i) for i in range(0x2000, 0x200c)]) + ']+'
)


def get_args(
    supported_corpus_types, add_nltk_tokenization_parameter=False, add_resume_argument=False, add_num_jobs=False
):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument(
        "--input_files",
        help="List of files with input data. You should also provide `--corpus_types` list which elements are types "
        "corresponding files.",
        nargs="+",
        type=Path,
        required=not add_resume_argument,
    )
    parser.add_argument(
        "--input_language",
        "-L",
        help="Used for punctuation normalization. en - English, de - German, cz - Czech, fr - French. "
        "Other options (List of supported languages https://fasttext.cc/docs/en/language-identification.html) are also "
        "possible but there is no special instructions for punctuation normalization. "
        "See https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/normalize-punctuation.perl",
        default="en",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to the output dir with dev.txt, train.txt, and test.txt files.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--corpus_types",
        "-c",
        help="List of names of WMT corpuses which is used as raw material for creating punctuation capitalization "
        "dataset. Number and order of elements in this list should be equal to the number of elements in `input_files` "
        "list.",
        choices=supported_corpus_types,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--size",
        "-S",
        help="Number of sequences in the created dataset. This number includes sequences in train, dev, and test "
        "datasets. By default it is equal to the total number of sentences in the input data.",
    )
    parser.add_argument("--dev_size", "-d", help="Number of sequences in dev data.", type=int, default=10 ** 4)
    parser.add_argument("--test_size", "-t", help="Percentage of test data.", type=int, default=10 ** 4)
    parser.add_argument(
        "--sequence_length_range",
        "-r",
        help="Minimum and maximum number words in model input sequences. Number of words is sampled "
        "using uniform distribution.",
        type=int,
        nargs=2,
        default=[2, 64],
    )
    parser.add_argument(
        "--percentage_segments_with_intact_sentences",
        "-w",
        help="For any number of words in a segment percentage of segments with whole sentences can not be lower than "
        "`--percentage_segments_with_intact_sentences`. If this condition can not be satisfied together with the "
        "dataset size and uniform distribution of segment lengths in dataset, distribution will not be uniform: "
        "Probably number of long and short segments will be less than number of segments with average length.",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--clean_data_dir",
        "-C",
        help="Path to directory where cleaned input files are saved. If not provided cleaned input files are "
        "not saved.",
        type=Path,
    )
    parser.add_argument(
        "--create_model_input",
        "-i",
        help="Whether to write text without punctuation to output directory",
        action="store_true",
    )
    parser.add_argument("--bert_labels", "-b", help="Whether create BERT labels.", action="store_true")
    parser.add_argument(
        "--autoregressive_labels", "-a", help="Whether create autoregressive labels", action="store_true"
    )
    parser.add_argument(
        "--allowed_punctuation",
        "-p",
        help=f"A string containing punctuation marks on which training is performed. Example: '.,?'. "
        f"Do not include single quote and space into it. If single quotes are included they will be ignored. "
        f"BERT labels can include only {SUPPORTED_BERT_PUNCTUATION} punctuation characters.",
        type=set,
        default=set('"!(),-.:;?'),
    )
    parser.add_argument(
        "--fasttext_model",
        "-f",
        help="Path to fastText model used for language verification. The description and download links are here "
        "https://fasttext.cc/docs/en/language-identification.html. If path to the model is not provided, then "
        "lid.176.bin is downloaded and saved in the same directory with this script.",
        type=Path,
    )
    parser.add_argument(
        "--max_fraction_of_wrong_language",
        "-R",
        help="Max fraction of characters in a document which are identified by fastText as belonging to wrong "
        "language. If the fraction is greater than `--max_fraction_of_wrong_language` a document is removed.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max_length_of_sentence_of_wrong_language",
        "-s",
        help="Max number of characters in a sentence identified by fastText model as written in wrong "
        "language. A document with such a sentence is removed.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--tokenizer",
        "-z",
        help="Tokenizer used for checking characters for tokenizability.",
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--only_first_punctuation_character_after_word_in_autoregressive",
        "-F",
        help="Add only first punctuation character after word to autoregressive labels.",
        action="store_true",
    )
    parser.add_argument(
        "--no_label_if_all_characters_are_upper_case",
        "-U",
        help="If this option is set all words capitalization are labelled as 'U' if the first character is in upper "
        "case. If this option is not set words which contain only uppercase letters (except one character words) "
        "are marked as 'U' and words which first character is in upper case but containing not lower case characters "
        "are marked as 'u'.",
        action="store_true",
    )
    if add_nltk_tokenization_parameter:
        parser.add_argument(
            "--nltk_tokenization",
            "-n",
            help="Tokenize lines into sentences using NLTK tokenization.",
            action="store_true",
        )
    if add_resume_argument:
        parser.add_argument(
            "--resume_from",
            choices=["cutting", "shuffling", "writing"],
            help="From which stage big dataset preparation is started."
        )
    if add_num_jobs:
        parser.add_argument("--num_jobs", default=1, type=int)
    args = parser.parse_args()
    if args.size is not None:
        if args.dev_size > args.size:
            raise ValueError(f"Parameter `--dev_size={args.dev_size}` is less than size of all dataset ({args.size})")
    args.input_files = [x.expanduser() for x in args.input_files]
    if len(args.input_files) != len(args.corpus_types):
        raise ValueError(
            f"Number {len(args.input_files)} of input files {args.input_files} is not equal to the number "
            f"{len(args.corpus_types)} of corpus types {args.corpus_types}."
        )
    args.output_dir = args.output_dir.expanduser()
    args.clean_data_dir = args.clean_data_dir.expanduser()
    if args.allowed_punctuation - SUPPORTED_BERT_PUNCTUATION:
        logging.warning(
            f"Punctuation marks {args.allowed_punctuation - SUPPORTED_BERT_PUNCTUATION} are not allowed for BERT "
            f"labels."
        )
    if args.fasttext_model is None:
        save_path = Path(__file__).parent / Path("lid.176.bin")
        if save_path.exists():
            logging.info(f"Found fastText model {save_path}. Loading fastText model...")
            try:
                fasttext.load_model(str(save_path))
            except ValueError:
                logging.exception(
                    f"Encountered ValueError when loading fastText model. Pass another model file or remove "
                    f"{save_path} and script download not corrupted model file.")
                raise
            logging.info(f"Loaded successfully.")
        else:
            logging.info(
                "Downloading fastText model lid.176.bin for language identification from "
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin..."
            )
            r = requests.get(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", allow_redirects=True
            )
            with save_path.open('wb') as f:
                f.write(r.content)
            logging.info(f"fastText model is saved to {save_path}")
        args.fasttext_model = save_path
    return args


def too_many_digits(text):
    num_alpha_chars = 0
    num_digits = 0
    if len(text) > MIN_NUM_CHARACTERS_IN_SENTENCE_FOR_DIGIT_ALPHA_QUOTIENT_APPLICATION:
        for c in text:
            if c in ASCII_LETTERS:
                num_alpha_chars += 1
            elif c in DIGITS:
                num_digits += 1
        too_many = num_digits / (num_alpha_chars + 1e-12) > MAX_DIGIT_ALPHA_QUOTIENT
    else:
        too_many = False
    return too_many


def too_many_uppercase(text):
    num_uppercase = 0
    num_lowercase = 0
    if len(text) > MIN_NUM_CHARACTERS_IN_SENTENCE_FOR_UPPERCASE_LIMIT:
        for c in text:
            if c.isupper():
                num_uppercase += 1
            if c.islower():
                num_lowercase += 1
        too_many = num_uppercase / (num_lowercase + 1e-12) > MAX_UPPER_CASE_QUOTIENT
    else:
        too_many = False
    return too_many


def remove_untokenizable_characters_from_text(
    text, tokenizer, tok_chars=None, untok_chars=None, remove_all_lines=False
):
    tok_chars = {' ', '\n'} if tok_chars is None else tok_chars.copy()
    untok_chars = set() if untok_chars is None else untok_chars.copy()
    all_chars = set(text)
    detected_untok_chars = list(all_chars & untok_chars)
    candidates_for_untok_chars = all_chars - tok_chars - untok_chars
    if not detected_untok_chars and not candidates_for_untok_chars:
        return text, tok_chars, untok_chars
    for c in candidates_for_untok_chars:
        if tokenizer.text_to_ids(c):
            tok_chars.add(c)
        else:
            untok_chars.add(c)
            detected_untok_chars.append(c)
    if not detected_untok_chars:
        return text, tok_chars, untok_chars
    if '\\' in detected_untok_chars:
        detected_untok_chars.remove('\\')
        detected_untok_chars = ['\\\\'] + detected_untok_chars
    if '^' == detected_untok_chars[0]:
        if len(detected_untok_chars) > 1:
            detected_untok_chars.remove('^')
            detected_untok_chars.append('^')
        else:
            detected_untok_chars = [r'\^']
    if '-' in detected_untok_chars:
        detected_untok_chars.remove('-')
        detected_untok_chars.append('-')
    uc = re.compile('[' + ''.join(detected_untok_chars) + ']', re.I)
    if remove_all_lines:
        result = ""
        i = 0
        for m in uc.finditer(text):
            right = text.rfind('\n', i, m.span()[0])
            if right > 0:
                result += text[i : right]
            cand = text.find('\n', m.span()[1])
            i = cand if cand > 0 else len(text)
        result += text[i:]
    else:
        result = uc.sub('', uc.sub('\n' + uc.pattern + '\n', '\n', text))
    return result, tok_chars, untok_chars


def remove_untokenizable_characters(docs, tokenizer):
    names = list(docs.keys())
    doc_texts = remove_untokenizable_characters_from_text(
        '\n\n\n'.join(['\n'.join(doc) for doc in docs.values()]), tokenizer
    ).split('\n\n\n')
    assert len(doc_texts) == len(names)
    for i, name in enumerate(names):
        docs[name] = doc_texts[i].split('\n')
    return docs


def preprocess_europarl(text):
    text = SPACING_CHARACTERS_TO_REPLACE.sub(' ', text)
    f = StringIO(text)
    docs = {}
    for i, line in enumerate(f):
        m = EUROPARL_LINE.match(line)
        if m is None:
            raise ValueError(f"Could not match {i} EUROPARL line {repr(line)}")
        text = m.group(1).strip()
        if (
            text
            and not too_many_digits(text)
            and text[-1] in SENTENCE_ENDINGS
            and WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.search(text) is not None
        ):
            doc = "europarl_" + m.group(2).strip()
            if doc not in docs:
                docs[doc] = [text]
            else:
                docs[doc].append(text)
    return docs


def preprocess_ted(text):
    text = SPACING_CHARACTERS_TO_REPLACE.sub(' ', text)
    soup = BeautifulSoup(text)
    result = {}
    for doc in soup.findAll("doc"):
        doc_id = doc["docid"]
        title = doc.find("title").text
        key = "TED_" + doc_id + "._" + title
        text = ''.join([e for e in doc if isinstance(e, NavigableString)]).strip()
        lines = [
            line.strip() for line in text.split('\n')
            if WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.search(line.strip()) is not None
        ]
        if lines:
            result[key] = lines
        else:
            logging.warning(f"Found empty document {doc_id} in TED dataset")
    return result


def contain_number(word):
    return DIGIT.search(word) is not None or ROMAN_NUMERAL.search(word) is not None


def all_words_contain_numbers(line):
    line = BRACKETS_AND_CONTENT.sub('', line).strip()
    words = WORD.findall(line)
    return not words or all([contain_number(w) for w in words])


def check_rapid_line(line):
    return (
        len(line) > 0
        and line[-1] in SENTENCE_ENDINGS
        and MORE_THAN_15_DOTS.search(line) is None
        and MORE_THAN_3_SPACE_DOTS.search(line) is None
        and MORE_THAN_10_HYPHENS.search(line) is None
        and DOT_DIGIT_5.search(line) is None
        and not too_many_digits(line)
        and line not in {'p.m.', 'Prov.', 'n.a.', 'Prof.'}
        and not (line.isupper() and len(line) > 10)
        and not (ABBR_STRING.match(line) is not None and line != "No.")
        and 'n.a. n.a.' not in line
        and not too_many_uppercase(line)
        and REF.match(line) is None
        and SE.match(line) is None
        and not line[: MAX_UPPERCASE_START_LEN + 1].isupper()
        and not all_words_contain_numbers(line)
        and 'HYPERLINK' not in line
        and HTTP.search(line) is None
        and WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.search(line) is not None
    )


def preprocess_rapid(text, verbose=False):
    text = SPACING_CHARACTERS_TO_REPLACE.sub(' ', text)
    soup = BeautifulSoup(text)
    result = {}
    for file in soup.findAll("file"):
        file_id = file["id"]
        file_utterances = []
        for unit in file.findAll("unit"):
            unit_id = unit["id"]
            segment = unit.find("segment")
            source = segment.find("source")
            target = segment.find("target")
            if source['xml:lang'] == "en":
                text = source.text
            elif target["xml:lang"] == "en":
                text = target.text
            else:
                raise ValueError(
                    f"No utterance in English was found in file {file_id} in unit {unit_id}. "
                    f"Source language: {source['lang']}. Target language: {target['lang']}"
                )
            if check_rapid_line(text):
                file_utterances.append(SPACE_DUP.sub(' ', text.replace(chr(61623), ' ')).strip())
        if file_utterances:
            result["rapid_file_" + file_id] = file_utterances
        elif verbose:
            logging.warning(f"Found empty RAPID document {file_id}")
    return result


def check_news_commentary_line(line):
    return (
        len(line) > 0
        and MORE_THAN_10_HYPHENS.search(line) is None
        and not too_many_digits(line)
        and not (line.isupper() and len(line) > 10)
        and not too_many_uppercase(line)
        and line[-1] in SENTENCE_ENDINGS
        and WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.search(line) is not None
    )


def preprocess_news_commentary(text):
    result = {}
    discussion_text = []
    discussion_count = 0
    line_idx = 0
    text = SPACING_CHARACTERS_TO_REPLACE.sub(' ', text)
    for line_i, line in enumerate(StringIO(text)):
        line = line.strip()
        if line:
            if line_idx == 1:
                location_string = NEWS_COMMENTARY_LOCATION_LINE.match(line)
                if location_string is not None:
                    line = line[location_string.span()[1] :]
                line = line.strip()
                if line and MORE_THAN_10_HYPHENS.search(line) is None:
                    discussion_text.append(line)
            elif line_idx > 1 and check_news_commentary_line(line):
                discussion_text.append(line)
            line_idx += 1
        else:
            if discussion_text:
                result[f"news-commentary_discussion{discussion_count}"] = discussion_text
            else:
                logging.warning(f"Found empty news-commentary discussion starting at line {line_i}")
            discussion_text = []
            discussion_count += 1
            line_idx = 0
    if discussion_text:
        result[f"news-commentary_discussion{discussion_count}"] = discussion_text
    else:
        logging.warning(f"Found empty news-commentary discussion starting at line {line_i}")
    return result


def cut_words(s, start_word, num_words):
    words = WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.findall(s)
    s = ''.join(words[start_word : start_word + num_words])
    ss = STRIP_START.match(s)
    if ss is not None:
        s = s[ss.span()[1] :]
    se = STRIP_END.match(s)
    if se is not None:
        s = s[: se.span()[0]]
    return s


def cut_words_from_all_docs(all_docs, borders):
    result = []
    for b in borders:
        doc = all_docs[b[0]]
        text = ' '.join(doc[b[1] : b[2]])
        result.append(cut_words(text, b[3], b[4]))
    return result


def add_docs(all_docs, file_docs, file_name):
    for k, v in file_docs.items():
        duplicate = False
        if k in all_docs:
            if v == all_docs[k]:
                duplicate = True
                logging.warning(f"Duplicate document with name {k} in file {file_name}")
            i = 2
            while k + "_" + str(i) in all_docs:
                if v == all_docs[k + "_" + str(i)]:
                    duplicate = True
                    logging.warning(
                        f"Duplicate documents with names {k} and {k + '_' + str(i)}. One of documents is "
                        f"from file {file_name}"
                    )
                i += 1
            k += "_" + str(i)
        if not duplicate:
            all_docs[k] = v


def get_lang(line, fasttext_model):
    labels, _ = fasttext_model.predict(line, k=1)
    lang = labels[0].split('__')[-1]
    return lang


def create_doc_example_string(docs):
    s = ""
    for k, v in docs.items():
        s += k + '\n'
        s += '\n'.join(v) + '\n' * 3
    return s[:-3]


def remove_wrong_docs(docs):
    keys = list(docs.keys())
    for name in keys:
        if any(['.....' in line for line in docs[name]]):
            del docs[name]


def remove_wrong_lang_docs(docs, lang, model_path, max_fraction, max_length):
    model = fasttext.load_model(str(model_path))
    num_fraction_removed = 0
    num_length_removed = 0
    examples_of_fraction_removed = {}
    examples_of_length_removed = {}
    keys = list(docs.keys())
    random.shuffle(keys)
    for name in keys:
        total_num_characters = 0
        wrong_num_characters = 0
        bad = False
        for line in docs[name]:
            if lang != get_lang(line, model):
                if len(line) > max_length:
                    bad = True
                    if num_length_removed < NUM_LENGTH_REMOVED_EXAMPLES:
                        examples_of_length_removed[name] = docs[name]
                    num_length_removed += 1
                    break
                wrong_num_characters += len(line)
            total_num_characters += len(line)
        if not bad and wrong_num_characters / total_num_characters > max_fraction:
            if num_fraction_removed < NUM_LENGTH_REMOVED_EXAMPLES:
                examples_of_fraction_removed[name] = docs[name]
            num_fraction_removed += 1
            bad = True
        if bad:
            del docs[name]
    logging.info(f"Number of documents removed because of too long sentences of wrong language: {num_length_removed}")
    logging.info(f"Number of documents removed because of too big fraction of wrong language: {num_fraction_removed}")
    logging.info(f"Original number of documents: {len(keys)}")
    logging.info(
        f"Examples of removed documents because of too long sentences of wrong language:\n"
        f"{create_doc_example_string(examples_of_length_removed)}"
    )
    logging.info(
        f"Examples of removed documents because of too big fraction of wrong language:\n"
        f"{create_doc_example_string(examples_of_fraction_removed)}"
    )


def arrange_sentences_by_number_of_words_in_1_doc(doc, sequence_length_range, values_to_prepend):
    if not isinstance(doc, (list, tuple)):
        raise ValueError(f"Parameter `doc` has to `list` or `tuple`, give {type(doc)}.")
    result = {n: [] for n in range(sequence_length_range[0], sequence_length_range[1])}
    lengths = [len(WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.findall(sent)) for sent in doc]
    for start_sentence_i, sentence in enumerate(doc):
        for end_sentence_i in range(start_sentence_i + 1, len(doc)):
            n_words = sum(lengths[start_sentence_i : end_sentence_i])
            if n_words >= sequence_length_range[1] or n_words < sequence_length_range[0]:
                break
            result[n_words].append(values_to_prepend + [start_sentence_i, end_sentence_i])
    return result, lengths


def arrange_sentences_by_number_of_words(docs, sequence_length_range):
    result = {n: [] for n in range(sequence_length_range[0], sequence_length_range[1])}
    for doc_id, doc in docs.items():
        doc_arrangement = arrange_sentences_by_number_of_words_in_1_doc(doc, sequence_length_range, [doc_id])[0]
        for k, v in doc_arrangement.items():
            result[k] += v
    return result


def select_close_to_uniform_distribution(
    sentences_by_number_of_words,
    planned_number_of_segments,
    percentage_of_segments_with_intact_sentences,
    number_of_sentences_by_docs,
    doc_id_field_i=0,
):
    result = []
    remaining_by_docs = {doc_id: set(range(num)) for doc_id, num in number_of_sentences_by_docs.items()}
    number_of_sentences_by_number_of_words = sorted([(len(v), k) for k, v in sentences_by_number_of_words.items()])
    number_of_words_stats = []
    min_number_of_sentences_for_sentence_len = ceil(
        planned_number_of_segments
        / len(sentences_by_number_of_words)
        / 100
        * percentage_of_segments_with_intact_sentences
    )
    for i, (n, len_) in tqdm(
        enumerate(number_of_sentences_by_number_of_words), total=len(number_of_sentences_by_number_of_words)
    ):
        if n < min_number_of_sentences_for_sentence_len:
            tmp = sentences_by_number_of_words[len_]
            # Readjust planned number of segments for remaining sentences lengths taking into account that for current
            # length there will be less sentences
            planned_number_of_segments -= len(tmp) * int(100 / percentage_of_segments_with_intact_sentences)
            min_number_of_sentences_for_sentence_len = ceil(
                planned_number_of_segments
                / (len(sentences_by_number_of_words) - i)
                / 100
                * percentage_of_segments_with_intact_sentences
            )
        else:
            tmp = random.sample(sentences_by_number_of_words[len_], min_number_of_sentences_for_sentence_len)
        result += tmp
        number_of_words_stats.append((len_, len(tmp)))
        for res in tmp:
            start_i, end_i = res[-2:]
            doc_id = res[doc_id_field_i]
            if doc_id not in remaining_by_docs:
                remaining_by_docs[doc_id] = set()
            remaining_by_docs[doc_id].difference_update(range(start_i, end_i))
    return result, dict(sorted(number_of_words_stats)), remaining_by_docs


def calculate_how_many_remain_to_cut(number_of_words_stats, size, percentage_segments_with_intact_sentences):
    if percentage_segments_with_intact_sentences > 0:
        factor = (100 - percentage_segments_with_intact_sentences) / percentage_segments_with_intact_sentences
        result = {k: ceil(v * factor) for k, v in number_of_words_stats.items()}
    else:
        n = ceil(size / len(number_of_words_stats))
        result = {k: n for k in number_of_words_stats}
    keys = sorted(result.keys(), key=lambda x: -number_of_words_stats[x])
    total = sum(number_of_words_stats.values()) + sum(result.values())
    key_i = 0
    while total > size:
        if result[keys[key_i]] > 0:
            total -= 1
            result[keys[key_i]] -= 1
        key_i = (key_i + 1) % len(keys)
    for k in keys:
        if result[k] == 0:
            del result[k]
    return result


def create_not_whole_sentence_segments(
    sentence_len_by_docs, remaining_by_docs, number_of_words_stats, size, percentage_segments_with_intact_sentences
):
    result = []
    remaining_by_docs = deepcopy(remaining_by_docs)
    yet_to_cut_by_number_of_words = calculate_how_many_remain_to_cut(
        number_of_words_stats, size, percentage_segments_with_intact_sentences
    )
    assert (
        size == sum(number_of_words_stats.values()) + sum(yet_to_cut_by_number_of_words.values()),
        f"size={size},"
        f"sum(number_of_words_stats.values())={sum(number_of_words_stats.values())},"
        f"sum(yet_to_cut_by_number_of_words.values())={sum(yet_to_cut_by_number_of_words.values())}",
    )
    nw_i = 0
    done = not bool(yet_to_cut_by_number_of_words)
    while not done:
        for doc_id, remaining in remaining_by_docs.items():
            next_sentence_i = -1
            for i in remaining:
                len_ = sentence_len_by_docs[doc_id][i]
                if i >= next_sentence_i and len_ > 1:
                    shift = random.randint(1, len_ // 2)
                    start_sentence_i = i
                    num_words = len_
                    next_sentence_i = i + 1
                    number_of_words = list(yet_to_cut_by_number_of_words.keys())
                    nw_i %= len(number_of_words)
                    while (
                        shift + number_of_words[nw_i] + 1 > num_words
                        and next_sentence_i < len(sentence_len_by_docs[doc_id])
                    ):
                        num_words += sentence_len_by_docs[doc_id][next_sentence_i]
                        next_sentence_i += 1
                    if shift + number_of_words[nw_i] < num_words:
                        result.append(
                            [
                                doc_id,
                                start_sentence_i,
                                next_sentence_i,
                                shift,
                                number_of_words[nw_i],
                            ]
                        )
                        yet_to_cut_by_number_of_words[number_of_words[nw_i]] -= 1
                        if yet_to_cut_by_number_of_words[number_of_words[nw_i]] == 0:
                            del yet_to_cut_by_number_of_words[number_of_words[nw_i]]
                            if not yet_to_cut_by_number_of_words:
                                done = True
                                break
                        nw_i += 1
                    else:
                        break
            if done:
                break
        remaining_by_docs = {doc_id: set(range(len(doc))) for doc_id, doc in sentence_len_by_docs.items()}
    assert (
        len(result) == size - sum(number_of_words_stats.values()),
        f"len(result)={len(result)}, size={size}, "
        f"sum(number_of_words_stats.values())={sum(number_of_words_stats.values())}",
    )
    return result


def create_dataset_string(file_docs):
    result = ""
    for doc_id, doc in file_docs.items():
        result += '\n'.join([f'<doc docid="{doc_id}">'] + doc + ['</doc>']) + '\n'
    return result


def normalize_punctuation(all_docs, lang):
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)
    normalize_process = subprocess.Popen(
        ["./normalize-punctuation.perl", "-l", lang], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    outs, errs = normalize_process.communicate(
        '\n\n\n'.join(['\n'.join(v) for v in all_docs.values()]).encode('utf-8')
    )
    counter = 0
    for k, text in zip(
        all_docs,
        DOT_SPACE_MULTIDOT.sub('.', MORE_THAN_3_DOTS.sub('...', outs.decode('utf-8'))).split('\n\n\n')
    ):
        counter += 1
        lines = [line.strip() for line in text.split('\n')]
        assert len(lines) == len(all_docs[k]), f"len(lines)={len(lines)}, len(all_docs[k])={len(all_docs[k])}"
        all_docs[k] = lines
    assert counter == len(all_docs), f"counter={counter}, len(all_docs)={len(all_docs)}"
    normalize_process.kill()
    os.chdir(cwd)


def create_bert_labels(line, allowed_punctuation, no_label_if_all_characters_are_upper_case):
    labels = ""
    allowed_punctuation = allowed_punctuation & SUPPORTED_BERT_PUNCTUATION
    for w_i, word in enumerate(line):
        if WORD.match(word):
            if len(word) > 0 and word.isupper():
                labels += 'U'
            elif word[0].isupper():
                labels += 'u'
            else:
                labels += "O"
            labels += line[w_i + 1][0] if w_i < len(line) - 1 and line[w_i + 1][0] in allowed_punctuation else 'O'
            labels += ' '
    if no_label_if_all_characters_are_upper_case:
        labels = labels.replace('u', 'U')
    return labels


def create_autoregressive_labels(
    line,
    allowed_punctuation,
    only_first_punctuation_character_after_word,
    no_label_if_all_characters_are_upper_case,
):
    labels = ""
    for w_i, word in enumerate(line):
        if WORD.match(word):
            if len(word) > 1 and word.isupper():
                labels += 'U'
            elif word[0].isupper():
                labels += 'u'
            else:
                labels += 'O'
        else:
            if only_first_punctuation_character_after_word:
                num_added = 0
                for c_i, c in enumerate(word):
                    if (
                        c in allowed_punctuation | {' '}
                        and (
                            num_added == 0
                            or num_added == 1
                            and labels
                            and labels[-1] != ' '
                            and c == ' '
                        )
                    ):
                        num_added += 1
                        labels += c
                if num_added == 0:
                    labels += ' '
            else:
                num_added = 0
                for c in word:
                    if c in allowed_punctuation | {' '}:
                        labels += c
                        num_added += 1
                if num_added == 0:
                    labels += ' '
    labels = SPACE_DUP.sub(' ', labels)
    if no_label_if_all_characters_are_upper_case:
        labels = labels.replace('u', 'U')
    return labels


def write_dataset(
    data,
    dir_,
    create_model_input,
    bert_labels,
    autoregressive_labels,
    allowed_punctuation,
    only_first_punctuation_character_after_word_in_autoregressive,
    no_label_if_all_characters_are_upper_case,
):
    dir_.mkdir(exist_ok=True, parents=True)
    with (dir_ / Path("text.txt")).open('w') as f:
        for line in data:
            f.write(line + '\n')
    split_data = [[s for s in WORD.split(line) if s] for line in data]
    if create_model_input:
        with (dir_ / Path("input.txt")).open('w') as f:
            for line in split_data:
                f.write(' '.join([s.lower() for s in line if WORD.match(s)]) + '\n')
    if bert_labels:
        with (dir_ / Path("bert_labels.txt")).open('w') as f:
            for line in split_data:
                f.write(
                    create_bert_labels(line, allowed_punctuation, no_label_if_all_characters_are_upper_case) + '\n'
                )
    if autoregressive_labels:
        with (dir_ / Path("autoregressive_labels.txt")).open('w') as f:
            for line in split_data:
                f.write(
                    create_autoregressive_labels(
                        line,
                        allowed_punctuation,
                        only_first_punctuation_character_after_word_in_autoregressive,
                        no_label_if_all_characters_are_upper_case,
                    ) + '\n'
                )


def main():
    args = get_args(SUPPORTED_CORPUS_TYPES)
    tokenizer = get_tokenizer(args.tokenizer)
    all_docs = {}
    number_of_sentences_in_input = 0
    for corpus_type, file_path in zip(args.corpus_types, args.input_files):
        logging.info(f"Processing file {file_path}..")
        with file_path.open() as f:
            if corpus_type == SUPPORTED_CORPUS_TYPES[0]:
                file_docs = remove_untokenizable_characters(preprocess_europarl(f.read()), tokenizer)[0]
            elif corpus_type == SUPPORTED_CORPUS_TYPES[1]:
                file_docs = remove_untokenizable_characters(preprocess_news_commentary(f.read()), tokenizer)[0]
            elif corpus_type == SUPPORTED_CORPUS_TYPES[2]:
                file_docs = remove_untokenizable_characters(preprocess_ted(f.read()), tokenizer)[0]
            elif corpus_type == SUPPORTED_CORPUS_TYPES[3]:
                file_docs = remove_untokenizable_characters(preprocess_rapid(f.read()), tokenizer)[0]
            else:
                raise ValueError(
                    f"Unsupported corpus type '{corpus_type}. Supported corpus types are {SUPPORTED_CORPUS_TYPES}"
                )
            if args.clean_data_dir is not None:
                args.clean_data_dir.mkdir(parents=True, exist_ok=True)
                with (args.clean_data_dir / Path(file_path.name)).open('w') as f:
                    f.write(create_dataset_string(file_docs))
            add_docs(all_docs, file_docs, file_path)
            number_of_sentences_in_input += sum([len(doc) for doc in file_docs.values()])
    remove_wrong_lang_docs(
        all_docs,
        args.input_language,
        args.fasttext_model,
        args.max_fraction_of_wrong_language,
        args.max_length_of_sentence_of_wrong_language,
    )
    normalize_punctuation(all_docs, args.input_language)
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
    result, number_of_words_stats, remaining_by_docs = select_close_to_uniform_distribution(
        sentences_by_number_of_words,
        args.size,
        args.percentage_segments_with_intact_sentences,
        {k: len(v) for k, v in all_docs.items()},
    )
    for i in range(len(result)):
        result[i] = ' '.join(all_docs[result[i][0]][result[i][1] : result[i][2]])
    result += cut_words_from_all_docs(
        all_docs,
        create_not_whole_sentence_segments(
            {
                k: [len(WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.findall(x)) for x in v]
                for k, v in all_docs.items()
            },
            remaining_by_docs,
            number_of_words_stats,
            args.size,
            args.percentage_segments_with_intact_sentences,
        )
    )
    result = list(set(result))
    random.shuffle(result)
    if args.dev_size > len(result):
        raise ValueError(f"Parameter `--dev_size={args.dev_size}` is less than size of all dataset ({len(result)})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_size > 0:
        write_dataset(
            result[:args.test_size],
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
            result[args.test_size : args.test_size + args.dev_size],
            args.output_dir / Path("dev"),
            args.create_model_input,
            args.bert_labels,
            args.autoregressive_labels,
            args.allowed_punctuation,
            args.only_first_punctuation_character_after_word_in_autoregressive,
            args.no_label_if_all_characters_are_upper_case,
        )
    write_dataset(
        result[args.test_size + args.dev_size :],
        args.output_dir / Path("train"),
        args.create_model_input,
        args.bert_labels,
        args.autoregressive_labels,
        args.allowed_punctuation,
        args.only_first_punctuation_character_after_word_in_autoregressive,
        args.no_label_if_all_characters_are_upper_case,
    )


if __name__ == "__main__":
    main()
