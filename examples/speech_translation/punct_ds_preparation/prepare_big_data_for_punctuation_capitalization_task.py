import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import pexpect
from tqdm import tqdm

import nltk
from nemo.collections.nlp.modules import get_tokenizer

import prepare_small_data_for_punctuation_capitalization_task as small


logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')


random.seed(42)


SUPPORTED_CORPUS_TYPES = ["wikipedia"]

PAGE_OPENING_NORMAL_TAG = re.compile(r'^ *<page>$')
PAGE_CLOSING_NORMAL_TAG = re.compile(r'^ *</page>$')
TITLE_OF_PAGE = re.compile(r'<title>(.+)</title>')
TEXT_OF_PAGE = re.compile(r'<text[^>]*>(.+)</text>', flags=re.DOTALL)
QUOTES = re.compile('"\'')
REDIRECT = re.compile(r'^\s*#REDIRECT +\[\[[^]]*]]')
DOUBLE_BRACES_WITH_CONTENT = re.compile(r'{{[^}{]*}}|\({{[^}{]*}}\)')
TABLE = re.compile('{|')
EQUALS_SIGN_HEADERS = re.compile('^[ \t]*==+[^\n=]+==+[ \t]*$', flags=re.MULTILINE)
FILE_DESCRIPTION = re.compile(
    r'\[\[File:\w'
    r'(?:'
    r'[^][]*'
    r'(?:'
    r'\[\['
    r'[^][]*'
    r']]'
    r')?'
    r')*'
    r'[^][]*'
    r']]'
)
DOUBLE_SQUARE_BRACKETS_WITH_CONTENT = re.compile(r'\[\[([^][]*)]]')
TRIPLE_QUOTES = re.compile(r"'''([^']+)'''")
END_SECTION = re.compile(
    r"==\s*(?:See also|References|Notes|Sources|Primary sources|Secondary sources|External links)\s*=="
)
NORMALIZE_ENDING_PATTERN = re.compile(b'.*EOFEOFEOF', flags=re.DOTALL)
NEW_LINE_DUP = re.compile('\n{2,}')

MAX_NUM_CHARACTERS_IN_1_FILE = 10**9


def remove_tables(text):
    result = ""
    tables_in_progress = 0
    for i in range(len(text)):
        if text[i : i + 2] == '{|':
            tables_in_progress += 1
        if tables_in_progress == 0:
            result += text[i]
        if text[i - 1 : i + 1] == '|}':
            tables_in_progress -= 1
    return result


def remove_remarks(text):
    result = ""
    remarks_in_progress = 0
    for i in range(len(text)):
        if text[i : i + 4] == '<!--':
            remarks_in_progress += 1
        if remarks_in_progress == 0:
            result += text[i]
        if text[i - 2 : i + 1] == '-->':
            remarks_in_progress -= 1
    return result


def remove_tag_with_content(text, tag, remove_whole_line=False):
    result = ""
    tags_in_progress = 0
    i = 0
    while i < len(text):
        if text[i: i + 2 + len(tag)] == f"<{tag}>":
            if tags_in_progress == 0 and remove_whole_line:
                i = len(result) - 1
                while i > 0 and result[i] != '\n':
                    i -= 1
                result = result[:i]
            tags_in_progress += 1
        if tags_in_progress == 0:
            result += text[i]
        if text[i - 2 - len(tag) : i + 1] == f"</{tag}>":
            tags_in_progress -= 1
            if tags_in_progress == 0 and remove_whole_line:
                while i < len(text) and text[i] != '\n':
                    i += 1
        i += 1
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


def normalize(text, normalize_process):
    pattern = NORMALIZE_ENDING_PATTERN
    ending = pattern.pattern[2:].decode('utf-8')
    updated_ending = False
    while ending in text:
        updated_ending = True
        ending += b"EOF"
    if updated_ending:
        pattern = re.compile(b'.*' + ending.encode('utf-8'), flags=re.DOTALL)
    with open('current_text.txt', 'w') as f:
        f.write(text + ending)
    normalize_process.send((text + ending).encode('utf-8'))
    normalize_process.expect(pattern)
    res = normalize_process.match.group(0).decode('utf-8')
    return res[:len(res) - len(ending)].replace('\r\n', '\n')


def get_wiki_text_lines(text, normalize_process, tokenizer):
    text = REDIRECT.sub('', text)
    while DOUBLE_BRACES_WITH_CONTENT.search(text) is not None:
        text = DOUBLE_BRACES_WITH_CONTENT.sub('', text)
    text = text.strip()
    if not text:
        return []
    end_section = END_SECTION.search(text)
    print("end_section:", end_section)
    if end_section is not None:
        text = text[:end_section.span()[0]].strip()
    text = EQUALS_SIGN_HEADERS.sub('\n', text)
    text = FILE_DESCRIPTION.sub('', text)
    text = remove_tables(text)
    text = TRIPLE_QUOTES.sub(r'\1', text)
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = remove_tag_with_content(text, 'ref')
    text = remove_tag_with_content(text, 'math', remove_whole_line=True)
    text = text.replace('<doc doc_id"', '')
    text = text.replace('</doc>', '')
    text = remove_remarks(text)
    text = text.replace("''", '"')
    text = text.replace("&quot;", '"')
    text = DOUBLE_SQUARE_BRACKETS_WITH_CONTENT.sub(double_square_brackets_replacement, text)
    text = NEW_LINE_DUP.sub('\n', text)
    if text[-1] != '\n':
        text += '\n'
    text = normalize(text, normalize_process)
    if tokenizer is not None:
        text = small.remove_untokenizable_characters_from_text(text, tokenizer)
    return [sent.strip() for sent in nltk.sent_tokenize(text) if sent.strip()]


def start_normalize_process(lang):
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)
    # normalize_process = pexpect.spawn(f"./normalize-punctuation.perl -l {lang}", maxread=50000)
    normalize_process = pexpect.spawn(f"/bin/bash", maxread=50000)
    normalize_process.sendline('stty -icanon')
    normalize_process.sendline(f"./normalize-punctuation.perl -l {lang}")
    os.chdir(cwd)
    return normalize_process


def preprocess_wikipedia(file_path, output_dir, lang, tokenizer, sequence_length_range, start_doc_id=0):
    normalize_process = start_normalize_process(lang)
    sentences_by_number_of_words = {n: [] for n in range(sequence_length_range[0], sequence_length_range[1])}
    sentence_len_by_docs = {}
    doc_id_to_file_i = {}
    page = ""
    page_i = 0
    page_in_progress = False
    total_number_of_characters_in_current_file = 0
    file_i = 0
    doc_id = start_doc_id
    output_dir.mkdir(exist_ok=True, parents=True)
    current_file_path = output_dir / Path(str(file_i) + '.xml')
    out_f = current_file_path.open('w')
    with file_path.open() as in_f:
        for i, line in tqdm(enumerate(in_f)):
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
                        opening_doc_tag = f'<doc docid="{doc_id}" source="{file_path}"'
                        if title is not None:
                            opening_doc_tag += f' title="{QUOTES.sub("", title)}">'
                        else:
                            opening_doc_tag += '>'
                        file_text = '\n'.join([opening_doc_tag] + text + ['</doc>']) + '\n'
                        out_f.write(file_text)
                        for k, v in small.arrange_sentences_by_number_of_words_in_1_doc(
                                text, sequence_length_range, [file_i, doc_id]
                        ).items():
                            sentences_by_number_of_words[k] += v
                        sentence_len_by_docs[doc_id] = np.array(
                            [len(small.WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.findall(line)) for line in text]
                        )
                        doc_id_to_file_i[doc_id] = file_i
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
    return sentences_by_number_of_words, sentence_len_by_docs, doc_id_to_file_i


def prepend_file_i(not_whole_segments, doc_id_to_file_i):
    return np.concatenate([np.vectorize(doc_id_to_file_i.get)(not_whole_segments[:, 0]), not_whole_segments], 1)


def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    return True


def write_dataset(
    segments,
    sorted_file,
    line_start_pos,
    output_dir,
    create_model_input,
    bert_labels,
    autoregressive_labels,
    allowed_punctuation,
    only_first_punctuation_character_after_word_in_autoregressive,
    no_label_if_all_characters_are_upper_case,
):
    text_fn, input_fn = output_dir / Path('text.txt'), output_dir / Path('input.txt')
    bert_fn, ar_fn = output_dir / Path('bert_labels.txt'), output_dir / Path('autoregressive_labels.txt')
    with sorted_file.open() as in_f, \
            text_fn.open('w') as tf, \
            input_fn.open('w') as inp_f, \
            bert_fn.open('w') as bf, \
            ar_fn.open('w') as af:
        for i in tqdm(segments):
            in_f.seek(line_start_pos[i])
            line = in_f.readline().strip()
            tf.write(line + '\n')
            line = [s for s in small.WORD.split(line) if s]
            if create_model_input:
                inp_f.write(' '.join([s.lower() for s in line if small.WORD.match(s)]) + '\n')
            if bert_labels:
                bf.write(
                    small.create_bert_labels(
                        line, allowed_punctuation, no_label_if_all_characters_are_upper_case
                    ) + '\n'
                )
            if autoregressive_labels:
                af.write(
                    small.create_autoregressive_labels(
                        line,
                        allowed_punctuation,
                        only_first_punctuation_character_after_word_in_autoregressive,
                        no_label_if_all_characters_are_upper_case,
                    ) + '\n'
                )


def read_doc(fd):
    text = []
    line = fd.readline()
    while line and not line.startswith('</doc>'):
        text.append(line.strip())
        line = fd.readline()
    return text


def cut_and_save(segments, doc_dir, output_file):
    line_start_pos = []
    current_pos = 0
    current_file_i = -1
    current_fd = None
    current_doc = None
    current_doc_id = -1
    line_i = 0
    with output_file.open('w') as f:
        for segment in tqdm(segments):
            file_i = segment[0]
            doc_id = segment[1]
            if current_doc_id + 1 != doc_id:
                logging.warning(f"Documents are not in order: current_doc_id={current_doc_id}, doc_id={doc_id}")
            if current_file_i != file_i:
                current_file_i = file_i
                if current_fd is not None:
                    current_fd.close()
                current_fd = (doc_dir / Path(str(current_file_i) + '.xml')).open()
                current_doc_id = -1
                line_i = 0
            if current_doc_id != doc_id:
                line = 'FILLER'
                count = 0
                while line and not line.startswith(f'<doc doc_id="{doc_id}"'):
                    line = current_fd.readline()
                    count += 1
                    line_i += 1
                if count != 1:
                    logging.warning(
                        f"The next document is supposed to start right after previous document: "
                        f"file_i={file_i} current_doc_id={current_doc_id} doc_id={doc_id} count={count} line_i={line_i}"
                    )
                current_doc = read_doc(current_fd)
                line_i += len(current_doc)
            text_seg = small.cut_words(' '.join(current_doc[segment[2] : segment[3]]), segment[4], segment[5]) + '\n'
            line_start_pos.append(current_pos)
            f.write(text_seg)
            current_pos += len(text_seg)
    return np.array(line_start_pos)


def main():
    args = small.get_args(SUPPORTED_CORPUS_TYPES)
    tokenizer = get_tokenizer(args.tokenizer)
    number_of_sentences_in_input = 0
    sentences_by_number_of_words = {n: [] for n in range(args.sequence_length_range[0], args.sequence_length_range[1])}
    sentence_len_by_docs = {}
    doc_id_to_file_i = {}
    document_dir = args.output_dir / Path("documents")
    for corpus_type, file_path in zip(args.corpus_types, args.input_files):
        if corpus_type == SUPPORTED_CORPUS_TYPES[0]:
            logging.info(f"Preprocessing wikipedia file {file_path}...")
            res = preprocess_wikipedia(
                file_path, document_dir, args.input_language, tokenizer, args.sequence_length_range, 0
            )
            corpus_sentences_by_number_of_words, corpus_sentence_len_by_docs, corpus_doc_id_to_file_i = res
            for k, v in corpus_sentences_by_number_of_words.items():
                sentences_by_number_of_words[k] += v
            sentence_len_by_docs.update(corpus_sentence_len_by_docs)
            doc_id_to_file_i.update(corpus_doc_id_to_file_i)
        else:
            raise ValueError(
                f"Unsupported corpus type '{corpus_type}. Supported corpus types are {SUPPORTED_CORPUS_TYPES}"
            )
        number_of_corpus_sentences = sum([len(v) for v in corpus_sentences_by_number_of_words.values()])
        logging.info(
            f"Finished preprocessing corpus {file_path}. Number of sentences the corpus: "
            f"{number_of_corpus_sentences}, number of documents in the corpus: {len(corpus_sentence_len_by_docs)}"
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
    logging.info(f"Selecting segments with intact sentences...")
    result, number_of_words_stats, remaining_by_docs = small.select_close_to_uniform_distribution(
        sentences_by_number_of_words,
        args.size,
        args.percentage_segments_with_intact_sentences,
        {k: len(v) for k, v in sentence_len_by_docs.items()}
    )
    result = np.array(result)
    result = np.concatenate(
        [
            result,
            np.zeros([result.shape[0], 1], dtype=result.dtype),
            np.full([result.shape[0], 1], -1, dtype=result.dtype),
        ],
        1,
    )
    logging.info("Selecting segments with not intact sentences...")
    result = np.concatenate(
        [
            result,
            prepend_file_i(
                np.array(
                    small.create_not_whole_sentence_segments(
                        sentence_len_by_docs,
                        remaining_by_docs,
                        number_of_words_stats,
                        args.size,
                        args.percentage_segments_with_intact_sentences,
                    ),
                ),
                doc_id_to_file_i
            )
        ]
    )
    result = result[np.argsort(result[:, 0])]  # sort by file index
    result = result[np.argsort(result[:, 1], kind='stable')]  # sort by document index
    sorted_text_file = args.output_dir / Path('sorted_text.txt')
    logging.info("Cutting segments...")
    line_start_pos = cut_and_save(result, document_dir, sorted_text_file)
    order = np.random.permutation(result.shape[0])
    if args.dev_size > len(result):
        raise ValueError(f"Parameter `--dev_size={args.dev_size}` is less than size of all dataset ({len(result)})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_size = int(args.size * args.test_ratio / 100)
    if test_size > 0:
        logging.info("Writing test dataset...")
        write_dataset(
            order[:test_size],
            sorted_text_file,
            line_start_pos,
            args.output_dir / Path("test"),
            args.create_model_input,
            args.bert_labels,
            args.autoregressive_labels,
            args.allowed_punctuation,
            args.only_first_punctuation_character_after_word_in_autoregressive,
            args.no_label_if_all_characters_are_upper_case,
        )
    if args.dev_size > 0:
        logging.info("Writing dev dataset...")
        write_dataset(
            order[test_size : test_size + args.dev_size],
            sorted_text_file,
            line_start_pos,
            args.output_dir / Path("dev"),
            args.create_model_input,
            args.bert_labels,
            args.autoregressive_labels,
            args.allowed_punctuation,
            args.only_first_punctuation_character_after_word_in_autoregressive,
            args.no_label_if_all_characters_are_upper_case,
        )
    logging.info("Writing train dataset...")
    write_dataset(
        order[test_size + args.dev_size :],
        sorted_text_file,
        line_start_pos,
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
