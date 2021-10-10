import logging
import os
import random
import re
from pathlib import Path
from subprocess import PIPE, Popen

import numpy as np
from tqdm import tqdm

import nltk
from nemo.collections.nlp.modules import get_tokenizer

import prepare_small_data_for_punctuation_capitalization_task as small

logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')

random.seed(42)

SUPPORTED_CORPUS_TYPES = ["wikipedia"]


def create_triplet(tag):
    start = re.compile(f'<{tag}[^>]*>')
    end = re.compile(f'</{tag}>')
    start_or_end = re.compile(start.pattern + '|' + end.pattern)
    return start, end, start_or_end


PAGE_OPENING_NORMAL_TAG = re.compile(r'^ *<page>$')
PAGE_CLOSING_NORMAL_TAG = re.compile(r'^ *</page>$')
TITLE_OF_PAGE = re.compile(r'<title>(.+)</title>')
TEXT_OF_PAGE = re.compile(r'<text[^>]*>(.+)</text>', flags=re.DOTALL)
QUOTES = re.compile('"\'')
REDIRECT = re.compile(r'^\s*#REDIRECT +\[\[[^]]*]]', flags=re.I)
DOUBLE_BRACES_WITH_CONTENT = re.compile(r'{{[^}{]*}}|\({{[^}{]*}}\)')
TABLE = re.compile('{|')
EQUALS_SIGN_HEADERS = re.compile('^[ \t]*==+[^\n=]+==+[ \t]*$', flags=re.MULTILINE)
SPECIAL_SQUARE_BRACKETS_START = re.compile(
    r'\[\[(?:[ :]{,2}File:|[ :]{,2}Image:|[ :]{,2}User:|[ :]{,2}User talk:|[ :]{,2}Special:| ?#footer| '
    r'?{{rdconfigarray)',
    flags=re.I
)
SPECIAL_SQUARE_BRACKETS_BORDER = re.compile(r'\[\[|]]')
SINGLE_SQUARE_BRACKETS_WITH_CONTENT = re.compile(r'(?<!\[)\[([^][]*)](?!])')
DOUBLE_SQUARE_BRACKETS_WITH_CONTENT = re.compile(r'\[\[([^][]*)]]')
TRIPLE_QUOTES = re.compile(r"'''([^']+)'''")
END_SECTION = re.compile(
    r"==\s*(?:See also|References|Notes|Sources|Primary sources|Secondary sources|External links)\s*=="
)
NORMALIZE_ENDING_PATTERN = re.compile(b'.*EOFEOFEOF', flags=re.DOTALL)
NEW_LINE_DUP = re.compile('\n{2,}')
DOC_HEAD = re.compile(
    '^<doc docid="({[0-9]*})" source="(.+)" title="(.+)" start_line="([0-9]+)" end_line="([0-9]+)">$',
    flags=re.MULTILINE
)
DOC_HEAD_TMPL = '<doc docid="{}" source="{}" title="{}" start_line="{}" end_line="{}">'
DOC_END = '</doc>'
# EM_TAG = re.compile('</?em>')
# BLOCKQUOTE_TAG = re.compile('</?blockquote>')
# DIV_TAG = re.compile('</?div[^>]*>')
AMP_DEL = re.compile(r'(\w)&amp;')
# SUP_TAG = re.compile(r'</?sup>')
# SPAN_TAG = re.compile(r'</?span[^>]*>')
DROP_TAGS = re.compile(r'</?(div|su[pb]|span|blockquote|em|big|small|s|br|nowiki)[^>]*>')
REFERENCE = re.compile('<ref[^>]*>[^<]*</ref>')
REFERENCE_SHORT = re.compile('<ref[^>]*/>')
REF_START, REF_END, REF_START_OR_END = create_triplet('ref')
MATH_START, MATH_END, MATH_START_OR_END = create_triplet('math')
# TABLE_START = re.compile('^:{,2}{\\|(?:[^\n]*\n\\|\n{\\|)?', flags=re.MULTILINE)
TABLE_START = re.compile('^:{,2}{\\|', flags=re.MULTILINE)
TABLE_END = re.compile('\n\\|}')
TABLE_START_OR_END = re.compile(TABLE_START.pattern + '|' + TABLE_END.pattern, flags=re.MULTILINE)
GALLERY_START, GALLERY_END, GALLERY_START_OR_END = create_triplet('gallery')
IMAGEMAP_START, IMAGEMAP_END, IMAGEMAP_START_OR_END = create_triplet('imagemap')
SCORE_START, SCORE_END, SCORE_START_OR_END = create_triplet('score')
CODE_START, CODE_END, CODE_START_OR_END = create_triplet('code')
EMPTY_PARENTHESES = re.compile(r' *\([ .,!;?|&#%^@$"\'<>{}/\\*~\][]*\) *')
DOUBLE_BRACES_START = re.compile('{{')
DOUBLE_BRACES_END = re.compile('}}')
DOUBLE_BRACES_START_OR_END = re.compile(DOUBLE_BRACES_START.pattern + '|' + DOUBLE_BRACES_END.pattern)

MAX_NUM_CHARACTERS_IN_1_FILE = 10 ** 6


def remove_remarks(text):
    result = ""
    remarks_in_progress = 0
    for i in range(len(text)):
        if text[i: i + 4] == '<!--':
            remarks_in_progress += 1
        if remarks_in_progress == 0:
            result += text[i]
        if text[i - 2: i + 1] == '-->':
            remarks_in_progress -= 1
    return result


def remove_tag_with_content(text, start_re, end_re, remove_whole_line, pos_info):
    result = ""
    start_iter = start_re.finditer(text)
    end_iter = end_re.finditer(text)
    last_end = 0
    for start_m, end_m in zip(start_iter, end_iter):
        if start_m.span()[0] >= end_m.span()[0]:
            logging.warning(
                f"Encountered closing tag {repr(end_m.group(0))} in position {end_m.span()[0]} before or simultaneously "
                f"with opening tag {repr(start_m.group(0))} in position {start_m.span()[0]}. start_re={start_re}, "
                f"end_re={end_re}. Document is in file {pos_info[0]} lines between {pos_info[1]} and {pos_info[2]}. "
                f"Discarding the remainder of the document."
            )
            return result
        if start_m.span()[0] < last_end:
            if remove_whole_line:
                if end_m.span()[0] > last_end:
                    logging.warning(
                        f"Encountered closing tag {repr(end_m.group(0))} in position {end_m.span()[0]} in not parsed "
                        f"text (starting with position {last_end}) whereas no starting tag {start_re} was found in not "
                        f"parsed text. Probably tags {start_re} and {end_re} are multiline. Document is in lines "
                        f"between {pos_info[1]} and {pos_info[2]} in file {pos_info[0]}. Discarding the remainder of "
                        f"the document."
                    )
                    return result
                continue
            else:
                logging.warning(
                    f"Encountered 2 opening tags with regex '{start_re.pattern}' (the last match '{start_m.group(0)}' "
                    f"in position {start_m.span()[0]}) before closing tag with regex '{end_re.pattern}' in position "
                    f"{last_end}. Probably here nested tags are used. Document is in lines between {pos_info[1]} and "
                    f"{pos_info[2]} in file {pos_info[0]}. Discarding the remainder of the document."
                )
                return result
        if remove_whole_line:
            ind = text.rfind('\n', last_end, start_m.span()[0])
            if ind == -1:
                ind = last_end
            result += text[last_end: ind]
            last_end = text.find('\n', end_m.span()[1])
        else:
            result += text[last_end: start_m.span()[0]]
            last_end = end_m.span()[1]
    if last_end > 0:
        result += text[last_end:]
    return result


def remove_tag_with_content_nested(text, start_re, end_re, start_or_end_re, remove_whole_line, pos_info):
    result = ""
    num_opened = 0
    last_end = 0
    for m in start_or_end_re.finditer(text):
        if start_re.match(m.group(0)) is not None:
            if num_opened == 0:
                right = text.rfind('\n', last_end, m.span()[0]) if remove_whole_line else m.span()[0]
                result += text[last_end: right]
            num_opened += 1
        else:
            if num_opened == 0:
                logging.warning(
                    f"Encountered closing tag {repr(m.group(0))} in position {m.span()[0]} before starting tag. "
                    f"10 characters and 10 characters after: {repr(text[max(m.span()[0] - 10, 0): m.span()[1] + 10])}. "
                    f"Probably the tag is multiline or there is an error in page markup. start_re={start_re}, "
                    f"end_re={end_re}. Document is in file {pos_info[0]} lines between {pos_info[1]} and "
                    f"{pos_info[2]}. Discarding the document after {last_end}."
                )
                return result
            else:
                num_opened -= 1
                if num_opened == 0:
                    last_end = text.find('\n', m.span()[1]) if remove_whole_line else m.span()[1]
    result += text[last_end:]
    return result


def remove_double_square_brackets_specials(text, pos_info):
    result = ""
    last_end = 0
    for m in SPECIAL_SQUARE_BRACKETS_START.finditer(text):
        start = m.span()[0]
        search_start = m.span()[1]
        result += text[last_end: start]
        num_openings = 1
        while num_openings > 0:
            mm = SPECIAL_SQUARE_BRACKETS_BORDER.search(text, search_start)
            if mm is None:
                logging.warning(
                    f"Encountered special square brackets without closing starting in position {start} of document in "
                    f"file {pos_info[0]} located in lines between {pos_info[1]} and {pos_info[2]}. The part of the "
                    f"document starting from position {start} will be discarded."
                )
                return result
            if mm.group(0) == ']]':
                num_openings -= 1
            else:
                num_openings += 1
            if num_openings > 0:
                search_start = mm.span()[1]
        last_end = mm.span()[1]
    return result + text[last_end:]


def get_wiki_text_lines(text, tokenizer, tok_chars, untok_chars, pos_info):
    text = small.SPACING_CHARACTERS_TO_REPLACE.sub(' ', text)
    text = REDIRECT.sub('', text)
    text = text.strip()
    if not text:
        return [], tok_chars, untok_chars
    end_section = END_SECTION.search(text)
    if end_section is not None:
        text = text[:end_section.span()[0]].strip()
    text = EQUALS_SIGN_HEADERS.sub('\n', text)
    # text = remove_tag_with_content_nested(
    #     text, DOUBLE_BRACES_START, DOUBLE_BRACES_END, DOUBLE_BRACES_START_OR_END, False, pos_info
    # )
    text = remove_double_square_brackets_specials(text, pos_info)

    # text = remove_tag_with_content_nested(text, TABLE_START, TABLE_END, TABLE_START_OR_END, True, pos_info)
    # text = remove_tag_with_content(text, TABLE_START, TABLE_END, True, pos_info)
    text = TRIPLE_QUOTES.sub(r'\1', text)
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = REFERENCE.sub('', text)
    text = REFERENCE_SHORT.sub('', text)
    # text = remove_tag_with_content(text, REF_START, REF_END, True, pos_info)
    text = remove_tag_with_content_nested(text, REF_START, REF_END, REF_START_OR_END, False, pos_info)
    # text = remove_tag_with_content(text, MATH_START, MATH_END, True, pos_info)
    text = remove_tag_with_content_nested(text, MATH_START, MATH_END, MATH_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, CODE_START, CODE_END, CODE_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(
        text, DOUBLE_BRACES_START, DOUBLE_BRACES_END, DOUBLE_BRACES_START_OR_END, False, pos_info
    )
    text = remove_tag_with_content_nested(text, TABLE_START, TABLE_END, TABLE_START_OR_END, True, pos_info)
    text = DROP_TAGS.sub('', text)
    text = text.replace('<doc doc_id"', '')
    text = text.replace('</doc>', '')
    text = SINGLE_SQUARE_BRACKETS_WITH_CONTENT.sub(r'(\1)', text)

    def double_square_brackets_replacement(match):
        match_text = match.group(1)
        match_text = match_text.split('|')
        if len(match_text) == 1:
            res = match_text[0]
        elif len(match_text) == 2:
            res = match_text[1]
        else:
            logging.warning(
                f"Found double square brackets with three sections {repr(match.group(0))} in document from lines "
                f"between {pos_info[1]} and {pos_info[2]}."
            )
            res = match_text[1]
        if ':' in res:
            split_res = res.split(':')
            if split_res[1]:
                res = split_res[1]
            else:
                res = split_res[0]
        if (
                "#if:" in res
                or "&amp;" in res
                or '\n' in res
                or "#ifeq:" in res
                or '{}' in res
                or "{{" in res
                or "}}" in res
                or "lc:" in res

        ):
            res = ""
        return res

    text = DOUBLE_SQUARE_BRACKETS_WITH_CONTENT.sub(double_square_brackets_replacement, text)
    text = remove_remarks(text)
    text = text.replace("''", '"')
    text = text.replace("&quot;", '"')
    text = text.replace('&nbsp;', ' ')
    text = AMP_DEL.sub(r'\1', text)
    text = text.replace('&amp;', 'and')
    text = EMPTY_PARENTHESES.sub(' ', text)
    text = remove_tag_with_content_nested(text, GALLERY_START, GALLERY_END, GALLERY_START_OR_END, False, pos_info)
    text = remove_tag_with_content_nested(text, IMAGEMAP_START, IMAGEMAP_END, IMAGEMAP_START_OR_END, False, pos_info)
    text = remove_tag_with_content_nested(text, SCORE_START, SCORE_END, SCORE_START_OR_END, True, pos_info)
    text = NEW_LINE_DUP.sub('\n', text)
    if text and text[-1] != '\n':
        text += '\n'
    if tokenizer is not None:
        text, tok_chars, untok_chars = small.remove_untokenizable_characters_from_text(text, tokenizer)
    if '<' in text or '>' in text:
        logging.warning(
            f"There are still 'greater than' or 'less than' signs in document in file {pos_info[0]} between lines "
            f"{pos_info[1]} and {pos_info[2]}."
        )
    stripped = [sent.strip() for sent in nltk.sent_tokenize(text)]
    return [sent for sent in stripped if sent], tok_chars, untok_chars


def start_normalize_process(lang):
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)
    normalize_process = Popen(['./normalize-punctuation.perl', '-l', lang], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    os.chdir(cwd)
    return normalize_process


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b.count('\n')


def count_lines_in_file(file_path):
    with file_path.open() as f:
        count = sum(blocks(f))
    return count


def preprocess_wikipedia(file_path, output_dir, tokenizer, sequence_length_range, start_doc_id=0):
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
    tok_chars, untok_chars = {'\n', ' '}, set()
    with file_path.open() as in_f:
        for i, line in tqdm(enumerate(in_f), total=count_lines_in_file(file_path)):
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
                    text, tok_chars, untok_chars = get_wiki_text_lines(
                        text.group(1), tokenizer, tok_chars, untok_chars, [file_path, start_line, end_line]
                    )
                    if text:
                        file_text = doc_to_str(doc_id, file_path, title, start_line, end_line, '\n'.join(text))
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


def read_docs_from_file(file_path):
    current_doc = ""
    current_doc_id = None
    docs = {}
    with file_path.open() as f:
        for i, line in enumerate(f):
            start = DOC_HEAD.match(line)
            if start is not None:
                if current_doc_id is not None:
                    raise ValueError(
                        f"Encountered start of document number {start.group(1)} on line {i} in file {file_path} while "
                        f"document number {current_doc_id} is still in progress."
                    )
                current_doc_id = int(start.group(1))
            if line.startswith("</doc>"):
                if current_doc_id is None:
                    raise ValueError(
                        f"Encountered end of document on line {i} in file {file_path} while there is no document in "
                        f"progress."
                    )
                docs[current_doc_id] = {
                    "source": start.group(2),
                    "title": start.group(3),
                    "start_line": start.group(4),
                    "end_line": start.group(5),
                    "text": current_doc
                }
                current_doc = ""
                current_doc_id = None
            if current_doc_id is not None:
                current_doc += line
    return docs


def doc_to_str(docid, source, title, start_line, end_line, text):
    res = DOC_HEAD_TMPL.format(docid, source, title, start_line, end_line) + '\n' + text
    if text[-1] != '\n':
        res += '\n'
    return res + DOC_END + '\n'


def write_docs_to_file(docs, file_path):
    with file_path.open('w') as f:
        for k, v in docs.items():
            f.write(doc_to_str(k, v['source'], v["title"], v["start_line"], v["end_line"], v["text"]))


def normalize_punctuation_in_all_documents(document_dir, lang):
    normalize_process = start_normalize_process(lang)
    for p in tqdm(list(document_dir.iterdir())):
        if is_int(p.stem) and p.suffixes == ['.xml']:
            file_docs = read_docs_from_file(p)
            outs, errs = normalize_process.communicate(
                '\n\n\n'.join([doc['text'] for doc in file_docs.values()]).encode('utf-8')
            )
            for k, text in zip(file_docs, outs.decode('utf-8').split('\n\n\n')):
                file_docs[k]["text"] = text
            write_docs_to_file(file_docs, Path(str(p) + '.norm'))


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
            res = preprocess_wikipedia(file_path, document_dir, tokenizer, args.sequence_length_range, 0)
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
    logging.info("Normalizing punctuation...")
    normalize_punctuation_in_all_documents(document_dir, args.input_language)
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
