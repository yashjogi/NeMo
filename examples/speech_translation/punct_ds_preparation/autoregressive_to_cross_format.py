import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from subprocess import run, PIPE

from tqdm import tqdm

from nemo.utils import logging


ENCODINGS = string.ascii_letters + string.digits
CAPITALIZATION_LABELS = 'uOU'
CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE = re.compile('[UuO][^uUO]*')
SPACE_BEFORE_PUNCTUATION = re.compile(' +([.,?!;):]+) *')


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--vocab", "-v", type=Path, required=True, help="Path to file where transition vocabulary will be saved."
    )
    parser.add_argument("--not_normalized_vocabulary", "-V", type=Path)
    parser.add_argument("--normalize", "-n", action="store_true")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    args.vocab = args.vocab.expanduser()
    args.not_normalized_vocabulary = args.not_normalized_vocabulary.expanduser()
    return args


def normalize(line):
    return SPACE_BEFORE_PUNCTUATION.sub(r'\1 ', line)


def collect_cross_vocabulary(input_file):
    usual_vocabulary = Counter()
    normalized_vocabulary = Counter()
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    num_lines = int(result.stdout.decode('utf-8').split()[0])
    with input_file.open() as f:
        for line in tqdm(f, total=num_lines, unit='line', desc="Creating vocabulary"):
            usual_vocabulary.update(CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(line.strip() + ' '))
            normalized_vocabulary.update(
                CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(normalize(line.strip() + ' '))
            )
    usual_vocabulary = dict(sorted(usual_vocabulary.items(), key=lambda x: -x[1]))
    normalized_vocabulary = dict(sorted(normalized_vocabulary.items(), key=lambda x: -x[1]))
    return usual_vocabulary, normalized_vocabulary, num_lines


def autoregressive_file_to_cross_format_file(
        input_file, output_file, vocab_file, normalize_vocabulary, save_not_normalized_vocabulary
):
    combinations, normalized_combinations, num_lines = collect_cross_vocabulary(input_file)
    if len(combinations) > len(ENCODINGS):
        raise ValueError(
            f"Too many cross labels were found in file {input_file}. Number of cross labels: {len(combinations)}, "
            f"number of available encodings: {len(ENCODINGS)}. You probably need to add more characters to parameter "
            f"`ENCODINGS` if you wish to process file {input_file}."
        )
    vocabulary = {}
    for i, (s, ctr) in enumerate(combinations.items()):
        vocabulary[ENCODINGS[i]] = {'string': s, "count": ctr}
    normalized_vocabulary = {}
    for i, (s, ctr) in enumerate(normalized_combinations.items()):
        normalized_vocabulary[ENCODINGS[i]] = {'string': s, "count": ctr}
    with vocab_file.open('w') as f:
        json.dump(normalized_vocabulary if normalize_vocabulary else vocabulary, f, indent=2)
    if normalized_vocabulary and save_not_normalized_vocabulary:
        not_normalized_vocab_file = Path(str(vocab_file) + '.not_normalized')
        with not_normalized_vocab_file.open('w') as f:
            json.dump(vocabulary, f, indent=2)
    inverse_vocabulary = {v['string']: k for k, v in vocabulary.items()}
    with input_file.open() as in_f, output_file.open('w') as out_f:
        for line in tqdm(in_f, total=num_lines, unit='line', desc="Encoding"):
            if normalized_vocabulary:
                line = normalize(line)
            for piece in CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(line.strip()):
                out_f.write(inverse_vocabulary[piece])
        out_f.write('\n')


def main():
    args = get_args()
    autoregressive_file_to_cross_format_file(args.input, args.output, args.vocab)


if __name__ == "__main__":
    main()
