import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


ENCODINGS = string.ascii_letters + string.digits
CAPITALIZATION_LABELS = 'uOU'
CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE = re.compile('[UuO][^uUO]*')


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--vocab", "-v", type=Path, required=True, help="Path to file where transition vocabulary will be saved."
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    args.vocab = args.vocab.expanduser()
    return args


def collect_cross_vocabulary(input_file):
    vocabulary = Counter()
    with input_file.open() as f:
        for line in f:
            vocabulary.update(CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(line.strip()))
    return dict(sorted(vocabulary.items(), key=lambda x: -x[1]))


def autoregressive_file_to_cross_format_file(input_file, output_file, vocab_file):
    combinations = collect_cross_vocabulary(input_file)
    if len(combinations) > len(ENCODINGS):
        raise ValueError(
            f"Too many cross labels were found in file {input_file}. Number of cross labels: {len(combinations)}, "
            f"number of available encodings: {len(ENCODINGS)}. You probably need to add more characters to parameter "
            f"`ENCODINGS` if you wish to process file {input_file}."
        )
    vocabulary = {}
    for i, (s, ctr) in enumerate(combinations.items()):
        vocabulary[ENCODINGS[i]] = {'string': s, "count": ctr}
    with vocab_file.open('w') as f:
        json.dump(vocabulary, f)
    inverse_vocabulary = {v['string']: k for k, v in vocabulary.items()}
    with input_file.open() as in_f, output_file.open() as out_f:
        for line in in_f:
            for piece in CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(line.strip()):
                out_f.write(inverse_vocabulary[piece])
        out_f.write('\n')


def main():
    args = get_args()
    autoregressive_file_to_cross_format_file(args.input, args.output, args.vocab)


if __name__ == "__main__":
    main()
