import argparse
import re
import string
from collections import Counter
from pathlib import Path


ENCODINGS = string.ascii_letters + string.digits
CAPITALIZATION_LABELS = 'uOU'
CAPITALIZATION_RE = re.compile('([UuO])')


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
            line = CAPITALIZATION_RE.split(line.strip())
            start_i = 0
            while line[start_i] not in CAPITALIZATION_LABELS:
                start_i += 1
            assert (len(line) - start_i) % 2 == 0
            vocabulary.update([''.join(line[i: i + 2]) for i in range(start_i, (len(line) - start_i) // 2)])
    return dict(sorted(vocabulary.items(), key=lambda x: -x[1]))


def autoregressive_file_to_cross_format_file(input_file, output_file, vocab_file):
    vocabulary = collect_cross_vocabulary(input_file)
    if len(vocabulary) > len(ENCODINGS):
        raise ValueError(
            f"Too many cross labels were found in file {input_file}. Number of cross labels: {len(vocabulary)}, "
            f"number of available encodings: {len(ENCODINGS)}. You probably need to add more characters to parameter "
            f"`ENCODINGS` if you wish to process file {input_file}."
        )
    with vocab_file.open('w') as f:
        for i, k in enumerate(vocabulary):
            f.write



def main():
    args = get_args()
    autoregressive_file_to_cross_format_file(args.input, args.output, args.vocab)


if __name__ == "__main__":
    main()