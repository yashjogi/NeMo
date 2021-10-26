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
            started = False
            for piece in line:
                if started:
                    if piece in CAPITALIZATION_LABELS:
                        pass



def autoregressive_file_to_cross_format_file(input_file, output_file, vocab_file):
    vocabulary = collect_cross_vocabulary(input_file)


def main():
    args = get_args()
    autoregressive_file_to_cross_format_file(args.input, args.output, args.vocab)


if __name__ == "__main__":
    main()