import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--vocab", "-v", type=Path, required=True)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    args.vocab = args.vocab.expanduser()
    return args


def main():
    args = get_args()
    with args.vocab.open() as f:
        vocab = json.load(f)
    with args.input.open() as in_f, args.output.open('w') as out_f:
        for line in in_f:
            out_f.write(''.join([vocab[lbl]['string'] for lbl in line.strip()]) + '\n')


if __name__ == "__main__":
    main()
