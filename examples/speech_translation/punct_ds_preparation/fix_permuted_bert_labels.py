import argparse
from pathlib import Path


BUFFER_SIZE = 2 ** 20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=Path, required=True)
    parser.add_argument('--output', '-o', type=Path, required=True)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    with args.input.open(buffering=BUFFER_SIZE) as in_f, args.output.open('w', buffering=BUFFER_SIZE) as out_f:
        for line in in_f:
            line = line.split()
            for i, pair in enumerate(line):
                out_f.write(pair[1] + pair[0])
                if i < len(line) - 1:
                    out_f.write(' ')
            out_f.write('\n')


if __name__ == "__main__":
    main()
