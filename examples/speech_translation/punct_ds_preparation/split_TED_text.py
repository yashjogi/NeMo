import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--segment_length", "-L", type=int)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    segments = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as in_f, args.output.open('w') as out_f:
        for line in in_f:
            line = line.split()
            i = 0
            while (i + 1) * args.length < len(line):
                segments.append(' '.join(line[i * args.length: (i + 1) * args.length]))
                i += 1
        out_f.write('\n'.join(segments) + '\n')


if __name__ == "__main__":
    main()
