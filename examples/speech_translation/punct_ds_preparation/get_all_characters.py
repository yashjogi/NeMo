import argparse
from pathlib import Path


BLOCK_SIZE = 2 ** 16


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    characters = set()
    with args.input.open() as in_f:
        block = "filler"
        while block:
            block = in_f.read(BLOCK_SIZE)
            characters.update(block)
    for c in '\n\t \r\v':
        characters.remove(c)
    characters = sorted(characters, key=lambda x: ord(x))
    with args.output.open('w') as out_f:
        for c in characters:
            out_f.write(f"{ord(c)} '{c}'\n")


if __name__ == "__main__":
    main()
