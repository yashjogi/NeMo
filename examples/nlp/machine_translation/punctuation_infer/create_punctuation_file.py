import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input", "-i", help="Path to input file with punctuation and capitalization labels", type=Path, required=True
    )
    parser.add_argument("--output", '-o', help="Path to output json file.", type=Path, required=True)
    parser.add_argument("--capitalization_labels", "-c", help="String with punctuation labels", default="uUO")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    rstrip = re.compile(f"[^{args.capitalization_labels}]*$")
    capitalization_re = re.compile(f'[{args.capitalization_labels}]', flags=re.I)
    with args.input.open() as f:
        punctuation = Counter()
        for line in f:
            line = line.strip()
            if ' ' not in rstrip.search(line).group(0):
                line += ' '
            punctuation.update(capitalization_re.split(line))
    punctuation = dict(sorted(punctuation.items(), key=lambda x: -x[1]))
    del punctuation['']
    with args.output.open('w') as f:
        json.dump(punctuation, f, indent=2)


if __name__ == "__main__":
    main()
