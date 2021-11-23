import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from subprocess import run, PIPE

from tqdm import tqdm


ENCODINGS = string.ascii_letters + string.digits
CAPITALIZATION_LABELS = 'uOU'
CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE = re.compile('[UuO][^uUO]*')
SPACE_BEFORE_PUNCTUATION = re.compile(' +([.,?!;):]+) *')


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    vocab = parser.add_mutually_exclusive_group(required=True)
    vocab.add_argument(
        "--output_vocab", "-v", type=Path, help="Path to JSON file where transition vocabulary will be saved."
    )
    vocab.add_argument("--ready_vocab", "-r", type=Path)
    parser.add_argument(
        "--not_normalized_vocabulary",
        "-V",
        type=Path,
        help="A path to a directory where vocabulary of not normalized combinations are saved. Normalization is just "
        "a removal of spaces before punctuation.",
    )
    parser.add_argument("--normalize", "-n", action="store_true")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    if args.ready_vocab is None:
        args.output_vocab = args.output_vocab.expanduser()
    else:
        args.ready_vocab = args.ready_vocab.expanduser()
    if args.output_vocab is None and args.not_normalized_vocabulary:
        raise ValueError(
            f"If parameter `--ready_vocab` is provided you should not provide parameter "
            f"`--not_normalized_vocabulary`."
        )
    if args.not_normalized_vocabulary is not None:
        args.not_normalized_vocabulary = args.not_normalized_vocabulary.expanduser()
    return args


def normalize(line):
    return SPACE_BEFORE_PUNCTUATION.sub(r'\1 ', line)


def get_num_lines(input_file):
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    return int(result.stdout.decode('utf-8').split()[0])


def collect_cross_vocabulary(input_file):
    usual_vocabulary = Counter()
    normalized_vocabulary = Counter()
    num_lines = get_num_lines(input_file)
    with input_file.open() as f:
        for line in tqdm(f, total=num_lines, unit='line', desc="Creating vocabulary"):
            usual_vocabulary.update(CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(line.strip() + ' '))
            normalized_vocabulary.update(
                CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(normalize(line.strip() + ' '))
            )
    usual_vocabulary = dict(sorted(usual_vocabulary.items(), key=lambda x: -x[1]))
    normalized_vocabulary = dict(sorted(normalized_vocabulary.items(), key=lambda x: -x[1]))
    return usual_vocabulary, normalized_vocabulary, num_lines


def encode(input_file, output_file, num_lines, inverse_vocabulary, normalize_labels):
    with input_file.open() as in_f, output_file.open('w') as out_f:
        for line in tqdm(in_f, total=num_lines, unit='line', desc="Encoding"):
            if normalize_labels:
                line = normalize(line)
            for piece in CAPITALIZATION_WITH_FOLLOWING_PUNCTUATION_RE.findall(line.strip() + ' '):
                if piece in inverse_vocabulary:
                    out_f.write(inverse_vocabulary[piece])
                else:
                    raise ValueError(f"String '{piece}' is not found in vocabulary.")
            out_f.write('\n')


def autoregressive_file_to_cross_format_file(
    input_file, output_file, ready_vocab_file, output_vocab_file, normalize_labels, not_normalized_vocab_file
):
    if ready_vocab_file is None:
        combinations, normalized_combinations, num_lines = collect_cross_vocabulary(input_file)
        if len(combinations) > len(ENCODINGS):
            raise ValueError(
                f"Too many cross labels were found in file {input_file}. Number of cross labels: {len(combinations)}, "
                f"number of available encodings: {len(ENCODINGS)}. You probably need to add more characters to "
                f"parameter `ENCODINGS` if you wish to process file {input_file}."
            )
        vocabulary = {}
        for i, (s, ctr) in enumerate(combinations.items()):
            vocabulary[ENCODINGS[i]] = {'string': s, "count": ctr}
        normalized_vocabulary = {}
        for i, (s, ctr) in enumerate(normalized_combinations.items()):
            normalized_vocabulary[ENCODINGS[i]] = {'string': s, "count": ctr}
        with output_vocab_file.open('w') as f:
            json.dump(normalized_vocabulary if normalize_labels else vocabulary, f, indent=2)
        if normalized_vocabulary and not_normalized_vocab_file is not None:
            with not_normalized_vocab_file.open('w') as f:
                json.dump(vocabulary, f, indent=2)
        inverse_vocabulary = {
            v['string']: k for k, v in (normalized_vocabulary.items() if normalize_labels else vocabulary.items())
        }
    else:
        with ready_vocab_file.open() as f:
            ready_vocabulary = json.load(f)
        inverse_vocabulary = {v['string']: k for k, v in ready_vocabulary.items()}
        num_lines = get_num_lines(input_file)
    encode(input_file, output_file, num_lines, inverse_vocabulary, normalize_labels)


def main():
    args = get_args()
    autoregressive_file_to_cross_format_file(
        args.input, args.output, args.ready_vocab, args.output_vocab, args.normalize, args.not_normalized_vocabulary
    )


if __name__ == "__main__":
    main()
