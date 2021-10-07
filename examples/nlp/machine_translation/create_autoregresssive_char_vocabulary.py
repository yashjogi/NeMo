import argparse
from pathlib import Path

from nemo.collections.common.tokenizers import CharTokenizer


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--characters_to_exclude", "-E", nargs="+")
    parser.add_argument("--vocab_size", "-v", type=int)
    parser.add_argument("--mask_token", "-m")
    parser.add_argument("--bos_token", "-b")
    parser.add_argument("--eos_token", "-e")
    parser.add_argument("--pad_token", "-p")
    parser.add_argument("--sep_token", "-s")
    parser.add_argument("--cls_token", "-c")
    parser.add_argument("--unk_token", "-u")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    CharTokenizer.build_vocab(
        args.output,
        text_file_name=args.input,
        characters_to_exclude=args.characters_to_exclude,
        vocab_size=args.vocab_size,
        mask_token=args.mask_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        sep_token=args.sep_token,
        cls_token=args.cls_token,
        unk_token=args.unk_token,
    )


if __name__ == "__main__":
    main()
