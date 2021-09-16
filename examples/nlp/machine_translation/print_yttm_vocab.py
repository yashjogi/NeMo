import argparse
from pathlib import Path

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    args.model = args.model.expanduser()
    return args


def main():
    args = get_args()
    tokenizer = get_nmt_tokenizer(
        library="yttm",
        tokenizer_model=args.model,
    )
    for i in range(tokenizer.vocab_size):
        print(i, repr(tokenizer.ids_to_tokens([i])[0]))


if __name__ == "__main__":
    main()
