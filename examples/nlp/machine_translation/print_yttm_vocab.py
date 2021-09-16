import argparse

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


def main():
    args = get_args()
    tokenizer = get_nmt_tokenizer(
        library="yttm",
        tokenizer_model=args.model,
    )
    print("bos id:", tokenizer.bos_id)
    print("eos id:", tokenizer.eos_id)
    print("pad id:", tokenizer.pad_id)
    print("unk id:", tokenizer.unk_id)



if __name__ == "__main__":
    main()
