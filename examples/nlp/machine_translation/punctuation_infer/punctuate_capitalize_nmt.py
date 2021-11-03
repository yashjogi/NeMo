import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="The script is for restoring punctuation and capitalization in text. Long strings are split into "
        "segments of length `--max_seq_length`. `--max_seq_length` is the length which includes [CLS] and [SEP] "
        "tokens. Parameter `--step` controls segments overlapping. `--step` is a distance between beginnings of "
        "consequent segments. Model outputs for tokens near the borders of tensors are less accurate and can be "
        "discarded before final predictions computation. Parameter `--margin` is number of discarded outputs near "
        "segments borders. If model predictions in overlapping parts of segments are different most frequent "
        "predictions is chosen.",
    )
    input_ = parser.add_mutually_exclusive_group(required=True)
    input_.add_argument(
        "--input_manifest",
        "-m",
        type=Path,
        help="Path to the file with NeMo manifest which needs punctuation and capitalization. If the first element "
        "of manifest contains key 'pred_text', 'pred_text' values are passed for tokenization. Otherwise 'text' "
        "values are passed for punctuation and capitalization. Exactly one parameter of `--input_manifest` and "
        "`--input_text` should be provided.",
    )
    input_.add_argument(
        "--input_text",
        "-t",
        type=Path,
        help="Path to file with text which needs punctuation and capitalization. Exactly one parameter of "
        "`--input_manifest` and `--input_text` should be provided.",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output_manifest",
        "-M",
        type=Path,
        help="Path to output NeMo manifest. Text with restored punctuation and capitalization will be saved in "
        "'pred_text' elements if 'pred_text' key is present in the input manifest. Otherwise text with restored "
        "punctuation and capitalization will be saved in 'text' elements. Exactly one parameter of `--output_manifest` "
        "and `--output_text` should be provided.",
    )
    output.add_argument(
        "--output_text",
        "-T",
        type=Path,
        help="Path to file with text with restored punctuation and capitalization. Exactly one parameter of "
        "`--output_manifest` and `--output_text` should be provided.",
    )
    parser.add_argument(
        "--model_path",
        "-P",
        type=Path,
        help=f"Path to .nemo checkpoint of `MTEncDecModel`. No more than one of parameters ",
    )
    parser.add_argument(
        "--max_seq_length",
        "-L",
        type=int,
        default=64,
        help="Length of segments into which queries are split. `--max_seq_length` includes [CLS] and [SEP] tokens.",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=8,
        help="Relative shift of consequent segments into which long queries are split. Long queries are split into "
        "segments which can overlap. Parameter `step` controls such overlapping. Imagine that queries are "
        "tokenized into characters, `max_seq_length=5`, and `step=2`. In such a case query 'hello' is tokenized "
        "into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.",
    )
    parser.add_argument(
        "--margin",
        "-g",
        type=int,
        default=16,
        help="A number of subtokens in the beginning and the end of segments which output probabilities are not used "
        "for prediction computation. The first segment does not have left margin and the last segment does not have "
        "right margin. For example, if input sequence is tokenized into characters, `max_seq_length=5`, `step=1`, "
        "and `margin=1`, then query 'hello' will be tokenized into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], "
        "['[CLS]', 'e', 'l', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`. These segments are passed to the "
        "model. Before final predictions computation, margins are removed. In the next list, subtokens which logits "
        "are not used for final predictions computation are marked with asterisk: `[['[CLS]'*, 'h', 'e', 'l'*, "
        "'[SEP]'*], ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]`.",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=128, help="Number of segments which are processed simultaneously.",
    )
    parser.add_argument(
        "--save_labels_instead_of_text",
        "-B",
        action="store_true",
        help="If this option is set, then punctuation and capitalization labels are saved instead text with restored "
        "punctuation and capitalization. Labels are saved in format described here "
        "https://docs.nvidia.com/deeplearning/nemo/"
        "user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=['cpu', 'cuda'],
        help="Which device to use. If device is not set and CUDA is available, then GPU will be used. If device is "
        "not set and CUDA is not available, then CPU is used.",
    )
    args = parser.parse_args()
    if args.input_manifest is None and args.output_manifest is not None:
        parser.error("--output_manifest requires --input_manifest")
    for name in ["input_manifest", "input_text", "output_manifest", "output_text", "model_path"]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser())
    return args


def main():
    args = get_args()



if __name__ == "__main__":
    main()
