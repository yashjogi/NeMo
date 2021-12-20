# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from nemo_text_processing.text_normalization.data_loader_utils import (
    load_file, load_manifest, write_file, write_manifest
)
from nemo_text_processing.text_normalization.normalize import Normalizer


'''
Runs normalization prediction on text data
'''


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    input_ = parser.add_mutually_exclusive_group(required=True)
    input_.add_argument(
        "--input", help="Path to input text file. Mutually exclusive with `--input_manifest` parameter."
    )
    input_.add_argument(
        "--input_manifest",
        help="Path to manifest file in NeMo format"
        "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/datasets.html"
        "?highlight=manifest#all-other-datasets. Mutually exclusive with `--input` parameter. Texts for processing "
        "are passed in the manifest under `--manifest_text_key`.",
    )
    parser.add_argument("--language", help="language", choices=['en'], default="en", type=str)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", help="Path to output text file. Mutually exclusive with `--output_manifest`.")
    output.add_argument(
        "--output_manifest",
        help="Path to output manifest file in NeMo format"
        "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/datasets.html"
        "?highlight=manifest#all-other-datasets. Mutually exclusive with `--output` parameter. To save results "
        "in `--output_manifest` you need to pass input text in `--input_manifest`.",
    )
    parser.add_argument(
        "--manifest_text_key",
        help="The key in ASR manifest which contains text for processing. If processed text is saved in "
        "`--output_manifest`, then it is saved under `--manifest_text_key`.",
        default="text",
    )
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="cased", type=str
    )
    parser.add_argument("--verbose", help="print meta info for debugging", action='store_true')
    parsed_args = parser.parse_args()
    if parsed_args.output_manifest is not None and parsed_args.input_manifest is None:
        parser.error(
            "If you provide `--output_manifest` argument you should also provide `--input_manifest` argument."
        )
    return parsed_args


if __name__ == "__main__":
    args = parse_args()
    normalizer = Normalizer(input_case=args.input_case, lang=args.language)

    if args.input is None:
        print("Loading data:", args.input_manifest)
        manifest_items = load_manifest(args.input_manifest)
        data = [item[args.manifest_text_key] for item in manifest_items]
    else:
        print("Loading data: " + args.input)
        data = load_file(args.input)

    print("- Data: " + str(len(data)) + " sentences")
    normalizer_prediction = normalizer.normalize_list(data, verbose=args.verbose)
    if args.output is None:
        for item, processed_text in zip(manifest_items, normalizer_prediction):
            item[args.manifest_text_key] = processed_text
        write_manifest(manifest_items, args.output_manifest)
        print(f"- Normalized. Writing out to {args.output_manifest}")
    else:
        write_file(args.output, normalizer_prediction)
        print(f"- Normalized. Writing out to {args.output}")
