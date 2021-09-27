import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging


UNK_LBL = "<UNK>"


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hyp", "-H", help="Path to file with predicted hypotheses.", type=Path, required=True)
    parser.add_argument("--ref", "-r", help="Path to file with ground truth references.", type=Path, required=True)
    parser.add_argument(
        "--output", "-o", help="Path to the file where results will be saved.", type=Path, required=True
    )
    parser.add_argument("--capitalization_labels", "-c", help="A string with capitalization labels.", default="uUO")
    parser.add_argument(
        "--include_leading_punctuation_in_metrics",
        "-L",
        help="If not set leading punctuation is removed both from hypotheses and references before metrics "
        "computation. If this option is chosen lines without leading punctuation is considered to be preceded with "
        "space character.",
        action="store_true",
    )
    parser.add_argument(
        '--punctuation_file',
        '-p',
        help="Path to a JSON file with punctuation. The file should contain all punctuation marks counts. Create the "
        "file using 'examples/nlp/machine_translation/punctuation_infer/create_punctuation_file.py' script.",
        type=Path,
    )
    parser.add_argument(
        "--evelina_data_format",
        "-e",
        help="Hypotheses and references are in format, described in https://docs.nvidia.com/deeplearning/nemo/"
        "user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format",
        action="store_true",
    )
    args = parser.parse_args()
    args.hyp = args.hyp.expanduser()
    args.ref = args.ref.expanduser()
    args.output = args.output.expanduser()
    if args.punctuation_file is not None:
        args.punctuation_file = args.punctuation_file.expanduser()
    return args


def transform_to_autoregressive_format(line, line_i):
    pairs = line.split()
    result = ""
    for pair in pairs:
        if len(pair) != 2:
            logging.warning(f"Pair '{pair}' in line {line_i} '{line}' contains wrong number of characters.")
        result += pair[1] if len(pair) > 1 else 'O'
        if pair[0] != 'O':
            result += pair[0]
        result += ' '
    return result


def read_lines(path, capitalization_labels, include_leading_punctuation_in_metrics, evelina_data_format):
    lstrip_re = re.compile(f"^[^{capitalization_labels}]+")
    rstrip_re = re.compile(f"[^{capitalization_labels}]*$")
    capitalization_re = re.compile(f'[{capitalization_labels}]', flags=re.I)
    punctuation, capitalization, lines = [], [], []
    with path.open() as f:
        for i, line in enumerate(f):
            print("before:", line)
            if evelina_data_format:
                line = transform_to_autoregressive_format(line, i)
            print("after:", line)
            if include_leading_punctuation_in_metrics:
                if line[0] in capitalization_labels:
                    line = ' ' + line
            else:
                line = lstrip_re.sub('', line)
            if ' ' not in rstrip_re.search(line).group(0):
                line += ' '
            lines.append(line)
            capitalization.append(capitalization_re.findall(line))
            print("punctuation:", capitalization_re.split(line))
            punctuation.append(capitalization_re.split(line))
    return punctuation, capitalization, lines


def encode(labels, ids):
    return [np.array([ids[lbl] if lbl in ids else ids[UNK_LBL] for lbl in label_line]) for label_line in labels]


def pad_or_clip_hyp(hyp, ref, pad_id):
    for i, (h, r) in enumerate(zip(hyp, ref)):
        if h.shape[0] > r.shape[0]:
            hyp[i] = h[:r.shape[0]]
        elif h.shape[0] < r.shape[0]:
            hyp[i] = np.concatenate([h, np.full([r.shape[0] - h.shape[0]], pad_id)])


def main():
    args = get_args()
    hyp_punctuation, hyp_capitalization, hyp_lines = read_lines(
        args.hyp, args.capitalization_labels, args.include_leading_punctuation_in_metrics, args.evelina_data_format
    )
    ref_punctuation, ref_capitalization, ref_lines = read_lines(
        args.ref, args.capitalization_labels, args.include_leading_punctuation_in_metrics, args.evelina_data_format
    )
    cer = word_error_rate(hyp_lines, ref_lines, use_cer=True)
    if args.punctuation_file is None:
        punctuation_labels = sorted(set(itertools.chain(*ref_punctuation)), key=lambda x: ref_punctuation.count(x))
    else:
        with args.punctuation_file.open() as f:
            punctuation_labels = list(json.load(f).keys())
    capitalization_labels_to_ids = {'O': 0, 'U': 1, 'u': 2}
    punctuation_labels_to_ids = {' ': 0, UNK_LBL: 1}
    count = 2
    for lbl in punctuation_labels:
        if lbl not in punctuation_labels_to_ids:
            punctuation_labels_to_ids[lbl] = count
            count += 1
    hyp_punctuation_ids = encode(hyp_punctuation, punctuation_labels_to_ids)
    ref_punctuation_ids = encode(ref_punctuation, punctuation_labels_to_ids)
    hyp_capitalization_ids = encode(hyp_capitalization, capitalization_labels_to_ids)
    ref_capitalization_ids = encode(ref_capitalization, capitalization_labels_to_ids)
    pad_or_clip_hyp(hyp_punctuation_ids, ref_punctuation_ids, 0)
    pad_or_clip_hyp(hyp_capitalization_ids, ref_capitalization_ids, 0)
    result = {
        'CER': cer,
        "accuracy_punctuation": accuracy_score(
            np.concatenate(ref_punctuation_ids), np.concatenate(hyp_punctuation_ids)
        ),
        "accuracy_capitalization": accuracy_score(
            np.concatenate(ref_capitalization_ids), np.concatenate(hyp_capitalization_ids)
        ),
        'punctuation': {
            "F1 macro average": f1_score(
                np.concatenate(ref_punctuation_ids), np.concatenate(hyp_punctuation_ids), average='macro'
            ),
        },
        'capitalization': {
            "F1 macro average": f1_score(
                np.concatenate(ref_capitalization_ids), np.concatenate(hyp_capitalization_ids), average='macro'
            ),
        },
    }
    punctuation_ids_present_in_ref = set()
    punctuation_ids_present_in_ref.update(*ref_punctuation_ids)
    capitalization_ids_present_in_ref = set()
    capitalization_ids_present_in_ref.update(*ref_capitalization_ids)
    for name, metric in [('precision', precision_score), ('recall', recall_score),  ('F1', f1_score)]:
        result['capitalization'][name] = {
            lbl: metric(
                np.concatenate(ref_capitalization_ids),
                np.concatenate(hyp_capitalization_ids),
                labels=[id_],
                average="micro",
                zero_division=0,
            )
            for lbl, id_ in capitalization_labels_to_ids.items() if id_ > 0 and id_ in capitalization_ids_present_in_ref
        }
        result['punctuation'][name] = {
            lbl: metric(
                np.concatenate(ref_punctuation_ids),
                np.concatenate(hyp_punctuation_ids),
                labels=[id_],
                average="micro",
                zero_division=0,
            )
            for lbl, id_ in punctuation_labels_to_ids.items() if id_ > 1 and id_ in punctuation_ids_present_in_ref
        }
    with args.output.open('w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
