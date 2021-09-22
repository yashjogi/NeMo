import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from nemo.collections.asr.metrics.wer import word_error_rate


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
    args = parser.parse_args()
    args.hyp = args.hyp.expanduser()
    args.ref = args.ref.expanduser()
    args.output = args.output.expanduser()
    args.punctuation_file = args.punctuation_file.expanduser()
    return args


def read_lines(path, capitalization_labels, include_leading_punctuation_in_metrics):
    rstrip_re = re.compile(f"^[^{capitalization_labels}]+")
    lstrip_re = re.compile(f"[^{capitalization_labels}]*$")
    capitalization_re = re.compile(f'[{capitalization_labels}]', flags=re.I)
    punctuation, capitalization, lines = [], [], []
    with path.open() as f:
        for line in f:
            if include_leading_punctuation_in_metrics:
                if line[0] in capitalization_labels:
                    line = ' ' + line
            else:
                line = rstrip_re.sub('', line)
            if ' ' not in lstrip_re.search(line).group(0):
                line += ' '
            lines.append(line)
            capitalization.append(capitalization_re.findall(line))
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
        args.hyp, args.capitalization_labels, args.include_leading_punctuation_in_metrics
    )
    ref_punctuation, ref_capitalization, ref_lines = read_lines(
        args.ref, args.capitalization_labels, args.include_leading_punctuation_in_metrics
    )
    cer = word_error_rate(hyp_lines, ref_lines, use_cer=True)
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
        'punctuation': {
            "F1 micro average": f1_score(
                np.concatenate(ref_punctuation_ids), np.concatenate(hyp_punctuation_ids), average='micro'
            ),
        },
        'capitalization': {
            "f1_micro": f1_score(
                np.concatenate(ref_capitalization_ids), np.concatenate(hyp_capitalization_ids), average='micro'
            ),
        },
    }
    punctuation_ids_present_in_ref = set.union(*ref_punctuation_ids)
    capitalization_ids_present_in_ref = set.union(*ref_capitalization_ids)
    for name, metric in [('precision', precision_score), ('recall', recall_score),  ('F1', f1_score)]:
        result['capitalization'][name] = {
            lbl: metric(ref_capitalization_ids, hyp_capitalization_ids, pos_label=id_)
            for lbl, id_ in capitalization_labels_to_ids.items() if id_ > 0 and id_ in capitalization_ids_present_in_ref
        }
        result['punctuation'][name] = {
            lbl: metric(ref_punctuation_ids, hyp_punctuation_ids, pos_label=id_)
            for lbl, id_ in punctuation_labels_to_ids.items() if id_ > 1 and id_ in punctuation_ids_present_in_ref
        }
    with args.output.open('w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
