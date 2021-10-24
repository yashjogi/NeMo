import argparse
import math
import multiprocessing as mp
import re
from pathlib import Path

import webdataset as wds
from joblib import Parallel, delayed
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset
)
from nemo.utils import logging


TAR_FRAGMENT_TMPL = "fragment{}.{}.tar"
TAR_FRAGMENT_PATTERN = re.compile("fragment(?:0|[1-9][0-9]*).(?:0|[1-9][0-9]*).tar$")

NUMBER_RE = "(?:0|[1-9][0-9]*)"
TAR_FINAL_TMPL = (
    "{prefix}.batches.tokens{tokens_in_batch}.max_seq_length{max_seq_length}.{tokenizer}.{ignore_start_end}."
    "{ignore_extra_tokens}.{{ctr}}.tar"
)
TAR_FINAL_RE = TAR_FINAL_TMPL.replace('{ctr}', NUMBER_RE)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--text", "-t", type=Path, required=True)
    parser.add_argument("--labels", "-L", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--max_seq_length", "-s", type=int, default=512)
    parser.add_argument("--tokens_in_batch", "-b", type=int, default=15000)
    parser.add_argument("--lines_per_dataset_fragment", type=int, default=10 ** 6)
    parser.add_argument("--num_batches_per_tarfile", type=int, default=1000)
    parser.add_argument("--tokenizer", "-T", default="bert-base-uncased")
    parser.add_argument("--ignore_start_end", "-S", action="store_true")
    parser.add_argument("--ignore_extra_tokens", "-e", action="store_true")
    parser.add_argument("--tar_file_prefix", "-p", default="punctuation_capitalization")
    parser.add_argument("--n_jobs", "-j", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    args.text = args.text.expanduser()
    args.labels = args.labels.expanduser()
    args.output_dir = args.output_dir.expanduser()
    return args


def count_lines_and_get_fragment_starting_positions(file_name, lines_per_dataset_fragment):
    pos = [0]
    with file_name.open() as f:
        i = 0
        line = f.readline()
        while line:
            i += 1
            if i % lines_per_dataset_fragment == 0:
                pos.append(f.tell())
            line = f.readline()
    return i, pos[:-1] if i % lines_per_dataset_fragment == 0 else pos


def process_fragment(
    text_file,
    labels_file,
    output_dir,
    text_start_pos,
    label_start_pos,
    lines_per_dataset_fragment,
    max_seq_length,
    tokens_in_batch,
    num_batches_per_tarfile,
    tokenizer,
    ignore_start_end,
    ignore_extra_tokens,
    fragment_idx,
):
    tokenizer = get_tokenizer(tokenizer)
    tmp_text = output_dir / f'tmp_text_{fragment_idx}.txt'
    tmp_labels = output_dir / f'tmp_labels_{fragment_idx}.txt'
    with text_file.open() as tf, labels_file.open() as lf, tmp_text.open('w') as otf, tmp_labels.open('w') as olf:
        tf.seek(text_start_pos)
        lf.seek(label_start_pos)
        for _ in range(lines_per_dataset_fragment):
            otf.write(tf.readline())
            olf.write(lf.readline())
    dataset = BertPunctuationCapitalizationDataset(
        tmp_text,
        tmp_labels,
        max_seq_length,
        tokenizer,
        tokens_in_batch=tokens_in_batch,
        njobs=0,
        use_cache=False,
        ignore_start_end=ignore_start_end,
        ignore_extra_tokens=ignore_extra_tokens,
        add_masks_and_segment_ids_to_batch=False,
        verbose=False,
    )
    tmp_text.unlink()
    tmp_labels.unlink()
    tar_ctr = 0
    sink = wds.TarWriter(str(output_dir / TAR_FRAGMENT_TMPL.format(fragment_idx, tar_ctr)))
    for batch_i, batch in enumerate(dataset):
        if batch_i % num_batches_per_tarfile == 0 and batch_i > 0:
            sink.close()
            tar_ctr += 1
            sink = wds.TarWriter(str(output_dir / TAR_FRAGMENT_TMPL.format(fragment_idx, tar_ctr)))
        sink.write(
            {
                "__key__": f"fragment-{fragment_idx}-batch-{batch_i}",
                "batch.pyd": batch,
            }
        )
    sink.close()


def remove_unexpected_tar_files(output_dir, output_file_tmpl):
    if not output_dir.is_dir():
        return
    unexpected_tar_files = [path for path in output_dir.iterdir() if TAR_FRAGMENT_PATTERN.match(path.name)]
    if unexpected_tar_files:
        logging.warning(
            f"Found {len(unexpected_tar_files)} unexpected fragment files in the output directory {output_dir}. "
            f"All of them are going to be removed. First 3 files: {unexpected_tar_files[:3]}"
        )
        for fn in unexpected_tar_files:
            fn.unlink()
    tar_final_pattern = re.compile(output_file_tmpl.format(ctr=NUMBER_RE))
    unexpected_tar_files = [path for path in output_dir.iterdir() if tar_final_pattern.match(path.name)]
    if unexpected_tar_files:
        logging.warning(
            f"Found {len(unexpected_tar_files)} unexpected final tar files matching pattern "
            f"{tar_final_pattern.pattern}. All of them are going to be removed. First 3 files: "
            f"{unexpected_tar_files[:3]}"
        )
        for fn in unexpected_tar_files:
            fn.unlink()


def create_tarred_dataset(
    text_file,
    label_file,
    output_dir,
    max_seq_length,
    tokens_in_batch,
    lines_per_dataset_fragment,
    num_batches_per_tarfile,
    tokenizer,
    ignore_start_end,
    ignore_extra_tokens,
    tar_file_prefix,
    n_jobs,
):
    output_file_tmpl = TAR_FINAL_TMPL.format(
        prefix=tar_file_prefix,
        tokens_in_batch=tokens_in_batch,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        ignore_start_end=ignore_start_end,
        ignore_extra_tokens=ignore_extra_tokens,
    )
    remove_unexpected_tar_files(output_dir, output_file_tmpl)
    result = Parallel(n_jobs=2)(
        delayed(count_lines_and_get_fragment_starting_positions)(file_name, lines_per_dataset_fragment)
        for file_name in [text_file, label_file]
    )
    if result[0][0] != result[1][0]:
        raise ValueError(
            f"Text file {text_file} and label file {label_file} contain different number of lines. Number of lines "
            f"in text file: {result[0][0]}, number of lines in label file: {result[1][0]}."
        )
    text_start_bytes, label_start_bytes = result[0][1], result[1][1]
    assert len(text_start_bytes) == len(label_start_bytes)
    if text_start_bytes:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning(f"Both {label_file} and {text_file} are empty. Tarred dataset cannot be created.")
        return
    Parallel(n_jobs=min(n_jobs, len(text_start_bytes)))(
        delayed(process_fragment)(
            text_file,
            label_file,
            output_dir,
            text_start_pos,
            label_start_pos,
            lines_per_dataset_fragment,
            max_seq_length,
            tokens_in_batch,
            num_batches_per_tarfile,
            tokenizer,
            ignore_start_end,
            ignore_extra_tokens,
            fragment_idx,
        ) for fragment_idx, (text_start_pos, label_start_pos) in enumerate(zip(text_start_bytes, label_start_bytes))
    )
    for i, fn in enumerate([fn for fn in output_dir.iterdir() if TAR_FRAGMENT_PATTERN.match(fn.name)]):
        fn.rename(output_dir / output_file_tmpl.format(i))


def main():
    args = get_args()
    create_tarred_dataset(
        args.text,
        args.labels,
        args.output_dir,
        args.max_seq_length,
        args.tokens_in_batch,
        args.lines_per_dataset_fragment,
        args.num_batches_per_tarfile,
        args.tokenizer,
        args.ignore_start_end,
        args.ignore_extra_tokens,
        args.tar_file_prefix,
        args.n_jobs,
    )


if __name__ == "__main__":
    main()
