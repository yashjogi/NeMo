import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path

import webdataset as wds
from joblib import Parallel, delayed

from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset, Progress
)
from nemo.utils import logging


NUMBER_RE = "(0|[1-9][0-9]*)"
TAR_FRAGMENT_TMPL_1 = "fragment{}.{}.tar"
TAR_FRAGMENT_TMPL_2 = "fragment{}.num_batches{}.{}.tar"
TAR_FRAGMENT_PATTERN_1 = re.compile(f"fragment{NUMBER_RE}.{NUMBER_RE}.tar$")
TAR_FRAGMENT_PATTERN_2 = re.compile(f"fragment{NUMBER_RE}.num_batches{NUMBER_RE}.{NUMBER_RE}.tar$")
EXTRACT_NUM_BATCHES_PATTERN = re.compile(r"fragment\d+.num_batches(\d+).\d+.tar")

DATASET_PARAMETERS_TMPL = (
    "{prefix}.tokens{tokens_in_batch}.max_seq_length{max_seq_length}.{tokenizer}.{ignore_start_end}."
    "{ignore_extra_tokens}"
)
TAR_FINAL_TMPL = ".batches{num_batches}.{ctr}.tar"

WRITING_DATASET_PROGRESS_REPORT_PERIOD = 10 ** 4


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
    tokenization_progress_queue,
    batch_mark_up_progress_queue,
    batch_building_progress_queue,
    writing_to_tar_progress_queue,
):
    tokenizer = get_tokenizer(tokenizer)
    tmp_text = output_dir / f'tmp_text_{fragment_idx}.txt'
    tmp_labels = output_dir / f'tmp_labels_{fragment_idx}.txt'
    with text_file.open() as tf, labels_file.open() as lf, tmp_text.open('w') as otf, tmp_labels.open('w') as olf:
        tf.seek(text_start_pos)
        lf.seek(label_start_pos)
        for _ in range(lines_per_dataset_fragment):
            text_line = tf.readline()
            if not text_line:
                break
            otf.write(text_line)
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
        tokenization_progress_queue=tokenization_progress_queue,
        batch_mark_up_progress_queue=batch_mark_up_progress_queue,
        batch_building_progress_queue=batch_building_progress_queue,
    )
    tmp_text.unlink()
    tmp_labels.unlink()
    tar_ctr = 0
    current_file_name = output_dir / TAR_FRAGMENT_TMPL_1.format(fragment_idx, tar_ctr)
    current_num_batches = 0
    sink = wds.TarWriter(str(current_file_name))
    progress_made = 0
    for batch_i, batch in enumerate(dataset):
        if batch_i % num_batches_per_tarfile == 0 and batch_i > 0:
            sink.close()
            current_file_name.rename(
                output_dir / TAR_FRAGMENT_TMPL_2.format(fragment_idx, current_num_batches, tar_ctr)
            )
            writing_to_tar_progress_queue.put(progress_made)
            progress_made = 0
            tar_ctr += 1
            current_file_name = output_dir / TAR_FRAGMENT_TMPL_1.format(fragment_idx, tar_ctr)
            current_num_batches = 0
            sink = wds.TarWriter(str(current_file_name))
        sink.write({"__key__": f"fragment-{fragment_idx}-batch-{batch_i}", "batch.pyd": batch})
        current_num_batches += 1
        progress_made += len(batch['input_ids'])
    sink.close()
    writing_to_tar_progress_queue.put(progress_made)
    current_file_name.rename(output_dir / TAR_FRAGMENT_TMPL_2.format(fragment_idx, current_num_batches, tar_ctr))


def remove_unexpected_files(output_dir, output_file_tmpl, metadata_file_name):
    if not output_dir.is_dir():
        return
    tar_final_pattern = re.compile(output_file_tmpl.format(ctr=NUMBER_RE, num_batches=NUMBER_RE))
    unexpected_tar_files = [
        path for path in output_dir.iterdir()
        if any(
            [
                p.match(path.name) is not None
                for p in [TAR_FRAGMENT_PATTERN_1, TAR_FRAGMENT_PATTERN_2, tar_final_pattern]
            ]
        )
    ]
    if unexpected_tar_files:
        logging.warning(
            f"Found {len(unexpected_tar_files)} unexpected tar files in the output directory {output_dir}. "
            f"All of them are going to be removed. The files match one of 3 patterns: "
            f"'{TAR_FRAGMENT_PATTERN_1.pattern}', '{TAR_FRAGMENT_PATTERN_2.pattern}', "
            f"'{tar_final_pattern.pattern}'. The first 3 unexpected files: {unexpected_tar_files[:3]}"
        )
        for fn in unexpected_tar_files:
            fn.unlink()
    if metadata_file_name.is_file():
        logging.warning(f"Found metadata file {metadata_file_name}. It is going to be removed.")
        metadata_file_name.unlink()


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
    ds_params_str = DATASET_PARAMETERS_TMPL.format(
        prefix=tar_file_prefix,
        tokens_in_batch=tokens_in_batch,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        ignore_start_end=ignore_start_end,
        ignore_extra_tokens=ignore_extra_tokens,
    )
    output_file_tmpl = ds_params_str + TAR_FINAL_TMPL
    metadata_file_name = output_dir / ('metadata.' + ds_params_str + '.json')
    remove_unexpected_files(output_dir, output_file_tmpl, metadata_file_name)
    logging.info(
        f"Counting lines in files {text_file} and {label_file} and creating segment borders. This may take "
        f"considerable time. 86GB, 1.27b lines file was processed in 7 minutes."
    )
    result = Parallel(n_jobs=2)(
        delayed(count_lines_and_get_fragment_starting_positions)(file_name, lines_per_dataset_fragment)
        for file_name in [text_file, label_file]
    )
    if result[0][0] != result[1][0]:
        raise ValueError(
            f"Text file {text_file} and label file {label_file} contain different number of lines. Number of lines "
            f"in text file: {result[0][0]}, number of lines in label file: {result[1][0]}."
        )
    num_lines = result[0][0]
    text_start_bytes, label_start_bytes = result[0][1], result[1][1]
    assert len(text_start_bytes) == len(label_start_bytes)
    if text_start_bytes:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning(f"Both {label_file} and {text_file} are empty. Tarred dataset cannot be created.")
        return
    with Progress(num_lines, "Tokenization", "query") as tok_queue, \
            Progress(num_lines, "Batch mark up", "query") as mark_up_queue, \
            Progress(num_lines, "Batch building", "query") as building_queue, \
            Progress(num_lines, "Writing tarred dataset", "query") as writing_queue:
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
                tok_queue,
                mark_up_queue,
                building_queue,
                writing_queue,
            ) for fragment_idx, (text_start_pos, label_start_pos) in enumerate(zip(text_start_bytes, label_start_bytes))
        )
    metadata = {"num_batches": 0, "tar_files": []}
    for i, fn in enumerate([fn for fn in output_dir.iterdir() if TAR_FRAGMENT_PATTERN_2.match(fn.name)]):
        nb = int(EXTRACT_NUM_BATCHES_PATTERN.match(fn.name).group(1))
        new_name = output_dir / output_file_tmpl.format(ctr=i, num_batches=nb)
        fn.rename(new_name)
        metadata['tar_files'].append(new_name.name)
        metadata["num_batches"] += nb
    with metadata_file_name.open('w') as f:
        json.dump(metadata, f, indent=2)


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
