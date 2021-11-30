import argparse
import logging
import os
from pathlib import Path
from subprocess import run, PIPE

from tqdm import tqdm


logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')

BUFFER_SIZE = 2 ** 25


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input_files",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output_files",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--line_delimiter",
        required=True,
        help="It has to a character which does not occur in any of input files."
    )
    parser.add_argument(
        "--united_file_name",
        default="united_lines.txt",
    )
    parser.add_argument(
        "--shuffled_file_name",
        default="shuffled_lines.txt",
    )
    args = parser.parse_args()
    for i, f in enumerate(args.input_files):
        args.input_files[i] = f.expanduser()
    for i, f in enumerate(args.output_files):
        args.output_files[i] = f.expanduser()
    if len(args.input_files) != len(args.output_files):
        parser.error("Number of elements in parameters `--input_files` and `--output_file` has to be equal")
    return args


def get_num_lines(input_file):
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    return int(result.stdout.decode('utf-8').split()[0])


def main():
    args = get_args()
    input_file_objects = [inp_file.open(buffering=BUFFER_SIZE) for inp_file in args.input_files]
    united_file_path = args.input_files[0].parent / args.united_file_name
    lines = [inp_obj.readline().strip('\n') for inp_obj in input_file_objects]
    line_number = 0
    num_lines = get_num_lines(args.input_files[0])
    progress_bar = tqdm(total=num_lines, unit='line', desc="Uniting files", unit_scale=True)
    with united_file_path.open('w', buffering=BUFFER_SIZE) as united_f:
        while all(lines):
            delimiter_in_line = [args.line_delimiter in line for line in lines]
            if any(delimiter_in_line):
                raise ValueError(
                    f"Line delimiter {repr(args.line_delimiter)} is present in line number {line_number} in file "
                    f"{args.input_files[delimiter_in_line.index(True)]}."
                )
            united_f.write(args.line_delimiter.join(lines) + '\n')
            progress_bar.n += 1
            progress_bar.update(0)
            lines = [inp_obj.readline().strip('\n') for inp_obj in input_file_objects]
    progress_bar.close()
    if any(lines):
        raise ValueError(
            f"Files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if not line])} "
            f"before files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if line])}."
        )
    for input_file_object in input_file_objects:
        input_file_object.close()
    shuffled_file_path = args.input_files[0].parent / args.shuffled_file_name
    logging.info(f"Shuffling: shuf {united_file_path} > {shuffled_file_path}")
    with shuffled_file_path.open('w') as f:
        run(['shuf', str(united_file_path)], stdout=f)
    os.remove(united_file_path)
    for out_file in args.output_files:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    output_file_objects = [out_file.open('w', buffering=BUFFER_SIZE) for out_file in args.output_files]
    with shuffled_file_path.open(buffering=BUFFER_SIZE) as f:
        for line_i, tmp_line in tqdm(enumerate(f), total=num_lines, unit='line', desc="spliting lines"):
            lines = tmp_line.strip().split(args.line_delimiter)
            assert len(lines) == len(output_file_objects), (
                f"Number of lines {len(lines)} in shuffled file {shuffled_file_path }does not equal number of output"
                f"file objects {output_file_objects}. Line from shuffled file: {repr(tmp_line)}"
            )
            for i, line in enumerate(lines):
                output_file_objects[i].write(line + '\n')
    for output_file_object in output_file_objects:
        output_file_object.close()
    os.remove(shuffled_file_path)


if __name__ == "__main__":
    main()