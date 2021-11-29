import argparse
import io
import os
import tempfile
from pathlib import Path
from subprocess import run


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
    args = parser.parse_args()
    for i, f in enumerate(args.input_files):
        args.input_files[i] = f.expanduser()
    for i, f in enumerate(args.output_files):
        args.output_files[i] = f.expanduser()
    if len(args.input_files) != len(args.output_files):
        parser.error("Number of elements in parameters `--input_files` and `--output_file` has to be equal")
    return args


def main():
    args = get_args()
    united_fd, united_path = tempfile.mkstemp(text=True)
    input_file_objects = [inp_file.open() for inp_file in args.input_files]
    lines = []
    for i, inp_obj in enumerate(input_file_objects):
        lines[i] = inp_obj.readline().strip('\n')
    line_number = 0
    while all(lines):
        for i, inp_obj in enumerate(input_file_objects):
            lines[i] = inp_obj.readline()
        delimiter_in_line = [args.line_delimiter in line for line in lines]
        if any(delimiter_in_line):
            raise ValueError(
                f"Line delimiter {repr(args.line_delimiter)} is present in line number {line_number} in file "
                f"{args.input_files[delimiter_in_line.index(True)]}."
            )
        os.write(united_fd, args.line_delimiter.join(lines) + '\n')
    if any(lines):
        raise ValueError(
            f"Files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if not line])} "
            f"before files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if line])}."
        )
    for input_file_object in input_file_objects:
        input_file_object.close()
    os.close(united_fd)
    shuffled_fd, shuffled_path = tempfile.mkstemp(text=True)
    shuffled_f = io.FileIO(shuffled_fd, mode='w', closefd=False)
    run(['shuf', str(united_path)], stdout=shuffled_f)
    os.remove(united_path)
    shuffled_f.close()
    output_file_objects = [out_file.open('w') for out_file in args.output_files]
    f = io.FileIO(shuffled_fd)
    for tmp_line in f:
        lines = tmp_line.strip().split(args.line_delimiter)
        assert len(lines) == len(output_file_objects)
        for i, line in enumerate(lines):
            output_file_objects[i].write(line + '\n')
    f.close()
    for output_file_object in output_file_objects:
        output_file_object.close()


if __name__ == "__main__":
    main()