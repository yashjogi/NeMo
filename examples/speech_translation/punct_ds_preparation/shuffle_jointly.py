import argparse
import os
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


def main():
    args = get_args()
    input_file_objects = [inp_file.open() for inp_file in args.input_files]
    united_file_path = args.input_files[0].parent / args.united_file_name
    lines = []
    for i, inp_obj in enumerate(input_file_objects):
        lines.append(inp_obj.readline().strip('\n'))
    line_number = 0
    with united_file_path.open('w') as united_f:
        while all(lines):
            for i, inp_obj in enumerate(input_file_objects):
                lines[i] = inp_obj.readline()
            delimiter_in_line = [args.line_delimiter in line for line in lines]
            if any(delimiter_in_line):
                raise ValueError(
                    f"Line delimiter {repr(args.line_delimiter)} is present in line number {line_number} in file "
                    f"{args.input_files[delimiter_in_line.index(True)]}."
                )
            united_f.write(args.line_delimiter.join(lines) + '\n')
    if any(lines):
        raise ValueError(
            f"Files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if not line])} "
            f"before files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if line])}."
        )
    for input_file_object in input_file_objects:
        input_file_object.close()
    shuffled_file_path = args.input_files[0].parent / args.shuffled_file_name
    with shuffled_file_path.open('w') as f:
        run(['shuf', str(united_file_path)], stdout=f)
    os.remove(united_file_path)
    for out_file in args.output_files:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    output_file_objects = [out_file.open('w') for out_file in args.output_files]
    with shuffled_file_path.open() as f:
        for tmp_line in f:
            lines = tmp_line.strip().split(args.line_delimiter)
            assert len(lines) == len(output_file_objects)
            for i, line in enumerate(lines):
                output_file_objects[i].write(line + '\n')
    for output_file_object in output_file_objects:
        output_file_object.close()
    os.remove(shuffled_file_path)


if __name__ == "__main__":
    main()