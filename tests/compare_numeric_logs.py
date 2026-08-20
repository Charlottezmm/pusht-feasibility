import argparse
import math
import re
from pathlib import Path


ABS_TOL = 1e-12
REL_TOL = 0.0
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_line(line):
    matches = list(NUMBER_PATTERN.finditer(line))
    structure = NUMBER_PATTERN.sub("<NUMBER>", line)
    values = [float(match.group(0)) for match in matches]
    return structure, values


def compare_logs(path_a, path_b):
    lines_a = path_a.read_text(encoding="utf-8").splitlines()
    lines_b = path_b.read_text(encoding="utf-8").splitlines()

    if len(lines_a) != len(lines_b):
        raise ValueError(
            f"line count differs: {len(lines_a)} != {len(lines_b)}"
        )

    max_abs_diff = 0.0

    for line_number, (line_a, line_b) in enumerate(
        zip(lines_a, lines_b), start=1
    ):
        structure_a, values_a = parse_line(line_a)
        structure_b, values_b = parse_line(line_b)

        if structure_a != structure_b or len(values_a) != len(values_b):
            raise ValueError(
                f"line {line_number}: non-numeric structure differs"
            )

        for number_index, (value_a, value_b) in enumerate(
            zip(values_a, values_b), start=1
        ):
            abs_diff = abs(value_a - value_b)
            max_abs_diff = max(max_abs_diff, abs_diff)

            if not math.isclose(
                value_a,
                value_b,
                rel_tol=REL_TOL,
                abs_tol=ABS_TOL,
            ):
                raise ValueError(
                    f"line {line_number}, number {number_index}: "
                    f"{value_a} != {value_b}, abs_diff={abs_diff}"
                )

    return len(lines_a), max_abs_diff


def main():
    parser = argparse.ArgumentParser(
        description="Compare two PushT logs with a fixed numeric tolerance."
    )
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args()

    try:
        line_count, max_abs_diff = compare_logs(args.run_a, args.run_b)
    except (OSError, ValueError) as error:
        print("FAIL", error)
        raise SystemExit(1)

    print(
        f"PASS lines={line_count} "
        f"max_abs_diff={max_abs_diff:.3e} "
        f"atol={ABS_TOL:.1e} rtol={REL_TOL}"
    )


if __name__ == "__main__":
    main()
