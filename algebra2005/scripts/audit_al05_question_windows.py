from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import pandas as pd


def install_pykt_preprocess_path(pykt_root: Path) -> None:
    for name, relative in (
        ("pykt", Path("pykt")),
        ("pykt.preprocess", Path("pykt") / "preprocess"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(pykt_root / relative)]
        package.__package__ = name
        sys.modules[name] = package


def shift_sequence(values: str, offset: int) -> str:
    return ",".join(
        str(value + offset if value >= 0 else value)
        for value in (int(token) for token in values.split(","))
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pykt-root", type=Path, required=True)
    parser.add_argument("--pykt-data-dir", type=Path, required=True)
    args = parser.parse_args()

    pykt_root = args.pykt_root.resolve()
    data_dir = args.pykt_data_dir.resolve()
    install_pykt_preprocess_path(pykt_root)
    from pykt.preprocess.split_datasets import generate_question_sequences

    test_frame = pd.read_csv(data_dir / "test.csv", dtype=str)
    effective_keys = set(test_frame.columns) - {"cidxs"}
    indices = sorted({0, len(test_frame) // 2, len(test_frame) - 1})
    offsets: dict[int, int] = {}
    question_offset = 0
    for index, row in test_frame.iterrows():
        if index in indices:
            offsets[index] = question_offset
        question_offset += row["is_repeat"].split(",").count("0")

    expected: dict[int, list[dict[str, str]]] = {}
    for index in indices:
        _, generated = generate_question_sequences(
            test_frame.iloc[[index]].copy(),
            effective_keys,
            window=True,
            min_seq_len=3,
            maxlen=200,
        )
        for column in ("qidxs", "orirow"):
            generated[column] = generated[column].map(
                lambda value, offset=offsets[index] if column == "qidxs" else index:
                shift_sequence(value, offset)
            )
        expected[index] = generated.to_dict(orient="records")

    actual: dict[int, list[dict[str, str]]] = {index: [] for index in indices}
    question_window_path = data_dir / "test_question_window_sequences.csv"
    with question_window_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns:
            raise ValueError("Question-window CSV has no header")
        for row in reader:
            first_orirow = int(row["orirow"].split(",", 1)[0])
            if first_orirow in actual:
                actual[first_orirow].append(row)

    comparisons = {}
    for index in indices:
        expected_rows = expected[index]
        actual_rows = actual[index]
        if len(expected_rows) != len(actual_rows):
            raise AssertionError(
                f"User row {index} window count differs: "
                f"{len(expected_rows)} != {len(actual_rows)}"
            )
        for window_index, (expected_row, actual_row) in enumerate(
            zip(expected_rows, actual_rows, strict=True)
        ):
            mismatches = [
                column
                for column in columns
                if str(expected_row[column]) != str(actual_row[column])
            ]
            if mismatches:
                raise AssertionError(
                    f"User row {index}, window {window_index} differs in "
                    f"columns {mismatches}"
                )
        comparisons[str(index)] = {
            "windows": len(actual_rows),
            "exact_match_to_pykt_generator": True,
        }

    report = {
        "dataset": "algebra2005",
        "source": "pyKT split_datasets.generate_question_sequences",
        "question_window_sha256": sha256(question_window_path),
        "sampled_test_user_rows": comparisons,
        "status": "passed",
    }
    output = data_dir / "al05_question_window_alignment_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Audit report: {output}")


if __name__ == "__main__":
    main()
