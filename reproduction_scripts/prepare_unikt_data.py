"""Convert the frozen pyKT ASSIST2009 split into UniKT's parquet layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl


SEQUENCE_LENGTH = 200
EXPECTED_FOLD_COUNTS = {0: 741, 1: 741, 2: 750, 3: 778, 4: 747}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sequence(row: dict[str, str], column: str) -> list[int]:
    values = [int(value) for value in row[column].split(",")]
    if len(values) != SEQUENCE_LENGTH:
        raise ValueError(
            f"{column} has {len(values)} values; expected {SEQUENCE_LENGTH}"
        )
    return values


def valid_positions(
    questions: list[int], skills: list[int], responses: list[int]
) -> list[int]:
    valid = [
        position
        for position, (question, skill, response) in enumerate(
            zip(questions, skills, responses, strict=True)
        )
        if question >= 0 and skill >= 0 and response in (0, 1)
    ]
    if valid != list(range(len(valid))):
        raise ValueError("Expected valid interactions to form a contiguous prefix")
    return valid


def validate_id_ranges(
    rows: list[dict[str, str]], num_questions: int, num_skills: int
) -> None:
    for row in rows:
        questions = parse_sequence(row, "questions")
        skills = parse_sequence(row, "concepts")
        responses = parse_sequence(row, "responses")
        for position in valid_positions(questions, skills, responses):
            if not 0 <= questions[position] < num_questions:
                raise ValueError(
                    f"Question ID {questions[position]} exceeds its vocabulary"
                )
            if not 0 <= skills[position] < num_skills:
                raise ValueError(
                    f"Skill ID {skills[position]} exceeds its vocabulary"
                )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def convert_train_valid(
    rows: list[dict[str, str]],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    columns: dict[str, list[int]] = {
        name: []
        for name in (
            "sequence_id",
            "seq_pos",
            "label",
            "user",
            "question",
            "interaction_id",
            "skill",
            "fold",
        )
    }
    fold_counts: Counter[int] = Counter()
    user_folds: dict[int, int] = {}
    sequence_lengths: list[int] = []

    for sequence_id, row in enumerate(rows):
        questions = parse_sequence(row, "questions")
        skills = parse_sequence(row, "concepts")
        responses = parse_sequence(row, "responses")
        positions = valid_positions(questions, skills, responses)
        if not positions:
            raise ValueError(f"Empty training sequence at row {sequence_id}")

        user = int(row["uid"])
        fold = int(row["fold"])
        if fold not in EXPECTED_FOLD_COUNTS:
            raise ValueError(f"Unexpected train/validation fold {fold}")
        previous_fold = user_folds.setdefault(user, fold)
        if previous_fold != fold:
            raise ValueError(f"User {user} is assigned to multiple folds")

        fold_counts[fold] += 1
        sequence_lengths.append(len(positions))
        for position in positions:
            columns["sequence_id"].append(sequence_id)
            columns["seq_pos"].append(position)
            columns["label"].append(responses[position])
            columns["user"].append(user)
            columns["question"].append(questions[position])
            columns["interaction_id"].append(sequence_id * SEQUENCE_LENGTH + position)
            columns["skill"].append(skills[position])
            columns["fold"].append(fold)

    actual_fold_counts = dict(sorted(fold_counts.items()))
    if actual_fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(
            f"Frozen fold counts differ: {actual_fold_counts} != "
            f"{EXPECTED_FOLD_COUNTS}"
        )

    frame = pl.DataFrame(columns).with_columns(
        pl.col("sequence_id", "seq_pos", "label", "user", "question",
               "interaction_id", "skill", "fold").cast(pl.Int64)
    )
    stats = {
        "sequence_count": len(rows),
        "fold_sequence_counts": actual_fold_counts,
        "train_sequences_fold0": sum(
            count for fold, count in fold_counts.items() if fold != 0
        ),
        "validation_sequences_fold0": fold_counts[0],
        "unique_train_valid_users": len(user_folds),
        "max_sequence_length": max(sequence_lengths),
        "interaction_rows": frame.height,
    }
    if stats["train_sequences_fold0"] != 3016:
        raise ValueError("Fold 0 training sequence count is not 3,016")
    return frame, stats


def convert_test_fixture(
    rows: list[dict[str, str]],
) -> pl.DataFrame:
    """Build one valid held-out window per user for UniKT loader construction.

    The official UniKT test callback is disabled for this run. The reportable
    test metric is produced separately by the frozen pyKT question-window data.
    """
    columns: dict[str, list[int]] = {
        name: []
        for name in (
            "sample_id",
            "position",
            "skill",
            "question",
            "response",
            "mask",
            "user_id",
            "group_id",
            "true_label",
        )
    }
    for sample_id, row in enumerate(rows):
        questions = parse_sequence(row, "questions")
        skills = parse_sequence(row, "concepts")
        responses = parse_sequence(row, "responses")
        positions = valid_positions(questions, skills, responses)
        if len(positions) < 2:
            raise ValueError(f"Held-out sequence {sample_id} has no target window")

        history_position, target_position = positions[0], positions[1]
        for local_position, source_position in enumerate(
            (history_position, target_position)
        ):
            columns["sample_id"].append(sample_id)
            columns["position"].append(local_position)
            columns["skill"].append(skills[source_position])
            columns["question"].append(questions[source_position])
            columns["response"].append(
                0 if local_position == 1 else responses[source_position]
            )
            columns["mask"].append(1 if local_position == 1 else 0)
            columns["user_id"].append(int(row["uid"]))
            columns["group_id"].append(sample_id)
            columns["true_label"].append(responses[source_position])

    return pl.DataFrame(columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pykt-data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.pykt_data_dir.resolve()
    output_root = args.output_root.resolve()
    dataset_dir = output_root / "assistments09"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config_path = source_dir / "data_config.generated.json"
    data_config = json.loads(config_path.read_text(encoding="utf-8"))[
        "assist2009_corrected_collapsed"
    ]
    train_path = source_dir / data_config["train_valid_file"]
    test_path = source_dir / data_config["test_file"]
    train_rows = read_csv(train_path)
    test_rows = read_csv(test_path)
    if len(test_rows) != 904:
        raise ValueError(f"Expected 904 held-out sequences, found {len(test_rows)}")

    validate_id_ranges(
        train_rows,
        int(data_config["num_q"]),
        int(data_config["num_c"]),
    )
    validate_id_ranges(
        test_rows,
        int(data_config["num_q"]),
        int(data_config["num_c"]),
    )
    train_frame, train_stats = convert_train_valid(train_rows)
    fixture_frame = convert_test_fixture(test_rows)
    train_frame.write_parquet(
        dataset_dir / "assistments09_split_skill_sequence.parquet",
        compression="zstd",
    )
    fixture_frame.write_parquet(
        dataset_dir / "assistments09_windowlate.parquet",
        compression="zstd",
    )

    source_num_skills = int(data_config["num_c"])
    source_num_questions = int(data_config["num_q"])
    metadata = {
        "dataset": "assistments09",
        "data_base_path": str(output_root),
        "processed": True,
        "random_seed": 42,
        "kfold_n_splits": 5,
        "max_skill_seq_len": SEQUENCE_LENGTH,
        "max_windowlate_seq_len": SEQUENCE_LENGTH,
        "num_users": train_stats["unique_train_valid_users"],
        "num_questions": source_num_questions,
        "num_skills": source_num_skills,
        "source_num_skills": source_num_skills,
        "skill_ids_zero_based": True,
        "windowlate_fixture_samples": len(test_rows),
        "windowlate_fixture_used_for_scoring": False,
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "source_dataset": "assist2009_corrected_collapsed",
        "unikt_dataset_alias": "assistments09",
        "source_files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (config_path, train_path, test_path)
        },
        "conversion": train_stats
        | {
            "held_out_test_sequences": len(test_rows),
            "windowlate_loader_fixture_samples": fixture_frame[
                "sample_id"
            ].n_unique(),
            "skill_ids": "source 0-based IDs passed through unchanged",
            "num_skills_metadata": metadata["num_skills"],
            "num_skills_source_vocabulary": source_num_skills,
        },
        "generated_files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (
                dataset_dir / "assistments09_split_skill_sequence.parquet",
                dataset_dir / "assistments09_windowlate.parquet",
                dataset_dir / "metadata.json",
            )
        },
    }
    manifest_path = output_root / "data_adapter_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["conversion"], indent=2))
    print(f"Generated UniKT parquet data at {dataset_dir}")
    print(f"Adapter manifest: {manifest_path}")


if __name__ == "__main__":
    main()
