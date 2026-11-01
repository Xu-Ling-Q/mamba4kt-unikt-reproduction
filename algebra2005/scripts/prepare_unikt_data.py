from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


FIELDS = (
    "sequence_id",
    "seq_pos",
    "label",
    "user",
    "question",
    "interaction_id",
    "skill",
    "fold",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ints(row: dict[str, str], key: str, maxlen: int) -> list[int]:
    values = [int(value) for value in row[key].split(",")]
    if len(values) != maxlen:
        raise ValueError(f"{key} width {len(values)} differs from {maxlen}")
    return values


def valid_positions(
    row: dict[str, str], maxlen: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    questions = parse_ints(row, "questions", maxlen)
    skills = parse_ints(row, "concepts", maxlen)
    labels = parse_ints(row, "responses", maxlen)
    masks = parse_ints(row, "selectmasks", maxlen)
    positions = [
        pos
        for pos, (question, skill, label, mask) in enumerate(
            zip(questions, skills, labels, masks, strict=True)
        )
        if question >= 0 and skill >= 0 and label in (0, 1) and mask == 1
    ]
    if positions != list(range(len(positions))):
        raise ValueError("Valid interactions must form a contiguous prefix")
    return positions, questions, skills, labels


def read_csv(path: Path):
    return path.open(newline="", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pykt-data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    args = parser.parse_args()

    source_dir = args.pykt_data_dir.resolve()
    output_root = args.output_root.resolve()
    dataset_dir = output_root / "algebra2005"
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {dataset_dir}")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config_path = source_dir / "data_config.generated.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["algebra2005"]
    maxlen = int(config["maxlen"])
    num_questions = int(config["num_q"])
    num_skills = int(config["num_c"])
    train_path = source_dir / config["train_valid_file"]
    test_path = source_dir / config["test_file"]
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Required pyKT concept-level sequence CSV is missing")

    schema = pa.schema([(name, pa.int64()) for name in FIELDS])
    writer = None
    buffers: dict[str, list[int]] = {name: [] for name in FIELDS}
    fold_users: dict[int, int] = {}
    fold_counts: Counter[int] = Counter()
    train_users: set[int] = set()
    train_sequence_count = 0
    interaction_rows = 0

    def flush() -> None:
        nonlocal writer
        size = len(buffers["sequence_id"])
        if not size:
            return
        table = pa.Table.from_pydict(buffers, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(
                dataset_dir / "algebra2005_split_skill_sequence.parquet",
                schema,
                compression="zstd",
            )
        writer.write_table(table, row_group_size=args.chunk_rows)
        for values in buffers.values():
            values.clear()

    with read_csv(train_path) as stream:
        for sequence_id, row in enumerate(csv.DictReader(stream)):
            positions, questions, skills, labels = valid_positions(row, maxlen)
            if not positions:
                raise ValueError(f"Empty training sequence {sequence_id}")
            user, fold = int(row["uid"]), int(row["fold"])
            if not 0 <= fold < 5:
                raise ValueError(f"Invalid training fold {fold}")
            previous = fold_users.setdefault(user, fold)
            if previous != fold:
                raise ValueError(f"User {user} is assigned to multiple folds")
            train_users.add(user)
            fold_counts[fold] += 1
            for pos in positions:
                question, skill, label = questions[pos], skills[pos], labels[pos]
                if not 0 <= question < num_questions:
                    raise ValueError(f"Question ID out of range: {question}")
                if not 0 <= skill < num_skills:
                    raise ValueError(f"Skill ID out of range: {skill}")
                buffers["sequence_id"].append(sequence_id)
                buffers["seq_pos"].append(pos)
                buffers["label"].append(label)
                buffers["user"].append(user)
                buffers["question"].append(question)
                buffers["interaction_id"].append(sequence_id * maxlen + pos)
                buffers["skill"].append(skill)
                buffers["fold"].append(fold)
            interaction_rows += len(positions)
            train_sequence_count += 1
            if len(buffers["sequence_id"]) >= args.chunk_rows:
                flush()
    flush()
    if writer is None:
        raise ValueError("No train/validation interactions were written")
    writer.close()
    writer = None

    test_users: set[int] = set()
    fixture_rows: dict[str, list[int]] = {
        name: []
        for name in (
            "sample_id",
            "position",
            "skill",
            "response",
            "mask",
            "group_id",
            "true_label",
            "question",
            "user_id",
        )
    }
    sample_id = 0
    fixture_users: set[int] = set()
    with read_csv(test_path) as stream:
        for row in csv.DictReader(stream):
            user = int(row["uid"])
            test_users.add(user)
            if user in fixture_users:
                continue
            positions, questions, skills, labels = valid_positions(row, maxlen)
            if len(positions) < 2:
                continue
            selected = positions[:2]
            for local_pos, source_pos in enumerate(selected):
                fixture_rows["sample_id"].append(sample_id)
                fixture_rows["position"].append(local_pos)
                fixture_rows["skill"].append(skills[source_pos])
                fixture_rows["response"].append(
                    labels[source_pos] if local_pos == 0 else 0
                )
                fixture_rows["mask"].append(1 if local_pos == 1 else 0)
                fixture_rows["group_id"].append(sample_id if local_pos == 1 else -1)
                fixture_rows["true_label"].append(labels[source_pos])
                fixture_rows["question"].append(questions[source_pos])
                fixture_rows["user_id"].append(user)
            fixture_users.add(user)
            sample_id += 1

    if train_users & test_users:
        raise ValueError("Train/validation and test users overlap")
    if sample_id != len(test_users):
        raise ValueError(
            "Every held-out user must have at least one two-step window "
            f"({sample_id} fixtures for {len(test_users)} users)"
        )

    window_path = dataset_dir / "algebra2005_windowlate.parquet"
    window_schema = pa.schema(
        [
            ("sample_id", pa.int64()),
            ("position", pa.int64()),
            ("skill", pa.int64()),
            ("response", pa.int64()),
            ("mask", pa.int64()),
            ("group_id", pa.int64()),
            ("true_label", pa.int64()),
            ("question", pa.int64()),
            ("user_id", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pydict(fixture_rows, schema=window_schema),
        window_path,
        compression="zstd",
    )

    split_path = dataset_dir / "algebra2005_split_skill_sequence.parquet"
    metadata: dict[str, Any] = {
        "dataset": "algebra2005",
        "data_base_path": str(output_root),
        "min_seq_len": 3,
        "truncation_stage": "post_expansion",
        "max_skill_seq_len": maxlen,
        "max_windowlate_seq_len": maxlen,
        "random_seed": 42,
        "num_users": len(train_users | test_users),
        "num_questions": num_questions,
        "num_skills": num_skills,
        "num_split_skill_users": len(train_users),
        "num_train_valid_users": len(train_users),
        "num_test_users": len(test_users),
        "kfold_n_splits": 5,
        "test_ratio": 0.2,
        "split_skill_sequence_data_md5": md5(split_path),
        "windowlate_data_md5": md5(window_path),
    }
    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    source_files = (config_path, train_path, test_path)
    manifest = {
        "dataset": "algebra2005",
        "mapping": (
            "one concept-expanded pyKT event per skill; each event keeps its "
            "source question ID and duplicated response label"
        ),
        "sequence_count": train_sequence_count,
        "train_valid_interactions": interaction_rows,
        "fold_sequence_counts": dict(sorted(fold_counts.items())),
        "train_valid_users": len(train_users),
        "test_users": len(test_users),
        "windowlate_loader_fixture_samples": sample_id,
        "windowlate_loader_fixture_used_for_scoring": False,
        "num_questions": num_questions,
        "num_skills": num_skills,
        "source_files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_files
        },
        "generated_files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (split_path, window_path, metadata_path)
        },
    }
    manifest_path = output_root / "al05_unikt_adapter_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Adapter manifest: {manifest_path}")


if __name__ == "__main__":
    main()
