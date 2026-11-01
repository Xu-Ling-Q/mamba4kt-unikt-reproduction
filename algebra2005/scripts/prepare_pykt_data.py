from __future__ import annotations

import argparse
import csv
import io
import hashlib
import json
import sys
import types
from contextlib import redirect_stdout
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ARCHIVE_BYTES = 22_438_892
EXPECTED_ARCHIVE_SHA256 = (
    "118ba5063aa3b428b4072d17533d158f25bde8072645a70058451e04449b2ce9"
)
USER_SUPPLIED_STATS = {
    "users": 574,
    "questions": 173_113,
    "concepts": 112,
    "interactions": 884_102,
    "questions_without_concepts": 9_926,
}
ARCHIVE_URL = (
    "https://base.ustc.edu.cn/data/KDD_Cup_2010/algebra_2005_2006.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_text_stats(path: Path) -> tuple[int, int]:
    users = interactions = 0
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index % 6 == 0:
                fields = line.rstrip("\r\n").split(",", 1)
                if len(fields) != 2:
                    raise ValueError(f"Malformed sequence header at line {index + 1}")
                users += 1
                interactions += int(fields[1])
    return users, interactions


def parse_converter_statistics(output: str) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for line in output.splitlines():
        fields = line.strip().split(",")
        if len(fields) != 7:
            continue
        try:
            rows.append(
                {
                    "interactions": int(fields[0]),
                    "users": int(fields[1]),
                    "questions": int(fields[2]),
                    "concepts": int(fields[3]),
                    "avg_interactions_per_user": float(fields[4]),
                    "avg_concepts_per_question": float(fields[5]),
                    "questions_without_concepts": int(fields[6]),
                }
            )
        except ValueError:
            continue
    if len(rows) < 2:
        raise ValueError("Could not parse pyKT AL05 preprocessing statistics")
    return rows


def read_id_map(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def install_pykt_preprocess_path(pykt_root: Path) -> None:
    for name, relative in (
        ("pykt", Path("pykt")),
        ("pykt.preprocess", Path("pykt") / "preprocess"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(pykt_root / relative)]
        package.__package__ = name
        sys.modules[name] = package


def split_summary(data_dir: Path, maxlen: int) -> dict[str, Any]:
    train_users: dict[int, int] = {}
    train_sequence_rows = 0
    with (data_dir / "train_valid_sequences.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            user, fold = int(row["uid"]), int(row["fold"])
            previous = train_users.setdefault(user, fold)
            if previous != fold:
                raise ValueError(f"Training user {user} crosses folds")
            train_sequence_rows += 1

    test_users: set[int] = set()
    test_sequence_rows = 0
    with (data_dir / "test_sequences.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            test_users.add(int(row["uid"]))
            test_sequence_rows += 1
    if set(train_users) & test_users:
        raise ValueError("Train/validation and test user IDs overlap")

    target_qidxs: set[int] = set()
    selected_targets = 0
    window_rows = 0
    with (data_dir / "test_question_window_sequences.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            window_rows += 1
            masks = row["selectmasks"].split(",")
            qidxs = row["qidxs"].split(",")
            if len(masks) != maxlen or len(qidxs) != maxlen:
                raise ValueError("Question-window row has an unexpected width")
            for mask, qidx in zip(masks, qidxs, strict=True):
                if mask == "1":
                    selected_targets += 1
                    target_qidxs.add(int(qidx))

    return {
        "train_valid_users": len(train_users),
        "held_out_test_users": len(test_users),
        "fold_user_counts": dict(sorted(Counter(train_users.values()).items())),
        "train_valid_sequence_rows": train_sequence_rows,
        "test_sequence_rows": test_sequence_rows,
        "question_window_rows": window_rows,
        "question_window_selected_targets": selected_targets,
        "question_window_groups": len(target_qidxs),
    }


def read_data_streaming(split_module: Any, path: str, min_seq_len: int = 3):
    data: dict[str, list[Any]] = {}
    effective_keys: set[str] = set()
    deleted_users = deleted_interactions = bad_responses = good_interactions = 0

    with Path(path).open(encoding="utf-8") as stream:
        while True:
            header = stream.readline()
            if not header:
                break
            header = header.strip()
            fields = header.split(",")
            if "(" in fields[0]:
                student_id, sequence_len = fields[0].replace("(", ""), int(fields[2])
            else:
                student_id, sequence_len = fields[0], int(fields[1])
            if sequence_len < min_seq_len:
                for _ in range(5):
                    if not stream.readline():
                        raise ValueError("Unexpected end of pyKT data.txt")
                deleted_users += 1
                deleted_interactions += sequence_len
                continue

            question_line, concept_line, response_line, time_line, use_line = (
                stream.readline().strip() for _ in range(5)
            )
            if any(line == "" for line in (
                question_line, concept_line, response_line, time_line, use_line
            )):
                raise ValueError("Unexpected end of pyKT data.txt")

            current: dict[str, Any] = {"uid": student_id}
            effective_keys.add("uid")
            current["questions"] = (
                question_line.split(",") if "NA" not in question_line else []
            )
            if "NA" not in question_line:
                effective_keys.add("questions")
            current["concepts"] = (
                concept_line.split(",") if "NA" not in concept_line else []
            )
            if "NA" not in concept_line:
                effective_keys.add("concepts")

            current_responses: list[int] = []
            valid_response_line = True
            effective_keys.add("responses")
            if "NA" not in response_line:
                try:
                    current_responses = [int(value) for value in response_line.split(",")]
                    valid_response_line = all(
                        value in (0, 1) for value in current_responses
                    )
                except ValueError:
                    valid_response_line = False
            current["responses"] = current_responses
            current["timestamps"] = (
                time_line.split(",") if "NA" not in time_line else []
            )
            if "NA" not in time_line:
                effective_keys.add("timestamps")
            current["usetimes"] = use_line.split(",") if "NA" not in use_line else []
            if "NA" not in use_line:
                effective_keys.add("usetimes")

            if not valid_response_line:
                bad_responses += 1
                continue
            if len(current_responses) != sequence_len:
                raise ValueError("Response count differs from the sequence header")
            for key in ("questions", "concepts", "timestamps", "usetimes"):
                if key in effective_keys and current[key] and len(current[key]) != sequence_len:
                    raise ValueError(f"{key} count differs from the sequence header")

            good_interactions += sequence_len
            for key in effective_keys:
                data.setdefault(key, [])
                data[key].append(
                    current[key]
                    if key == "uid"
                    else ",".join(str(value) for value in current[key])
                )

    frame = split_module.pd.DataFrame(data)
    print(
        "delete bad stu num of len: "
        f"{deleted_users}, delete interactions: {deleted_interactions}, "
        f"of r: {bad_responses}, good num: {good_interactions}"
    )
    return frame, effective_keys


def append_frame(frame: Any, path: Path) -> None:
    if frame is None or frame.empty:
        return
    frame.to_csv(
        path,
        index=None,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
    )


def offset_sequence_column(frame: Any, column: str, offset: int) -> None:
    def shift(values: str) -> str:
        return ",".join(
            str(value + offset if (value := int(token)) >= 0 else value)
            for token in values.split(",")
        )

    frame[column] = frame[column].map(shift)


def split_pykt_data(
    output_dir: Path,
    data_txt: Path,
    config_path: Path,
    maxlen: int = 200,
    kfold: int = 5,
) -> None:
    from pykt.preprocess import split_datasets as split

    total_df, effective_keys = read_data_streaming(split, str(data_txt))
    if "concepts" in effective_keys:
        max_concepts = split.get_max_concepts(total_df)
    else:
        max_concepts = -1

    total_df, effective_keys = split.extend_multi_concepts(total_df, effective_keys)
    total_df, id_maps = split.id_mapping(total_df)
    id_maps["max_concepts"] = max_concepts
    split.save_id2idx(id_maps, output_dir / "keyid2idx.json")
    effective_keys.add("fold")
    save_keys = [key for key in split.ALL_KEYS if key in effective_keys]

    train_df, test_df = split.train_test_split(total_df, 0.2)
    del total_df
    train_valid = split.KFold_split(train_df, kfold)
    del train_df
    train_valid[save_keys].to_csv(output_dir / "train_valid.csv", index=None)
    train_sequences = split.generate_sequences(
        train_valid, effective_keys, min_seq_len=3, maxlen=maxlen
    )
    train_sequences.to_csv(
        output_dir / "train_valid_sequences.csv", index=None
    )
    del train_sequences, train_valid

    test_df = test_df.copy()
    test_df["fold"] = [-1] * test_df.shape[0]
    test_df["cidxs"] = split.get_inter_qidx(test_df)
    test_keys = list(effective_keys) + ["cidxs"]
    test_sequences = split.generate_sequences(
        test_df, test_keys, min_seq_len=3, maxlen=maxlen
    )
    test_sequences.to_csv(output_dir / "test_sequences.csv", index=None)
    del test_sequences

    test_window_path = output_dir / "test_window_sequences.csv"
    question_path = output_dir / "test_question_sequences.csv"
    question_window_path = output_dir / "test_question_window_sequences.csv"
    qidx_offset = 0
    for user_index in range(test_df.shape[0]):
        user_df = test_df.iloc[[user_index]].copy()
        window_sequences = split.generate_window_sequences(
            user_df, test_keys, maxlen=maxlen
        )
        append_frame(window_sequences, test_window_path)
        del window_sequences

        question_count = str(user_df.iloc[0]["is_repeat"]).split(",").count("0")
        for window, path in ((False, question_path), (True, question_window_path)):
            generated, frame = split.generate_question_sequences(
                user_df.copy(),
                effective_keys,
                window=window,
                min_seq_len=3,
                maxlen=maxlen,
            )
            if not generated:
                raise ValueError("AL05 question-window generation was unavailable")
            offset_sequence_column(frame, "qidxs", qidx_offset)
            offset_sequence_column(frame, "orirow", user_index)
            append_frame(frame, path)
            del frame
        qidx_offset += question_count

    if not question_path.is_file() or not question_window_path.is_file():
        raise ValueError("pyKT question-level test files were not generated")

    test_df[save_keys + ["cidxs"]].to_csv(
        output_dir / "test.csv", index=None
    )
    split.write_config(
        dataset_name="algebra2005",
        dkeyid2idx=id_maps,
        effective_keys=effective_keys,
        configf=str(config_path),
        dpath=str(output_dir),
        k=kfold,
        min_seq_len=3,
        maxlen=maxlen,
        flag=True,
    )
    print(
        "bounded question-window generation: "
        f"users={test_df.shape[0]}, question_ids={qidx_offset}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pykt-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()

    pykt_root = args.pykt_root.resolve()
    archive = args.archive.resolve()
    raw_train = args.raw_train.resolve()
    output_dir = args.output_dir.resolve()
    if not archive.is_file() or not raw_train.is_file():
        raise FileNotFoundError("Expected AL05 archive and extracted train file")
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("AL05 archive size does not match the recorded source")
    archive_hash = sha256(archive)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("AL05 archive SHA-256 does not match the recorded source")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.audit_existing:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    install_pykt_preprocess_path(pykt_root)
    data_txt = output_dir / "data.txt"
    config_path = output_dir / "data_config.generated.json"
    converter_stats_path = output_dir / "al05_converter_observed.json"
    if not args.audit_existing:
        import pandas as pd

        pd.options.future.infer_string = False
        from pykt.preprocess.algebra2005_preprocess import read_data_from_csv

        config_path.write_text("{}\n", encoding="utf-8")
        converter_output = io.StringIO()
        with redirect_stdout(converter_output):
            read_data_from_csv(str(raw_train), str(data_txt))
        print(converter_output.getvalue(), end="")
        converter_stats = parse_converter_statistics(converter_output.getvalue())
        converter_stats_path.write_text(
            json.dumps(converter_stats, indent=2) + "\n", encoding="utf-8"
        )
        split_pykt_data(output_dir, data_txt, config_path)
    else:
        converter_stats = read_id_map(converter_stats_path)

    ids = read_id_map(output_dir / "keyid2idx.json")
    data_config = read_id_map(config_path)["algebra2005"]
    converter_after_drop = converter_stats[-1]
    observed = {
        key: int(converter_after_drop[key])
        for key in ("users", "questions", "concepts", "interactions")
    }

    text_users, text_interactions = raw_text_stats(data_txt)
    summary = split_summary(output_dir, maxlen=200)
    manifest = {
        "dataset": "algebra2005",
        "source": {
            "claim": "obtained from the USTC public mirror; not a direct CMU download",
            "url": ARCHIVE_URL,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_hash,
            "train_member": "algebra_2005_2006/algebra_2005_2006_train.txt",
            "train_bytes": raw_train.stat().st_size,
            "train_sha256": sha256(raw_train),
        },
        "preprocessing": {
            "implementation": "pyKT AL05 preprocessing and split primitives; test windows emitted one user at a time",
            "min_seq_len": 3,
            "max_seq_len": 200,
            "test_ratio": 0.2,
            "random_state": 1024,
            "kfold": 5,
            "statistics_after_drop": converter_after_drop,
            "statistics_comparison": {
                "user_supplied": USER_SUPPLIED_STATS,
                "observed_after_drop": converter_after_drop,
                "matches": {
                    "users": observed["users"] == USER_SUPPLIED_STATS["users"],
                    "questions": observed["questions"] == USER_SUPPLIED_STATS["questions"],
                    "concepts": observed["concepts"] == USER_SUPPLIED_STATS["concepts"],
                    "interactions": observed["interactions"]
                    == USER_SUPPLIED_STATS["interactions"],
                    "questions_without_concepts": converter_after_drop[
                        "questions_without_concepts"
                    ]
                    == USER_SUPPLIED_STATS["questions_without_concepts"],
                },
            },
            "data_txt": {
                "users": text_users,
                "interactions": text_interactions,
            },
            "model_vocabulary_after_min_seq_filter": {
                "users": len(ids["uid"]),
                "questions": len(ids["questions"]),
                "concepts": len(ids["concepts"]),
                "max_concepts": data_config["max_concepts"],
            },
            "split": summary,
            "data_config": data_config,
        },
    }
    manifest_path = output_dir / "al05_preprocess_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
