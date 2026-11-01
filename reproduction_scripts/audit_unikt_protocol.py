"""Audit the frozen AS09 split, UniKT adapter, and checkpoint predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluate_pykt_question_window import (
    Mamba4KTPyKTAdapter,
    best_validation_record,
    build_model,
    install_pykt_package_paths,
    sha256,
)


def sequence(row: dict[str, str], name: str) -> np.ndarray:
    return np.asarray([int(x) for x in row[name].split(",")], dtype=np.int64)


def pykt_batch(
    questions: torch.Tensor,
    concepts: torch.Tensor,
    responses: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pair_mask = valid[:, :-1] & valid[:, 1:]
    batch = {}
    for name, values in (
        ("qseqs", questions),
        ("cseqs", concepts),
        ("rseqs", responses),
        ("validseqs", valid),
    ):
        batch[name] = values[:, :-1] * pair_mask
        batch["shft_" + name] = values[:, 1:] * pair_mask
    batch["masks"] = pair_mask
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unikt-root", type=Path, required=True)
    parser.add_argument("--pykt-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--asikt-arrays", type=Path, required=True)
    parser.add_argument("--mamba-arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    unikt_root = args.unikt_root.resolve()
    pykt_root = args.pykt_root.resolve()
    data_dir = args.data_dir.resolve()
    config = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    dataset_dir = (
        Path(config["data"]["data_base_path"]) / config["data"]["dataset"]
    )
    metadata = json.loads((dataset_dir / "metadata.json").read_text())

    sys.path.insert(0, str(unikt_root))
    from utils.model_data.packed_sequences import pack_sequences, to_first_skill_ids

    frame = pl.read_parquet(
        dataset_dir / "assistments09_split_skill_sequence.parquet"
    )
    packed = pack_sequences(frame, max_seq_len=200, num_folds=5)
    concepts = to_first_skill_ids(packed.skills)
    csv_path = data_dir / "train_valid_sequences.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != len(packed.sequence_id):
        raise AssertionError("Converted sequence count does not match the CSV")
    folds: dict[int, int] = {}
    for idx, row in enumerate(rows):
        if packed.sequence_id[idx] != idx:
            raise AssertionError(f"Sequence index mismatch at CSV row {idx}")
        q, c, r, selected = (
            sequence(row, name)
            for name in ("questions", "concepts", "responses", "selectmasks")
        )
        valid = (q >= 0) & (c >= 0) & ((r == 0) | (r == 1))
        if not np.array_equal(valid, selected == 1):
            raise AssertionError(f"Selection mask mismatch at CSV row {idx}")
        if not np.array_equal(packed.mask[idx], valid):
            raise AssertionError(f"Valid mask mismatch at CSV row {idx}")
        for name, actual, expected in (
            ("question", packed.question[idx], np.where(valid, q, 0)),
            ("concept", concepts[idx], np.where(valid, c, 0)),
            ("response", packed.response[idx], np.where(valid, r, 0)),
        ):
            if not np.array_equal(actual, expected):
                raise AssertionError(f"{name} mismatch at CSV row {idx}")
        if int(packed.user[idx]) != int(row["uid"]):
            raise AssertionError(f"User mismatch at CSV row {idx}")
        if int(packed.fold[idx]) != int(row["fold"]):
            raise AssertionError(f"Fold mismatch at CSV row {idx}")
        folds[int(packed.fold[idx])] = folds.get(int(packed.fold[idx]), 0) + 1

    part = packed.partition(0)
    if (len(part.train), len(part.val)) != (3016, 741):
        raise AssertionError("Unexpected fold 0 train/validation partition")
    train_users = set(packed.user[part.train].tolist())
    val_users = set(packed.user[part.val].tolist())
    with (data_dir / "test_sequences.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        test_users = {int(row["uid"]) for row in csv.DictReader(stream)}
    if train_users & val_users or train_users & test_users or val_users & test_users:
        raise AssertionError("Train, validation, and test users overlap")

    asikt = np.load(args.asikt_arrays)
    mamba = np.load(args.mamba_arrays)
    for key in ("qidxs", "targets"):
        if not np.array_equal(asikt[key], mamba[key]):
            raise AssertionError(f"ASIKT/Mamba held-out {key} arrays differ")
    if len(mamba["targets"]) != 61_882:
        raise AssertionError("Unexpected number of question-window targets")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Native Mamba audit requires CUDA")
    model = build_model(unikt_root, config, metadata).to(device)
    state = torch.load(
        run_dir / "best_model.pth", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    adapter = Mamba4KTPyKTAdapter(model, device)

    validation_targets = []
    validation_predictions = []
    max_adapter_difference = 0.0
    with torch.inference_mode():
        for indices in np.array_split(part.val, (len(part.val) + 63) // 64):
            q = torch.as_tensor(packed.question[indices], device=device)
            c = torch.as_tensor(concepts[indices], device=device)
            r = torch.as_tensor(packed.response[indices], device=device)
            valid = torch.as_tensor(packed.mask[indices], device=device)
            pair_mask = valid[:, :-1] & valid[:, 1:]
            pid = torch.where(valid, q + 1, torch.zeros_like(q))
            raw_scores, _ = model(c, r, valid, pid)
            adapted_scores = adapter(pykt_batch(q, c, r, valid))
            difference = (
                (raw_scores[:, :-1] - adapted_scores).abs()[pair_mask].max().item()
            )
            max_adapter_difference = max(max_adapter_difference, difference)
            validation_targets.append(r[:, 1:][pair_mask].cpu().numpy())
            validation_predictions.append(raw_scores[:, :-1][pair_mask].cpu().numpy())

    validation_auc = float(
        roc_auc_score(
            np.concatenate(validation_targets),
            np.concatenate(validation_predictions),
        )
    )
    best_epoch, metric_name, recorded_auc = best_validation_record(run_dir, config)
    if metric_name != "auc" or abs(validation_auc - recorded_auc) > 1e-7:
        raise AssertionError(
            f"Validation AUC differs: {validation_auc} versus {recorded_auc}"
        )
    if max_adapter_difference > 1e-6:
        raise AssertionError("Adapter differs from the UniKT training model")

    # Test on real held-out windows, but do not compute another test metric.
    install_pykt_package_paths(pykt_root / "pykt")
    from pykt.datasets.data_loader import KTDataset

    window_csv = data_dir / "test_question_window_sequences.csv"
    with window_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        window_rows = [next(reader) for _ in range(128)]

    with tempfile.TemporaryDirectory(prefix="unikt_protocol_audit_") as temp:
        fixture_path = Path(temp) / "windows.csv"
        with fixture_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(window_rows)
        dataset = KTDataset(
            str(fixture_path), ["questions", "concepts"], {-1}, qtest=True
        )
        batch, question_info = next(iter(DataLoader(dataset, batch_size=128)))
        scores = adapter(batch)
        manual_q = torch.as_tensor(
            np.stack([np.maximum(sequence(row, "questions"), 0) for row in window_rows]),
            device=device,
        )
        manual_c = torch.as_tensor(
            np.stack([np.maximum(sequence(row, "concepts"), 0) for row in window_rows]),
            device=device,
        )
        manual_r = torch.as_tensor(
            np.stack([np.maximum(sequence(row, "responses"), 0) for row in window_rows]),
            device=device,
        )
        manual_valid = torch.as_tensor(
            np.stack([sequence(row, "questions") >= 0 for row in window_rows]),
            device=device,
        )
        manual_pid = torch.where(manual_valid, manual_q + 1, torch.zeros_like(manual_q))
        with torch.inference_mode():
            manual_scores, _ = model(manual_c, manual_r, manual_valid, manual_pid)
        selected = batch["smasks"].to(device)
        actual_targets = batch["shft_rseqs"].to(device)
        manual_targets = manual_r[:, 1:]
        if not torch.equal(actual_targets[selected].long(), manual_targets[selected]):
            raise AssertionError("Window target alignment differs from source CSV")
        if not torch.allclose(
            scores[selected], manual_scores[:, :-1][selected], rtol=0, atol=1e-6
        ):
            raise AssertionError("Window adapter predictions differ from raw model")
        qidx = question_info["qidxs"].to(device)
        expected_qidx = torch.as_tensor(
            np.stack([sequence(row, "qidxs")[1:] for row in window_rows]),
            device=device,
        )
        if not torch.equal(qidx[selected], expected_qidx[selected]):
            raise AssertionError("Question index alignment differs from source CSV")

        selected_positions = selected.nonzero()
        chosen = selected_positions[
            torch.linspace(
                0, len(selected_positions) - 1, min(16, len(selected_positions)),
                device=device,
            ).long()
        ]
        max_future_effect = 0.0
        max_history_effect = 0.0
        for sample, score_pos in chosen.tolist():
            target_pos = score_pos + 1
            flipped = manual_r[sample : sample + 1].clone()
            flipped[:, target_pos:] = 1 - flipped[:, target_pos:]
            with torch.inference_mode():
                altered, _ = model(
                    manual_c[sample : sample + 1],
                    flipped,
                    manual_valid[sample : sample + 1],
                    manual_pid[sample : sample + 1],
                )
            effect = abs(
                float(altered[0, score_pos])
                - float(manual_scores[sample, score_pos])
            )
            max_future_effect = max(max_future_effect, effect)
            history_flipped = manual_r[sample : sample + 1].clone()
            history_flipped[:, score_pos] = 1 - history_flipped[:, score_pos]
            with torch.inference_mode():
                altered_history, _ = model(
                    manual_c[sample : sample + 1],
                    history_flipped,
                    manual_valid[sample : sample + 1],
                    manual_pid[sample : sample + 1],
                )
            max_history_effect = max(
                max_history_effect,
                abs(
                    float(altered_history[0, score_pos])
                    - float(manual_scores[sample, score_pos])
                ),
            )
        if max_future_effect > 1e-6:
            raise AssertionError("Current/future responses affect target prediction")
        if max_history_effect <= 1e-6:
            raise AssertionError("Causality control did not respond to history")

    result = {
        "train_valid_csv_sha256": sha256(csv_path),
        "window_csv_sha256": sha256(window_csv),
        "checkpoint_sha256": sha256(run_dir / "best_model.pth"),
        "prediction_head": str(
            config["model"].get("prediction_head", "unikt")
        ),
        "separate_qa": bool(config["model"]["separate_qa"]),
        "implementation_profile": (
            "paper_logic_reconstruction"
            if bool(config["model"]["separate_qa"])
            and str(config["model"].get("prediction_head", "unikt")) == "unikt"
            else "unikt_reimplementation_variant"
        ),
        "converted_sequences_exact": len(rows),
        "fold_counts": folds,
        "train_sequences": len(part.train),
        "validation_sequences": len(part.val),
        "train_validation_test_users_disjoint": True,
        "asikt_test_qidxs_and_targets_identical": int(len(mamba["targets"])),
        "checkpoint_best_epoch_zero_based": best_epoch,
        "recorded_validation_auc": recorded_auc,
        "recomputed_validation_auc": validation_auc,
        "validation_adapter_max_abs_difference": max_adapter_difference,
        "real_window_fixture_rows": len(window_rows),
        "real_window_fixture_selected_targets": int(selected.sum().item()),
        "window_alignment_matches_raw_model": True,
        "causality_probe_count": len(chosen),
        "max_future_response_effect": max_future_effect,
        "max_history_response_effect": max_history_effect,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
