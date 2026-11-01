from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score


SEQUENCE_LENGTH = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode:
        return {"commit": None, "dirty": None, "status": "not a git worktree"}
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    status_text = status.stdout.strip() if status.returncode == 0 else None
    return {
        "commit": commit.stdout.strip(),
        "dirty": bool(status_text) if status_text is not None else None,
        "status": status_text,
    }


def install_pykt_package_paths(pykt_root: Path) -> None:
    for name, relative in (
        ("pykt", Path(".")),
        ("pykt.datasets", Path("datasets")),
        ("pykt.models", Path("models")),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(pykt_root / relative)]
        package.__package__ = name
        sys.modules[name] = package


def build_model(
    unikt_root: Path, config: dict[str, Any], metadata: dict[str, Any]
) -> torch.nn.Module:
    sys.path.insert(0, str(unikt_root))
    from model.Mamba4KT.Mamba4KT_model import Mamba4KT

    model_config = config["model"]
    return Mamba4KT(
        num_c=int(metadata["num_skills"]),
        n_pid=int(metadata["num_questions"]) if model_config["use_rasch"] else 0,
        d_model=int(model_config["d_model"]),
        n_blocks=int(model_config["n_blocks"]),
        d_state=int(model_config["d_state"]),
        d_conv=int(model_config["d_conv"]),
        expand=int(model_config["expand"]),
        dropout=float(model_config["dropout"]),
        l2=float(model_config["l2"]),
        separate_qa=bool(model_config["separate_qa"]),
        prediction_head=str(model_config["prediction_head"]),
    )


def iter_window_batches(
    path: Path, batch_size: int
) -> Iterator[dict[str, np.ndarray]]:
    keys = ("questions", "concepts", "responses", "selectmasks", "qidxs")
    buffers: dict[str, list[list[int]]] = {key: [] for key in keys}

    def parse(row: dict[str, str], key: str) -> list[int]:
        values = [int(value) for value in row[key].split(",")]
        if len(values) != SEQUENCE_LENGTH:
            raise ValueError(
                f"{key} has width {len(values)}, expected {SEQUENCE_LENGTH}"
            )
        return values

    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key in keys:
                buffers[key].append(parse(row, key))
            if len(buffers["questions"]) == batch_size:
                yield {key: np.asarray(values, dtype=np.int64) for key, values in buffers.items()}
                buffers = {key: [] for key in keys}
    if buffers["questions"]:
        yield {key: np.asarray(values, dtype=np.int64) for key, values in buffers.items()}


def aggregate_late_mean(
    qidxs: np.ndarray, targets: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(qidxs, kind="stable")
    sorted_qidxs = qidxs[order]
    unique_qidxs, starts, counts = np.unique(
        sorted_qidxs, return_index=True, return_counts=True
    )
    target_means = np.add.reduceat(targets[order].astype(np.float64), starts) / counts
    prediction_means = (
        np.add.reduceat(predictions[order].astype(np.float64), starts) / counts
    )
    return (
        unique_qidxs.astype(np.int64, copy=False),
        np.trunc(target_means).astype(np.int64),
        np.round(prediction_means, 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unikt-root", type=Path, required=True)
    parser.add_argument("--pykt-root", type=Path, required=True)
    parser.add_argument("--pykt-data-dir", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    unikt_root = args.unikt_root.resolve()
    pykt_root = args.pykt_root.resolve()
    data_dir = args.pykt_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    dataset_dir = Path(config["data"]["data_base_path"]) / config["data"]["dataset"]
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    adapter_manifest = json.loads(
        args.adapter_manifest.resolve().read_text(encoding="utf-8")
    )
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Mamba4KT test evaluation requires CUDA")
    model = build_model(unikt_root, config, metadata).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_format = "full_training_checkpoint"
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        checkpoint_format = "weights_only_state_dict"
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()

    install_pykt_package_paths(pykt_root / "pykt")
    evaluator = importlib.import_module("pykt.models.evaluate_model")
    window_path = data_dir / "test_question_window_sequences.csv"
    target_chunks: list[np.ndarray] = []
    prediction_chunks: list[np.ndarray] = []
    qidx_chunks: list[np.ndarray] = []
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch in iter_window_batches(window_path, args.batch_size):
            questions = torch.as_tensor(batch["questions"], device=device)
            skills = torch.as_tensor(batch["concepts"], device=device)
            responses = torch.as_tensor(batch["responses"], device=device)
            valid = (questions >= 0) & (skills >= 0)
            input_skills = torch.where(valid, skills, torch.zeros_like(skills))
            input_questions = torch.where(
                valid, questions + 1, torch.zeros_like(questions)
            )
            input_responses = torch.where(
                valid, responses, torch.zeros_like(responses)
            )
            predictions, _ = model(
                input_skills, input_responses, valid, input_questions
            )
            selected = torch.as_tensor(
                batch["selectmasks"][:, 1:] != -1, device=device
            )
            target_chunks.append(
                torch.masked_select(responses[:, 1:].float(), selected)
                .cpu()
                .numpy()
            )
            prediction_chunks.append(
                torch.masked_select(predictions[:, :-1], selected)
                .cpu()
                .numpy()
            )
            qidx_chunks.append(
                torch.as_tensor(batch["qidxs"][:, 1:], device=device)
                .masked_select(selected)
                .cpu()
                .numpy()
            )

    qidxs = np.concatenate(qidx_chunks).astype(np.int64, copy=False)
    targets = np.concatenate(target_chunks)
    predictions = np.concatenate(prediction_chunks)
    if not len(targets) or np.any(qidxs < 0):
        raise ValueError("No valid question-window targets were selected")

    aucs, accs = evaluator.question_window_metrics(qidxs, targets, predictions)
    grouped_qidxs, grouped_targets, grouped_scores = aggregate_late_mean(
        qidxs, targets, predictions
    )
    saved_auc = float(roc_auc_score(grouped_targets, grouped_scores))
    saved_acc = float(accuracy_score(grouped_targets, grouped_scores >= 0.5))
    reported_auc = float(aucs["late_mean"])
    reported_acc = float(accs["late_mean"])
    if not math.isclose(saved_auc, reported_auc, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Saved question-level table does not reproduce pyKT AUC")
    if not math.isclose(saved_acc, reported_acc, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Saved question-level table does not reproduce pyKT ACC")

    prediction_path = output_dir / "question_window_predictions.tsv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("qidx", "late_trues", "late_mean"))
        writer.writerows(
            zip(grouped_qidxs, grouped_targets, grouped_scores, strict=True)
        )
    np.savez_compressed(
        output_dir / "question_window_arrays.npz",
        qidxs=qidxs,
        targets=targets,
        predictions=predictions,
    )

    result = {
        "dataset": "algebra2005",
        "model": "Mamba4KT",
        "implementation": "UniKT initial Mamba4KT model path",
        "implementation_profile": "paper_logic_reconstruction",
        "source_attribution_note": (
            "UniKT is a third-party implementation; this is not an "
            "author-released end-to-end reproduction."
        ),
        "model_profile": {
            "separate_qa": bool(config["model"]["separate_qa"]),
            "prediction_head": config["model"]["prediction_head"],
            "d_model": int(config["model"]["d_model"]),
            "n_blocks": int(config["model"]["n_blocks"]),
            "d_state": int(config["model"]["d_state"]),
            "d_conv": int(config["model"]["d_conv"]),
            "expand": int(config["model"]["expand"]),
            "dropout": float(config["model"]["dropout"]),
            "use_rasch": bool(config["model"]["use_rasch"]),
            "l2": float(config["model"]["l2"]),
        },
        "unikt_git": git_info(unikt_root),
        "fold": int(config["data"]["fold"]),
        "seed": int(config["general"]["seed"]),
        "protocol": "pyKT question-window late-mean",
        "evaluation_chain": (
            "causal next-item scores -> selected question-window rows -> "
            "pyKT question_window_metrics late_mean"
        ),
        "metric": "windowauclate_mean",
        "windowauclate_mean": reported_auc,
        "windowacclate_mean": reported_acc,
        "windowaucconcepts": float(aucs["concepts"]),
        "windowaccconcepts": float(accs["concepts"]),
        "independent_windowauclate_mean": saved_auc,
        "independent_windowacclate_mean": saved_acc,
        "question_window_rows": int(len(targets)),
        "question_window_groups": int(len(grouped_qidxs)),
        "evaluation_batch_size": args.batch_size,
        "evaluation_memory_mode": "streamed CSV batches; no full-window tensor materialization",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "format": checkpoint_format,
        },
        "source_files": {
            name: sha256(unikt_root / name)
            for name in (
                "model/Mamba4KT/Mamba4KT_model.py",
                "model/Mamba4KT/Mamba4KT_data.py",
                "model/Mamba4KT/Mamba4KT_trainer.py",
            )
        },
        "pykt_evaluator": {
            "path": str(pykt_root / "pykt/models/evaluate_model.py"),
            "sha256": sha256(pykt_root / "pykt/models/evaluate_model.py"),
        },
        "source_data": {
            "question_window_file": {
                "path": str(window_path),
                "sha256": sha256(window_path),
                "bytes": window_path.stat().st_size,
            },
            "preprocess_manifest": {
                "path": str(data_dir / "al05_preprocess_manifest.json"),
                "sha256": sha256(data_dir / "al05_preprocess_manifest.json"),
            },
            "adapter_manifest": adapter_manifest,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "windowauclate_mean": reported_auc,
                "windowacclate_mean": reported_acc,
                "question_window_rows": len(targets),
                "question_window_groups": len(grouped_qidxs),
                "peak_gpu_memory_allocated_bytes": result["runtime"][
                    "peak_gpu_memory_allocated_bytes"
                ],
                "result": str(result_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
