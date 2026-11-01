"""Evaluate a UniKT Mamba4KT checkpoint through pyKT's native question path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


EXPECTED_QUESTION_WINDOW_TARGETS = 61_882
ASIKT_REFERENCE = 0.7815264027514466


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
    if commit.returncode != 0:
        return {
            "commit": None,
            "dirty": None,
            "status": "not a git worktree",
        }
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        return {
            "commit": commit.stdout.strip(),
            "dirty": None,
            "status": "git status unavailable",
        }
    status_text = status.stdout.strip()
    return {
        "commit": commit.stdout.strip(),
        "dirty": bool(status_text),
        "status": status_text,
    }


def best_validation_record(
    run_dir: Path, config: dict[str, Any]
) -> tuple[int, str, float]:
    monitor = str(config["early_stopping"]["monitor"])
    mode = str(config["early_stopping"]["mode"])
    metrics_path = run_dir / "metrics_val.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or monitor not in rows[0]:
        raise ValueError(
            f"Cannot find validation metric {monitor!r} in {metrics_path}"
        )
    best = max(rows, key=lambda row: float(row[monitor]))
    if mode == "min":
        best = min(rows, key=lambda row: float(row[monitor]))
    elif mode != "max":
        raise ValueError(f"Unsupported early-stopping mode: {mode}")
    return int(best["epoch"]), monitor, float(best[monitor])


def install_pykt_package_paths(pykt_root: Path) -> None:
    """Load only the evaluator and data loader, avoiding model-wide imports."""
    for name, relative in (
        ("pykt", Path(".")),
        ("pykt.datasets", Path("datasets")),
        ("pykt.models", Path("models")),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(pykt_root / relative)]
        package.__package__ = name
        sys.modules[name] = package


class Mamba4KTPyKTAdapter:
    """Expose Mamba4KT next-item scores to pyKT's evaluator contract."""

    model_name = "asikt"

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def eval(self) -> "Mamba4KTPyKTAdapter":
        self.model.eval()
        return self

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = torch.cat(
            (batch["qseqs"][:, :1], batch["shft_qseqs"]), dim=1
        ).to(self.device)
        c = torch.cat(
            (batch["cseqs"][:, :1], batch["shft_cseqs"]), dim=1
        ).to(self.device)
        r = torch.cat(
            (batch["rseqs"][:, :1], batch["shft_rseqs"]), dim=1
        ).to(self.device)
        valid = torch.cat(
            (batch["validseqs"][:, :1], batch["shft_validseqs"]), dim=1
        ).to(self.device).bool()

        skill_ids = torch.where(valid, c.long(), torch.zeros_like(c.long()))
        question_ids = torch.where(
            valid, q.long() + 1, torch.zeros_like(q.long())
        )
        response_ids = torch.where(
            valid, r.long(), torch.zeros_like(r.long())
        )
        full_scores, _ = self.model(
            skill_ids, response_ids, valid, question_ids
        )
        return full_scores[:, :-1]


def build_model(
    unikt_root: Path, run_config: dict[str, Any], metadata: dict[str, Any]
) -> torch.nn.Module:
    sys.path.insert(0, str(unikt_root))
    from model.Mamba4KT.Mamba4KT_model import Mamba4KT

    model_config = run_config["model"]
    num_pid = (
        int(metadata["num_questions"])
        if model_config["use_rasch"]
        else 0
    )
    return Mamba4KT(
        num_c=int(metadata["num_skills"]),
        n_pid=num_pid,
        d_model=int(model_config["d_model"]),
        n_blocks=int(model_config["n_blocks"]),
        d_state=int(model_config["d_state"]),
        d_conv=int(model_config["d_conv"]),
        expand=int(model_config["expand"]),
        dropout=float(model_config["dropout"]),
        l2=float(model_config["l2"]),
        separate_qa=bool(model_config["separate_qa"]),
        prediction_head=str(model_config.get("prediction_head", "unikt")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unikt-root", type=Path, required=True)
    parser.add_argument("--pykt-root", type=Path, required=True)
    parser.add_argument("--pykt-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    unikt_root = args.unikt_root.resolve()
    pykt_root = args.pykt_root.resolve()
    data_dir = args.pykt_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / "run_config.yaml"
    config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    dataset_dir = (
        Path(config["data"]["data_base_path"]) / config["data"]["dataset"]
    )
    metadata = json.loads(
        (dataset_dir / "metadata.json").read_text(encoding="utf-8")
    )
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The native Mamba evaluation requires the CUDA device")
    model = build_model(unikt_root, config, metadata).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_format = "full_training_checkpoint"
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        checkpoint_format = "weights_only_state_dict"
    if not isinstance(state_dict, dict) or any(
        not torch.is_tensor(value) for value in state_dict.values()
    ):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    best_epoch, validation_metric, best_validation_score = (
        best_validation_record(run_dir, config)
    )

    install_pykt_package_paths(pykt_root / "pykt")
    evaluator = importlib.import_module("pykt.models.evaluate_model")
    data_loader_module = importlib.import_module("pykt.datasets.data_loader")
    test_csv = data_dir / "test_question_window_sequences.csv"
    expected_source_hash = (
        "26cc650b450c40547c5df68d0a7e944f462bb90750634b7f71f4f54abb1d2b86"
    )
    source_hash = sha256(test_csv)
    if source_hash != expected_source_hash:
        raise ValueError(
            "Question-window CSV does not match the frozen ASIKT comparison "
            f"source: {source_hash}"
        )

    bridge = Mamba4KTPyKTAdapter(model, device)
    torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory(
        prefix="mamba4kt_pykt_eval_", dir=output_dir
    ) as temporary:
        local_test_csv = Path(temporary) / test_csv.name
        local_test_csv.write_bytes(test_csv.read_bytes())
        dataset = data_loader_module.KTDataset(
            str(local_test_csv),
            ["questions", "concepts"],
            {-1},
            qtest=True,
        )
        selected_target_count = int(dataset.dori["smasks"].sum().item())
        if selected_target_count != EXPECTED_QUESTION_WINDOW_TARGETS:
            raise ValueError(
                f"Expected {EXPECTED_QUESTION_WINDOW_TARGETS} selected "
                f"targets, found {selected_target_count}"
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        aucs, accs = evaluator.evaluate_question(
            bridge,
            loader,
            "asikt",
            ["late_fusion"],
            str(output_dir / "question_window_predictions.tsv"),
        )
        qidxs, targets, predictions = (
            evaluator.collect_question_window_arrays(bridge, loader)
        )
        independent_aucs, independent_accs = evaluator.question_window_metrics(
            qidxs, targets, predictions
        )
        np.savez_compressed(
            output_dir / "question_window_arrays.npz",
            qidxs=qidxs,
            targets=targets,
            predictions=predictions,
        )
        question_groups = list(
            csv.DictReader(
                (output_dir / "question_window_predictions.tsv").open(
                    encoding="utf-8"
                ),
                delimiter="\t",
            )
        )

    prediction_file = output_dir / "question_window_predictions.tsv"
    saved_targets = np.asarray(
        [int(row["late_trues"]) for row in question_groups], dtype=np.int64
    )
    saved_scores = np.asarray(
        [float(row["late_mean"]) for row in question_groups], dtype=np.float64
    )
    saved_auc = float(roc_auc_score(saved_targets, saved_scores))
    reported_auc = float(aucs["late_mean"])
    independent_auc = float(independent_aucs["late_mean"])
    if not math.isclose(saved_auc, reported_auc, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Saved prediction table does not reproduce pyKT AUC")
    if not math.isclose(
        independent_auc, reported_auc, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError("Independent question-window recompute differs")

    data_files = {}
    for name in (
        "train_valid_sequences.csv",
        "test_sequences.csv",
        "test_question_window_sequences.csv",
    ):
        path = data_dir / name
        data_files[name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    separate_qa = bool(config["model"]["separate_qa"])
    prediction_head = str(config["model"].get("prediction_head", "unikt"))
    if separate_qa and prediction_head == "unikt":
        implementation_profile = "paper_logic_reconstruction"
    else:
        implementation_profile = "unikt_reimplementation_variant"

    result = {
        "dataset": "assist2009_corrected_collapsed",
        "model": "Mamba4KT",
        "implementation": "UniKT model/Mamba4KT",
        "implementation_profile": implementation_profile,
        "separate_qa": separate_qa,
        "prediction_head": prediction_head,
        "unikt_commit": git_info(unikt_root),
        "fold": 0,
        "seed": 42,
        "protocol": "native_pykt_question_window",
        "evaluation_chain": (
            "evaluate_question -> group_fusion -> effective_fusion -> late_fusion"
        ),
        "metric": "windowauclate_mean",
        "windowauclate_mean": reported_auc,
        "windowacclate_mean": float(accs["late_mean"]),
        "windowaucconcepts": float(aucs["concepts"]),
        "windowaccconcepts": float(accs["concepts"]),
        "independent_windowauclate_mean": independent_auc,
        "independent_windowacclate_mean": float(
            independent_accs["late_mean"]
        ),
        "question_window_rows": int(len(targets)),
        "question_window_groups": len(question_groups),
        "evaluation_batch_size": args.batch_size,
        "pykt_aggregation_groups": 3,
        "comparison": {
            "asikt_windowauclate_mean": ASIKT_REFERENCE,
            "difference_vs_asikt": reported_auc - ASIKT_REFERENCE,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "format": checkpoint_format,
            "best_epoch_zero_based": best_epoch,
            "best_validation_metric": validation_metric,
            "best_validation_score": best_validation_score,
            "best_validation_auc": (
                best_validation_score if validation_metric == "auc" else None
            ),
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
            "git": git_info(pykt_root),
        },
        "source_data": data_files,
        "data_adapter": json.loads(
            (
                Path(config["data"]["data_base_path"])
                / "data_adapter_manifest.json"
            ).read_text(encoding="utf-8")
        ),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "polars",
                    "pyarrow",
                    "mamba-ssm",
                    "causal-conv1d",
                )
            },
        },
        "source_attribution_note": (
            "This is the UniKT repository's Mamba4KT implementation, not an "
            "author-released end-to-end reproduction of the original "
            "Mamba4KT paper."
        ),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "windowauclate_mean": reported_auc,
        "windowacclate_mean": float(accs["late_mean"]),
        "independent_windowauclate_mean": independent_auc,
        "question_window_rows": int(len(targets)),
        "question_window_groups": len(question_groups),
        "result": str(result_path),
    }, indent=2))


if __name__ == "__main__":
    main()
