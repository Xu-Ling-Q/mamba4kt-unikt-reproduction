from __future__ import annotations

import argparse
import json
import sys
import types
from argparse import Namespace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unikt-root", type=Path, required=True)
    parser.add_argument("--data-base-path", type=Path, required=True)
    args = parser.parse_args()

    unikt_root = args.unikt_root.resolve()
    data_base_path = args.data_base_path.resolve()
    sys.path.insert(0, str(unikt_root))
    from model.Mamba4KT.Mamba4KT_data import Mamba4KTModelData
    from utils.data_process.kddcup2010 import Algebra2005Data

    source = Algebra2005Data(
        Namespace(
            dataset="algebra2005",
            data_base_path=str(data_base_path),
            seed=42,
        )
    )
    model_data = Mamba4KTModelData(source)
    rc = types.SimpleNamespace(
        data=types.SimpleNamespace(fold=0),
        model=types.SimpleNamespace(batch_size=64),
    )
    train_dataset, val_dataset, test_loader = model_data.prepare_data(rc)
    if len(train_dataset) != 3110:
        raise AssertionError(f"Expected 3,110 fold-0 train sequences, got {len(train_dataset)}")
    if len(val_dataset) != 870:
        raise AssertionError(f"Expected 870 fold-0 validation sequences, got {len(val_dataset)}")
    if len(test_loader.dataset) != 114 or len(test_loader) != 2:
        raise AssertionError(
            "Expected 114 test fixture windows in two batches, got "
            f"{len(test_loader.dataset)} samples / {len(test_loader)} batches"
        )

    sequence, response, mask, question = train_dataset[0]
    if sequence.shape != (200,) or response.shape != (200,):
        raise AssertionError("Training sample does not have the frozen length 200")
    if not mask.any() or bool((sequence[mask] < 0).any()):
        raise AssertionError("Training sample has invalid concept IDs")
    if bool((question[mask] < 0).any()):
        raise AssertionError("Training sample has invalid question IDs")

    first_test_batch = next(iter(test_loader))
    if len(first_test_batch) != 7:
        raise AssertionError(f"Expected seven windowlate tensors, got {len(first_test_batch)}")
    selected = first_test_batch[2]
    if selected.shape != (64, 200) or int(selected.sum()) != 64:
        raise AssertionError(
            f"Unexpected windowlate target mask: {tuple(selected.shape)}, "
            f"{int(selected.sum())} selected"
        )
    if bool((first_test_batch[1][selected] != 0).any()):
        raise AssertionError("Masked target responses must be zero to prevent leakage")

    result = {
        "dataset": "algebra2005",
        "device": "cpu",
        "fold": 0,
        "train_sequences": len(train_dataset),
        "validation_sequences": len(val_dataset),
        "test_fixture_samples": len(test_loader.dataset),
        "test_loader_batches": len(test_loader),
        "train_sample_length": int(sequence.numel()),
        "first_test_batch_shape": list(selected.shape),
        "first_test_batch_selected_targets": int(selected.sum()),
        "target_response_masked": True,
        "num_questions": source.get_metadata("num_questions"),
        "num_skills": source.get_metadata("num_skills"),
    }
    output = data_base_path.parent / "al05_dataset_smoke.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Smoke report: {output}")


if __name__ == "__main__":
    main()
