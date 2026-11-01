# Final Mamba4KT Unified-Protocol Report

Date: 2026-10-30

> Update: the later paper-faithful reconstruction from UniKT's initial
> Mamba4KT commit completed at test AUC `0.7672619`. See
> `PAPER_FAITHFUL_REPORT.md`. This file remains the earlier third-party
> UniKT implementation report and is not the final source-fidelity record.

## Bottom Line

This work does **not** contain an official-author-source end-to-end
reproduction of Mamba4KT. No such public source was found. The completed run
is a reproduction of the third-party UniKT implementation, pinning commit
`0493bfbba31373a3058f287cd6625ed10d48895e` and evaluating it on the frozen
AS09 fold-0 question-window protocol.

The final UniKT-head model reaches test AUC `0.7771054691669028`, which is
`0.0044209335845437625` below the same-protocol ASIKT reference
`0.7815264027514466`. This is a valid honest comparison, not a failed result
that must be hidden.

## Results

| Variant | Configuration | Validation AUC | Test AUC | Test ACC |
| --- | --- | ---: | ---: | ---: |
| Old UniKT baseline | d128, lr 0.001 | 0.8248046 | 0.7422034 | 0.7213525 |
| UniKT head candidate | d64, lr 0.003 | 0.8461483 | 0.7719114 | 0.7289320 |
| **UniKT head final** | **d256, lr 0.003** | **0.8491687** | **0.7771055** | **0.7321394** |
| Paper head d64 best | d64, lr 0.003 | 0.8126328 | 0.7243584 | 0.7097255 |
| **Paper head final** | **d256, lr 0.0001** | **0.8155762** | **0.7295355** | **0.7157394** |

The UniKT head exceeds the old baseline by 3.49 test-AUC percentage points.
The paper head is 4.76 test-AUC percentage points below the final UniKT head,
showing that UniKT's additional `q_next` concatenation materially changes the
implementation.

## Why The ASIKT Number Is 0.7815, Not 0.7978

`0.7978` is not a result from this frozen unified protocol. It cannot be
subtracted directly from `0.7771055`. The comparable local number is ASIKT
`0.7815264027514466`, evaluated on exactly the same 61,882 targets, same
52,378 question-window groups, same test arrays, and same pyKT aggregation.

ASIKT has stronger source fidelity: its release files are retained
byte-for-byte and its model, masked BCE, optimizer, and early-stopping logic
are used. But it is adapted to the project AS09 split and uses batch size 32
instead of the official default 48 because batch 48 exhausted the 6 GB GPU.
It is therefore an official-source unified-protocol run, not an unchanged
author-split/launcher run.

## Protocol Integrity

- Train, validation, and test users are disjoint.
- ASIKT and Mamba4KT test `qidxs` and targets match for all 61,882 rows.
- The frozen test question-window CSV hash is
  `26cc650b450c40547c5df68d0a7e944f462bb90750634b7f71f4f54abb1d2b86`.
- Validation adapter recomputation matches the trainer.
- The saved test predictions independently reproduce the reported late-mean
  AUC exactly.
- Sixteen causality probes show future-response leakage at numerical-noise
  scale (`1.1920928955078125e-07`) and substantial history-response effects.

## Resource Use

The GPU is a 6,144 MiB RTX 3060 Laptop. Across 17 monitored grid runs, the
maximum whole-card memory reached 5,635 MiB. The final d256/lr0.003 run
peaked at 5,508 MiB, with a 3,412 MiB pre-existing baseline and about
2,096 MiB incremental use.

WSL exposes 7,868 MiB RAM and 2,048 MiB swap. The maximum WSL memory in use
was 5,372 MiB. The reported process-tree RSS maximum of 8,943 MiB is a sum
of parent and child RSS and double-counts shared pages; it is not a physical
memory measurement.

GPU numbers are whole-device samples and include desktop/unrelated processes.
The first three d64 paper-head runs predate the resource sampler. All 20
configurations have training logs, but resource JSON exists for 17. No
Mamba4KT training, evaluation, or audit process remains running.

## Reproduction Boundary For The Paper

Allowed wording:

> We evaluate the third-party UniKT implementation of Mamba4KT on our unified
> AS09 question-window protocol. On fold 0 with seed 42, it obtains test AUC
> 0.7771, 0.0044 below the official-source ASIKT adaptation (0.7815).

Do not write:

> We reproduce the official Mamba4KT paper implementation and obtain 0.7771.

Do not compare `0.7771` directly with the `0.7978` paper number without
reproducing that number on the identical split and metric.

## Artifact Index

- `SOURCE_PREFLIGHT.md`: full source, protocol, result, audit, and resource
  record.
- `final_summary.json`: compact machine-readable final result.
- `grid_unikt/results.json`: 12 UniKT-head configurations.
- `grid_unikt/paper_head_result.json` and
  `grid_unikt/paper_head_results.json`: 4 paper-head d64 configurations.
- `grid_unikt/paper_head_d256_results.json`: 4 paper-head d256
  configurations.
- `evaluation_d256_lr0.003/result.json`: final UniKT-head evaluation.
- `evaluation_paper_d256_lr0.0001/result.json`: final paper-head evaluation.
- `audits/d256_lr0.003.json`: final UniKT-head independent audit.
- `audits/paper_d256_lr0.0001.json`: final paper-head independent audit.
