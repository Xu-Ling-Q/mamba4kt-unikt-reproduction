# Mamba4KT Unified-Protocol Reproduction Record

Date: 2026-10-30

> Update: a later paper-faithful reconstruction was completed from UniKT's
> initial Mamba4KT commit `553c344`. Its frozen AS09 validation-champion test
> AUC is `0.7672619`. See `PAPER_FAITHFUL_REPORT.md` and schema version 2 in
> `final_summary.json`. The `0.7771055` result below remains a valid
> third-party additive-embedding comparison, but it is not the paper-faithful
> reconstruction.

Status: COMPLETE for the third-party UniKT implementation under the project's
frozen AS09 question-window protocol. The original Mamba4KT author source was
not found, so this is not an official-author-source end-to-end reproduction.

The machine-readable summary is `final_summary.json`; the human-readable
summary is `FINAL_REPORT.md`.

## Scope And Source Boundary

- Target model: Mamba4KT from the user-specified `szhhwh/UniKT` repository.
- Source: <https://github.com/szhhwh/UniKT/tree/main/model/Mamba4KT>.
- Pinned upstream commit: `0493bfbba31373a3058f287cd6625ed10d48895e`.
- This checkout contains the third-party UniKT implementation, not an
  author-released Mamba4KT end-to-end repository.
- The original paper is arXiv `2405.16542`. It reports ASSISTments 2012,
  ASSISTments 2017, and Eedi, not the project AS09 split.
- Public searches did not identify an author-maintained Mamba4KT source
  archive. That finding cannot prove that no private, removed, renamed, or
  unindexed copy exists.
- Earlier local evidence that called `Yumo216/MCKT` commit
  `18937e7ab6580e9c56f81f165f6842cfd7983ac7` an author source was rejected:
  it belongs to a different paper and repository.

The defensible label for the result below is:

> Third-party UniKT Mamba4KT implementation evaluated on the project's
> unified AS09 fold-0 question-window protocol.

It is not:

> Official Mamba4KT author-source reproduction.

## Local Source State

The pinned UniKT tree was modified only for the experiment adapter and the
paper-head ablation:

- `Mamba4KT_model.py`: added `prediction_head="unikt"|"paper"`.
  - `unikt`: retains the UniKT implementation's `concat(f, q_next)` head.
  - `paper`: applies the paper's `Sigmoid(fW+b)` head without `q_next`.
- `Mamba4KT_trainer.py`: exposes and forwards `prediction_head`.
- `Mamba4KT_data.py`: permits a bounded test-loader worker count through
  `KT_DATALOADER_WORKERS`.
- `utils/config/data_config.py`: resolves `"auto"` workers with the same
  environment-variable cap instead of passing `"auto"` to PyTorch.

Current file SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `Mamba4KT_model.py` | `f2f0756204fc7b013e00509ecb3491f2e7b27d67d7850d9008d78358cd5c2d69` |
| `Mamba4KT_data.py` | `e9704964fbf6dcf74bd23eca6f553f1212d00e96dd4a7c8e1a27f07fc9c27811` |
| `Mamba4KT_trainer.py` | `51281b91600158ececc7765f8d1dae2db1eb99b6dcc1691251a5100f8cd5578f` |

The old `unikt` checkpoint loads strictly after the new configuration fields
were added. The `paper` head was validated and evaluated as an explicit
implementation-sensitivity ablation, not as proof of author-source identity.

## Unified Data Protocol

- Dataset: project `assist2009_corrected_collapsed` (`AS09`).
- Fold: 0.
- Train: folds 1-4, 3,016 sequences.
- Validation: fold 0, 741 sequences.
- Test: 904 held-out sequences.
- Seed: 42.
- Maximum sequence length: 200.
- Concepts: 123.
- Converted sequences: 3,757; fold counts are `741/741/750/778/747`.
- Train, validation, and test users are disjoint.
- Test question-window rows: 61,882.
- Aggregated question-window groups: 52,378.
- Evaluation chain:
  `evaluate_question -> group_fusion -> effective_fusion -> late_fusion`.
- Reported metric: pyKT `windowauclate_mean`.
- Frozen question-window CSV SHA-256:
  `26cc650b450c40547c5df68d0a7e944f462bb90750634b7f71f4f54abb1d2b86`.
- ASIKT and Mamba4KT test `qidxs` and targets are identical element by
  element for all 61,882 rows.

## Hyperparameter Search

The search matches the paper's candidate hidden sizes and learning rates:
`hidden_size in {64,128,256}` and
`learning_rate in {0.003,0.002,0.001,0.0001}`. All configurations use batch
size 64, five Mamba blocks, Adam, and validation-AUC checkpoint selection.

Two heads were searched:

1. `unikt`: the UniKT implementation, including the extra `q_next`
   concatenation.
2. `paper`: the paper's Eq. 10 prediction head without `q_next`.

There are 20 completed configurations: 12 `unikt` and 8 `paper` (`paper` was
searched at hidden size 64 first, then hidden size 256 because d64 was
clearly weaker).

## Final Results

| Variant | Configuration | Validation AUC | Test AUC | Test ACC |
| --- | --- | ---: | ---: | ---: |
| Old UniKT baseline | d128, lr 0.001 | 0.8248046 | 0.7422034 | 0.7213525 |
| UniKT head candidate | d64, lr 0.003 | 0.8461483 | 0.7719114 | 0.7289320 |
| **UniKT head winner** | **d256, lr 0.003** | **0.8491687** | **0.7771055** | **0.7321394** |
| Paper head, d64 best | d64, lr 0.003 | 0.8126328 | 0.7243584 | 0.7097255 |
| **Paper head winner** | **d256, lr 0.0001** | **0.8155762** | **0.7295355** | **0.7157394** |

The final UniKT-head model improves test AUC by `0.0349021` (3.49 percentage
points) over the old d128/lr0.001 UniKT baseline. The paper-head winner is
lower by `0.0475699` test AUC (4.76 percentage points), while its validation
gap is 3.36 percentage points. The `q_next` concatenation is therefore a
material implementation difference between the UniKT head and the paper head.

## Same-Protocol ASIKT Comparison

The local ASIKT reference is `0.7815264027514466` window AUC and
`0.7362824086448508` window ACC. The final Mamba4KT UniKT-head result is:

- AUC difference: `-0.0044209335845437625` (-0.4421 percentage points).
- ACC difference: `-0.004142960785062377` (-0.4143 percentage points).

The user-cited `0.7978` is not the result of this frozen local AS09 protocol
and must not be compared directly with `0.7771055`.

ASIKT source fidelity is stronger than Mamba4KT source fidelity here. The
ASIKT run copies release commit
`db8eb4cac6bcf9138b2179e8f279b93a49de2832` from
<https://github.com/mrsser/ASIKT> byte-for-byte and uses its architecture,
masked BCE, optimizer, and early stopping through a narrow pyKT adapter.
However, it runs on the project AS09 split, not the author's embedded split,
and uses batch size 32 because batch 48 exhausted the 6 GB GPU. Strictly
speaking, it is an official-source unified-protocol reproduction, not an
unchanged run of the author's own split and launcher.

## Independent Audit

Final UniKT-head audit:

- Checkpoint SHA-256:
  `9cd9c57d2f596b0a71a71d08f29cc3e984e42118c7a8c83ae14f0cbae5127907`.
- Best epoch: zero-based 3.
- Recorded validation AUC: `0.849168692687067`.
- Recomputed validation AUC: `0.8491686942619963`.
- Validation adapter maximum absolute difference: `0.0`.
- Real-window alignment against raw model outputs: true.
- Causality probes: 16.
- Maximum future-response effect: `1.1920928955078125e-07`.
- Maximum history-response effect: `0.33653759956359863`.
- Independent late-mean recomputation equals the pyKT result exactly.

Paper-head audit:

- Checkpoint SHA-256:
  `a99ffbf640d23242d6fc6a07e7077d01e1a2bf882d667364c0f73fbcc5006d66`.
- Best epoch: zero-based 8.
- Recomputed validation AUC: `0.81557623830256`.
- Validation adapter maximum absolute difference: `0.0`.
- Maximum future-response effect: `1.1928955078125e-07`.
- Maximum history-response effect: `0.3381367325782776`.

## Resource Envelope

The host has a 6,144 MiB RTX 3060 Laptop GPU. WSL exposes 7,868 MiB RAM and
2,048 MiB swap. All 20 configurations have logs; 17 later configurations
have formal resource JSON. The first three d64 paper-head runs predate the
resource sampler.

Across the 17 monitored configurations:

- Maximum whole-GPU memory in use: `5,635 MiB / 6,144 MiB` (91.7%).
- Maximum WSL memory in use: `5,372 MiB / 7,868 MiB` (68.3%).
- Maximum summed process-tree RSS: `8,943 MiB`.
- The process-tree RSS is larger than WSL RAM because it sums parent and
  child RSS and double-counts shared pages. It is not the physical-memory
  peak.
- GPU sampling measures the entire card, including desktop and unrelated
  host processes, not only the training process.

For the final d256/lr0.003 UniKT-head run:

- Whole-GPU baseline: `3,412 MiB`.
- Whole-GPU peak: `5,508 MiB`.
- Incremental GPU use during that run: about `2,096 MiB`.
- WSL memory peak: `5,159 MiB`.
- Training runtime: `225.768` seconds.
- Sampled points: 110.

For the final paper-head d256/lr0.0001 run, whole-GPU peak was `4,793 MiB`
and WSL memory peak was `5,061 MiB`. Final evaluation and audit processes
have exited; no Mamba4KT training is currently running.

## Artifacts

- Full UniKT-head grid: `grid_unikt/results.json`.
- Paper-head d64 grid: `grid_unikt/paper_head_results.json` plus
  `grid_unikt/paper_head_result.json`.
- Paper-head d256 grid: `grid_unikt/paper_head_d256_results.json`.
- Final UniKT-head evaluation: `evaluation_d256_lr0.003/result.json`.
- Final UniKT-head audit: `audits/d256_lr0.003.json`.
- Final paper-head evaluation:
  `evaluation_paper_d256_lr0.0001/result.json`.
- Final paper-head audit: `audits/paper_d256_lr0.0001.json`.
- Resource records: `grid_unikt/resources/*.json`.

## Manuscript Use

The result may be included as an honest same-protocol empirical comparison,
provided the paper states all of the following:

1. The implementation is third-party UniKT, not official author code.
2. The split is the project AS09 fold-0 extension, not the paper's original
   AS12/AS17/Eedi benchmark.
3. This is one fold and one seed.
4. The metric is frozen question-window late-mean AUC/ACC, not a
   paper-table metric.
5. `0.7771055` must not be compared directly with a non-unified `0.7978`.
6. The ASIKT comparator is official-source but protocol-adapted to the same
   split and uses batch size 32 rather than its official default 48.

A suitable result-table label is:

> Mamba4KT (third-party UniKT implementation; unified AS09 protocol)
