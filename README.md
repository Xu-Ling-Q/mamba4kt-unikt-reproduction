# Mamba4KT Reproduction (Unified Protocol)

Reproduction of Mamba4KT on the project's **unified question-window
protocol**, so the result is directly comparable with other models evaluated
on the same data split, checkpoint rule, and metric (see the ASIKT
reproduction in
[asikt-official-reproduction](https://github.com/Xu-Ling-Q/asikt-official-reproduction)).

Two datasets are included in this repository:

- **ASSIST2009** (`results/`): the primary unified-protocol comparison number.
- **Algebra2005** (`algebra2005/`): a second-dataset single-fold evaluation
  on the same pyKT question-window protocol (full details in
  `algebra2005/FINAL_REPORT.md`).

## Important attribution note

**No official author-released source for the Mamba4KT paper was found.**
This work is an evaluation of the **third-party UniKT implementation**
(https://github.com/szhhwh/UniKT, commit
`0493bfbba31373a3058f287cd6625ed10d48895e`), pinned and evaluated on the
frozen unified AS09 protocol. The `paper-faithful` variant is a
**paper-logic reconstruction** built from the model files of UniKT's
*initial* commit (`553c3448685f33b84637601767f309a2bdace8e6`).

It must **not** be presented as "official Mamba4KT end-to-end
reproduction". It is a valid, honest comparison number only on the unified
protocol (details in `results/FINAL_REPORT.md`).

## Result (ASSIST2009, fold 0, seed 42, question-window)

Reported metric: `windowauclate_mean`
(evaluation chain `evaluate_question -> group_fusion -> effective_fusion -> late_fusion`).

| Variant | Configuration | `windowauclate_mean` | Best val AUC |
| --- | --- | ---: | ---: |
| **UniKT head (best)** | d256, lr 0.003, epoch 3 | **0.7771054692** | 0.8491687 |
| Paper-faithful reconstruction | d128, lr 0.003, separate_qa | 0.7672618665 | 0.8434410 |
| Paper head | d256, lr 0.0001, epoch 8 | 0.7295355213 | 0.8155762 |
| Old UniKT baseline | d128, lr 0.001 | 0.7422033840 | 0.8248046 |

Same-protocol references:

| Model | `windowauclate_mean` |
| --- | ---: |
| Mamba4KT (UniKT head, this repo) | 0.7771055 |
| Official ASIKT (unified protocol) | 0.7815264 |
| Project `r316_transfer_ledger` | 0.7881642 |

## Result (Algebra2005, fold 0, seed 42, question-window)

Reported metric: `windowauclate_mean` (same pyKT question-window late-mean
chain as the ASSIST2009 run above).

| Champion configuration | Value |
| --- | ---: |
| d_model / n_blocks / d_state / d_conv / expand | 256 / 5 / 16 / 4 / 2 |
| learning_rate / batch_size / dropout | 0.002 / 64 / 0.1 |
| prediction_head | unikt (`separate_qa=true`) |
| Total parameters | 3,063,611 |
| Best val AUC (epoch 2, zero-based) | 0.9156418 |
| Test `windowauclate_mean` | **0.8140126** |
| Test `windowacclate_mean` | 0.8063671 |
| Test `windowaucconcepts` | 0.8148890 |
| Question-window rows / groups / targets | 125,363 / 93,481 / 135,246 |
| Group bootstrap interval (approx.) | [0.810911, 0.817157] |

- Hyperparameters selected on validation AUC only (12 candidates, test never
  used for selection); test evaluated once on the champion checkpoint.
- The champion checkpoint SHA256, every grid/eval config, and all audit
  manifests are recorded in `algebra2005/SHA256SUMS.json` and
  `algebra2005/final_summary.json`.
- **Do not compare 0.8140 (Algebra2005) against the ASSIST2009 numbers above
  or against ASIKT's 0.7815**: different dataset and test file. A matched
  ASIKT-on-Algebra2005 run is required for any same-protocol cross-model
  claim on this dataset.

## Protocol

- Dataset: `assist2009_corrected_collapsed` (17,737 questions, 123 concepts)
- Fold 0: train folds 1-4 (3,016 sequences), validation fold 0 (741 sequences),
  test (904 sequences); max sequence length 200
- Test question-window rows: 61,882; groups: 52,378
- Train/validation/test users are disjoint; Mamba4KT and ASIKT share
  identical test `qidxs` and targets for all 61,882 rows
- Frozen test question-window CSV SHA256:
  `26cc650b450c40547c5df68d0a7e944f462bb90750634b7f71f4f54abb1d2b86`

## Audit

- 16 causality probes: max future-response effect `1.1920928955078125e-07`
  (numerical noise), substantial history-response effect
- Validation AUC independent recomputation matches the trainer
  (difference `1.6e-9`)
- Independent test AUC from saved arrays matches exactly
- Initial-commit and profiled forward passes are identical (max abs diff 0.0)

## Repository layout

```
source/
  unikt_current/           Mamba4KT files from the evaluated UniKT commit
                           (Mamba4KT_model/data/trainer, Mamba4KTQ_data/trainer)
  unikt_initial_553c344/   Mamba4KT files from UniKT's initial commit
                           (basis of the paper-logic reconstruction)
reproduction_scripts/
  prepare_unikt_data.py             convert unified AS09 to UniKT format
  run_unikt_grid.py                 hyperparameter grid
  audit_unikt_protocol.py           protocol/causality audit
  evaluate_pykt_question_window.py  unified pyKT question-window evaluation
  mamba4kt_fold0_seed42.yaml        run config (UniKT head)
  mamba4kt_paper_faithful_fold0_seed42.yaml  run config (paper faithful)
results/
  final_summary.json        machine-readable final result
  FINAL_REPORT.md           full report (UniKT-head evaluation)
  PAPER_FAITHFUL_REPORT.md  paper-faithful reconstruction report
  SOURCE_PREFLIGHT.md       source/attribution preflight
  protocol_audit.json       protocol audit record
  train_console.log, evaluate_console.log
  evaluation_*/result.json  per-variant unified evaluation (full metrics)
  grid_unikt/, grid_paper_faithful_run1/   grid search results
  audits/                   per-run independent audits
algebra2005/
  final_summary.json        machine-readable Algebra2005 result
  FINAL_REPORT.md           full Algebra2005 report (incl. data provenance)
  SHA256SUMS.json           SHA256 of source, checkpoints, and test windows
  mamba4kt_algebra2005_fold0_seed42.yaml    run config (champion)
  al05_dataset_smoke.json, data_count_audit.json
  scripts/                  data prep, grid, evaluation, and audit scripts
  grid_al05/                grid results (12 candidates) + per-run resources
  evaluation_champion_d256_lr0.002/result.json   champion unified eval
  evaluation_d128_lr0.001/result.json            secondary run
  pykt_data/                preprocessing manifests + alignment audit
  unikt_data/               UniKT adapter manifest
```

## Environment

Python 3.12.3, PyTorch 2.6.0+cu118, CUDA 11.8, mamba-ssm 2.3.1,
causal-conv1d 1.6.2.post1, NVIDIA RTX 3060 Laptop (6 GB).

Model weights (`best_model.pth`), the full UniKT checkout, and per-row
prediction dumps are intentionally not included to keep this repository
light; the runs are reproducible from the scripts and configs above, and
every reported number is independently auditable from the JSON artifacts.

## License

The UniKT source files under `source/` belong to the UniKT project
(https://github.com/szhhwh/UniKT) and carry their own license.
