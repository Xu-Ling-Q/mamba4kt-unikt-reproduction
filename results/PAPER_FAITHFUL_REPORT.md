# Mamba4KT AS09 Paper-Faithful Reconstruction

Date: 2026-10-30

## Bottom Line

This run is the strongest reproducible Mamba4KT proxy currently available, but
it is still **not an official-author-source end-to-end reproduction**. No
public Mamba4KT source archive maintained by the paper authors was found.

The implemented profile follows the model mathematics of UniKT's first
Mamba4KT commit `553c3448685f33b84637601767f309a2bdace8e6`:

- Rasch embeddings from Eq. 4 using the joint `(concept, response)` table.
- Five Mamba blocks from Eq. 5.
- FFN from Eq. 9 and the shifted-question prediction head from Eq. 10.
- BCE plus Eq. 11 Rasch regularization.
- Next-item alignment with target-window causality checks.

The correct manuscript label is:

> Mamba4KT (paper-logic reconstruction from the initial UniKT implementation; unified AS09 protocol)

It must not be labeled as the authors' official implementation.

## Why This Profile

UniKT commit `553c344` explicitly states that it implements Eq. 4/5/9/10/11,
uses next-item alignment, and reports AS12 test AUC `0.7713` / ACC `0.7549`
against the paper's `0.7764` / `0.7543`. That is useful implementation
evidence, but it is third-party evidence rather than author code.

Later commit `83688be293da7304dbfdd1a8e28057d68f30ddd5` added the
`separate_qa` switch and made the default interaction embedding additive. The
earlier failed paper-head run also removed `q_next`, so it did not represent
the initial implementation. This run instead freezes:

```yaml
model:
  separate_qa: 1
  prediction_head: unikt
```

With those settings, the pinned model code is source-equivalent to the initial
model mathematics. The isolated check in
`audits/model_equivalence_initial_vs_profile.json` found identical state-dict
keys and values and a maximum forward-prediction difference of exactly `0.0`
for 828,363 parameters. This proves model-path equivalence, not full trainer
or data-loader identity.

## Protocol And Search

- Dataset: project `assist2009_corrected_collapsed` (AS09).
- Frozen split: fold 0, with 3,016 train, 741 validation, and 904 test sequences.
- Seed: 42; maximum sequence length: 200; batch size: 64.
- Optimizer: Adam; loss: `BCELoss`; rollout: up to 150 epochs.
- Early stopping: validation AUC, patience 10, minimum delta 0.
- Grid: hidden size `{64,128,256}` times learning rate
  `{0.003,0.002,0.001,0.0001}`.
- Model selection used validation AUC only.
- Test was run once for the frozen validation winner, `d128/lr=0.003`.
- Metric: pyKT `windowauclate_mean` over 61,882 rows and 52,378 groups.

All 12 configurations completed. The grid is recorded in
`grid_paper_faithful_run1/results.json`.

## Results

| Profile | Configuration | Validation AUC | Test AUC | Test ACC |
| --- | --- | ---: | ---: | ---: |
| ASIKT official source, unified adapter | official architecture; batch 32 | 0.8507975 | **0.7815264** | 0.7362824 |
| Current UniKT additive embedding plus `q_next` | d256, lr 0.003 | 0.8491687 | **0.7771055** | 0.7321394 |
| **Paper-faithful initial model path** | **d128, lr 0.003** | **0.8434410** | **0.7672619** | **0.7270037** |
| Paper Eq.10 head only, additive embedding | d256, lr 0.0001 | 0.8155762 | 0.7295355 | 0.7157394 |

The paper-faithful result is:

- `0.0098436` AUC below the current UniKT additive-embedding implementation.
- `0.0142645` AUC below the unified-protocol ASIKT result.
- `0.0051357` ACC below the current UniKT implementation.
- `0.0092787` ACC below unified-protocol ASIKT.

The initial commit's joint interaction embedding therefore does not recover
the stronger AS09 result. The result is an honest negative finding rather than
a failed run that should be replaced by test-set tuning.

## Can It Go Into The Paper?

Yes, but only as a same-protocol reconstruction baseline with the source
boundary stated explicitly. A defensible sentence is:

> We reconstructed Mamba4KT from the paper equations and the initial public
> UniKT implementation. Under our frozen AS09 question-window protocol, it
> obtained test AUC 0.7673 and ACC 0.7270, below our ASIKT adaptation at
> 0.7815 and 0.7363.

The cited ASIKT value `0.7978` cannot be placed in that comparison. It was not
produced on the same AS09 split, question-window targets, or evaluation chain.
The comparable local ASIKT value is `0.7815264`, not `0.7978`.

Do not write either of these claims:

> We reproduce the authors' official Mamba4KT source end to end.

> Mamba4KT is worse than ASIKT by 0.0305 because 0.7978 - 0.7673 = 0.0305.

## Integrity And Resources

- Checkpoint SHA-256:
  `89172ae95f59eac5b8b0d4b84a2473f75754a6d31344f3b6d061c27f7327d984`.
- Best validation epoch: zero-based 3.
- Recomputed validation AUC: `0.8434410367`.
- Validation adapter maximum absolute difference: `0.0`.
- Real-window alignment against raw outputs: passed.
- ASIKT/Mamba targets and question indices: identical for all 61,882 rows.
- Causality probes: 16.
- Maximum future-response effect: `1.1920928955078125e-07`.
- Maximum history-response effect: `0.3288009167`.
- Independent late-mean recomputation equals the pyKT result exactly.

All 12 runs used one DataLoader worker and two CPU threads. Across the grid:

- Maximum whole-GPU memory in use: `5,469 MiB / 6,144 MiB`.
- Maximum WSL memory in use: `5,234 MiB / 7,868 MiB`.
- Total serial training time: `2,890.585` seconds.
- No training, evaluation, or audit process remained afterward.

The whole-GPU value includes unrelated desktop processes. Process-tree RSS is
not used as physical memory because it double-counts shared pages.

## Artifacts

- `mamba4kt_paper_faithful_fold0_seed42.yaml`: frozen profile.
- `grid_paper_faithful_run1/results.json`: complete 12-run grid.
- `evaluation_paper_faithful_d128_lr0.003/result.json`: frozen test result.
- `audits/paper_faithful_d128_lr0.003.json`: split, alignment, and causality audit.
- `audits/model_equivalence_initial_vs_profile.json`: source-level model check.
- `UniKT_initial_553c344/`: detached worktree at the initial Mamba4KT commit.
