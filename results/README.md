# Results

All files are CSV. `A` (or `A_base`, `A_probe`) is the area under the target-hitting
ECDF of one run, in `[0, 1]`; multiply by 100 for points.

## Per-run results

| File | Produced by | Rows describe |
|---|---|---|
| `exp1_anchor_validation.csv` | `exp1_anchor_validation.py` | anchors on the 14 canonical functions: `optimizer, function, D, instance, probe, run, A_base, A_probe, delta` |
| `exp1b_dyadic_scaling.csv` | `exp1b_dyadic_scaling.py` | same format, dilation by `1e3` and `2^±10` |
| `exp2_audit.csv` | `exp2_metabox_audit.py --family audit` | one episode of a pretrained agent: `family, agent, fid, instance, probe, seed, A, best_final, real_fe, seconds, error, b0..b30` |
| `exp2_ood.csv` | `exp2_metabox_audit.py --family ood` | same format, held-out functions, untransformed |
| `exp3_audit.csv`, `exp3_ood.csv` | `exp3_anchors_on_bbob.py` | anchors on the MetaBox instances: `family, agent, fid, instance, probe, seed, A` |
| `exp4_frame.csv` | `exp2_metabox_audit.py --family frame` | agents with the frame layer of the harness |
| `exp5_dose.csv` | `exp2_metabox_audit.py --family dose` | agents under realistic domains |
| `exp6_harness_anchors.csv` | `exp6_harness_anchors.py` | anchors with both harness layers, experiment-1 format |
| `exp7_rank_ablation.csv` | `exp7_rank_ablation.py` | `optimizer, function, instance, mode, run, A` |

Columns of the agent files:

- `real_fe` — evaluations actually performed in the episode (not the agent's own counter)
- `best_final` — best canonical objective value reached (the optimum is 0)
- `b0 … b30` — best canonical objective value at the 31 logarithmically spaced budgets
- `seconds` — wall-clock time of the episode
- `error` — exception message if the episode raised one (such runs are scored as zero in
  the analysis; file-system paths are replaced by `<path>`)

## Derived tables

| File | Produced by |
|---|---|
| `table_validation_scores.csv` | `make_figures.py` — violation `V`, signed shift, pathwise defect, `p`, `q` for the anchors |
| `table_audit_scores.csv` | `make_figures.py` — the same for all 26 methods on the MetaBox training functions |
| `table_law.csv` | `make_figures.py` — aggregate violation, mean ranks, rank gap `G`, mean performance |
| `table_instances.csv` | `instance_report.py` — optimum radius, bias and admissibility of each MetaBox instance |
| `table_dose.csv` | `analyze_interventions.py` — realistic domains |
| `table_frame_intervention.csv` | `analyze_interventions.py` — agents before / after the frame layer |
| `table_harness_anchors.csv` | `analyze_interventions.py` — anchors without (`_b`) / with (`_a`) the harness |
| `table_harness_cost.csv` | `analyze_interventions.py` — in-distribution cost of the harness on the anchors |
