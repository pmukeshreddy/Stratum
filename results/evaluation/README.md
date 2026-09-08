# Buffalo evaluation results

The supported comparison holds Astra XHigh fixed:

| Harness | Model | Reported score |
| --- | --- | ---: |
| ARC Standard harness | Astra XHigh | ≈ 59% |
| Buffalo harness | Astra XHigh | ≈ 81% |

**59 → 81: approximately +22 percentage points.** Buffalo improves the same
underlying model by replacing the ARC Standard harness.

These are reported, rounded results supplied by the project owner; their underlying
score artifacts and the ARC Standard runner are not included in this checkout.

Fresh Buffalo executions write separate `arc-buffalo-*` directories containing
actual measured scores, official evaluator output, trajectories, resource accounting,
and pinned source/environment identities. Those large local directories are ignored
by Git.

[Setup and reproduction instructions](../../docs/evaluation.md).
