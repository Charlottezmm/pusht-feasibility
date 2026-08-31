# Frozen three-path formal evaluation

- Date: 2026-08-31 (Asia/Singapore)
- Stage / gate: Gate B formal paired evaluation
- Evidence status: `PASS` for completeness, pairing, and reproducibility
- Hypothesis status: not supported by the observed primary metric
- Repository base commit at runtime: `3c73d1c340fef63c9c5be0521bb1116c5694f51f`
- Raw output: `runs/2026-08-31-three-path-formal-evaluation-v0.1`
- Claim boundary: bounded simulator-instrumented rule evaluation only

## Question and primary comparison

Under the frozen controller, probe, threshold, budget, and simulator settings, how does final
coverage change when the probe response is allowed to enter the later-action rule?

The primary paired quantity was frozen as:

```text
delta = final_coverage(probe-adjust) - final_coverage(probe-no-adjust)
```

`fixed` was an auxiliary reference because comparing either probe path with `fixed` also changes
whether the dedicated probe is executed. The formal manifest contained evaluation seeds `20..29`,
settings `damping={0.0,1.0}`, and paths `fixed / probe-no-adjust / probe-adjust`: 60 planned episodes
and 20 primary pairs. Calibration seeds `100..109` were excluded.

## Frozen identity and environment

- Frozen `tau`: `73.6528373524437`
- Calibration summary SHA-256: `3781772ff24a17b352ed176b1215ad34cfcf14fd101d47c9758ffd48f93333b9`
- Checkpoint weights SHA-256: `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`
- Compatibility-config SHA-256: `188568e0cb5c28188bf3ea411d88fbb8e9287843840bad606ea0edb4db297c11`
- Protocol SHA-256: `c389226f824539b2c387233c2b05d8c331145b01bcd73a076754e6ef77ab4502`
- Wrapper SHA-256: `38664873ba9bc311e3c64ced403313f735a287b21c25ad10628877a2bd178577`
- Formal evaluator SHA-256: `ee4cdb384cf35d774cec5dabaa9467018c3cbd91a1a2af762fd5cae5dcfd55e6`
- macOS `26.6.2` (`arm64`); Python `3.11.1`; CPU, no AMP
- Environment: `gym_pusht/PushT-v0`, `pixels_agent_pos`, maximum 300-step budget
- Dependency readback after the run: `pip check` reported no broken requirements

## Procedure and operational note

The output directory did not exist at preflight. Set the controller Python and pinned checkpoint
locations for the local machine, then reproduce the frozen invocation with:

```bash
PUSHT_CONTROLLER_PYTHON="${PUSHT_CONTROLLER_PYTHON:?set controller Python}" \
PUSHT_CHECKPOINT_DIR="${PUSHT_CHECKPOINT_DIR:?set pinned checkpoint directory}" \
"$PUSHT_CONTROLLER_PYTHON" \
  experiments/three_path_evaluator.py \
  --checkpoint-dir "$PUSHT_CHECKPOINT_DIR" \
  --compat-config-dir runs/2026-08-29-controller-gate/compat-config \
  --calibration-dir runs/2026-08-31-probe-response-calibration-v0.1 \
  --device cpu \
  --output-dir runs/2026-08-31-three-path-formal-evaluation-v0.1
```

The ignored raw config retains the exact machine-local executable and artifact locations used for
this run.

The evaluator validated each completed three-path set before continuing. While assessing whether
the laptop needed to be packed, the same evaluator process was briefly paused with `SIGSTOP` and
resumed with `SIGCONT` after about 20 seconds. No episode, manifest, threshold, output directory, or
process identity was restarted or replaced. This can slightly inflate wall-clock runtime but does
not change the recorded action/state sequence or primary metrics.

## Primary raw results

| Seed | damping | no-adjust final | adjust final | `delta` | probe `x` |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.0 | 0.930006 | 0.950816 | +0.020810 | 6.402866 |
| 21 | 0.0 | 0.955014 | 0.950479 | -0.004535 | 62.425561 |
| 22 | 0.0 | 0.952356 | 0.958493 | +0.006137 | 58.511856 |
| 23 | 0.0 | 0.897179 | 0.952869 | +0.055690 | 25.447925 |
| 24 | 0.0 | 0.931968 | 0.428961 | -0.503007 | 58.452223 |
| 25 | 0.0 | 0.967607 | 0.956485 | -0.011122 | 49.614551 |
| 26 | 0.0 | 0.959871 | 0.967522 | +0.007651 | 31.098374 |
| 27 | 0.0 | 0.924413 | 0.970142 | +0.045729 | 56.448619 |
| 28 | 0.0 | 0.950757 | 0.926965 | -0.023792 | 58.474242 |
| 29 | 0.0 | 0.596713 | 0.000000 | -0.596713 | 44.367598 |
| 20 | 1.0 | 0.285832 | 0.000000 | -0.285832 | -3.853698 |
| 21 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 139.522056 |
| 22 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 153.231846 |
| 23 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 52.149515 |
| 24 | 1.0 | 0.526442 | 0.526442 | +0.000000 | 155.553874 |
| 25 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 98.553816 |
| 26 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 69.777322 |
| 27 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 142.477435 |
| 28 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 147.587978 |
| 29 | 1.0 | 0.000000 | 0.000000 | +0.000000 | 3.190486 |

## Independently recomputed aggregates

| Scope | pairs | mean `delta` | median | sample SD | range | positive / zero / negative |
|---|---:|---:|---:|---:|---:|---:|
| `damping=0.0` | 10 | -0.100315 | +0.000801 | 0.239185 | [-0.596713, +0.055690] | 5 / 0 / 5 |
| `damping=1.0` | 10 | -0.028583 | 0.000000 | 0.090388 | [-0.285832, 0.000000] | 0 / 9 / 1 |
| pooled descriptive | 20 | -0.064449 | 0.000000 | 0.179787 | [-0.596713, +0.055690] | 5 / 9 / 6 |

All 20 probes were valid, so the labeled valid-probe-only diagnostic equals the declared primary
set. No pair was removed because its `delta` was zero or negative.

Auxiliary path means provide context but not response-effect attribution:

| damping | fixed mean final | no-adjust mean final | adjust mean final | fixed successes | no-adjust successes | adjust successes |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.959601 | 0.906589 | 0.806273 | 10/10 | 5/10 | 7/10 |
| 1.0 | 0.000000 | 0.081227 | 0.052644 | 0/10 | 0/10 | 0/10 |

## Independent evidence audit

- All 60 manifest and episode identities matched exactly; all episode statuses were `succeeded`.
- The 60 step logs contained 15,406 finite rows. Twenty-two episodes terminated early only after
  success; the remaining 38 used the full 300-step budget and ended by environment truncation.
- All step-log, episodes, manifest, summary, calibration, checkpoint, config, protocol, wrapper,
  and evaluator hashes matched their persisted identities.
- All 20 pair-validator files passed all 18 checks: shared identity/budget/prefix/probe states and
  measurement, complete logs, finite valid actions, response isolation, and rule-consistent branch use.
- The calibration artifact independently read back as 20/20 valid attempts, 500 finite rows, and
  the same recomputed `tau`.
- Independent recomputation from `episodes.jsonl` matched every per-setting and pooled primary
  aggregate in `summary.json`.

## Interpretation and decision

The evidence-validity Gate is `PASS`: the planned data are complete, paired, hash-consistent, and
usable. A negative or zero primary result is not a reason to repeat a valid run.

The directional hypothesis that the frozen `probe-adjust` rule improves final coverage over the
same-probe `probe-no-adjust` control is **not supported**. Mean `delta` was negative in both tested
settings. Nominal results were heterogeneous and included two large negative pairs; the declared
shift condition was mostly tied at a final-coverage floor, with one negative pair and no positive
pairs. The data therefore do not support saying `probe-adjust` is better.

This is a bounded descriptive result for one checkpoint, one frozen response rule, 10 paired seeds
per setting, privileged simulator `block_pose` measurement, and the declared PushT simulator
conditions. It does not establish general adaptation, robustness, a causal physical mechanism,
real friction identification, real-robot performance, or failure of every possible adjustment
rule. No post-hoc statistical test or threshold tuning was added.

The next scientific action requires a new, predeclared design rather than reusing these evaluation
seeds to tune `tau` or `scale`. Before that design is chosen, Charlotte's learner-owned result
interpretation and closed-book boundary check remain separate from this runtime evidence Pass.
