# PushT Robot-Learning Feasibility Project

An undergraduate, learning-first research project built around the
[`gym-pusht`](https://github.com/huggingface/gym-pusht) environment.

**Current status: the frozen three-path evaluation is complete; its directional improvement
hypothesis is not supported.** The latest raw experiment is
`runs/2026-08-31-three-path-formal-evaluation-v0.1`, documented in the
[formal evaluation record](experiments/2026-08-31-three-path-formal-evaluation.md).
No new intervention or scientific question has been selected in this repository.

## Current question and start here

With controller, probe, threshold and budget frozen, does using the measured probe response in the
later-action rule improve final coverage relative to `probe-no-adjust`? The primary metric is
`final_coverage(probe-adjust) - final_coverage(probe-no-adjust)`; `fixed` is auxiliary.
Mean paired differences were negative in both tested settings. This is a bounded simulator result.

Use the [start and review workflow](experiments/WORKFLOW.md) before launching anything. It provides
an offline preflight, independent raw-log readback, failure handling and explicit command evidence.
The [2026-09-05 audit](experiments/2026-09-05-workflow-audit.md) recomputed the current result without
running inference or training. The older Gates 1–5 below are historical context, not a new work queue.

## Learning objectives

By completing the first research cycle, the project owner should be able to:

1. explain the PushT agent–environment loop in her own words;
2. create and reproduce an isolated Python environment;
3. read and explain each line of a minimal Gymnasium interaction loop;
4. distinguish an environment smoke test from policy evaluation;
5. define independent variables, dependent metrics, controls, seeds, and stopping conditions;
6. run a baseline and a controlled comparison without changing multiple factors at once;
7. preserve failures and write conclusions that stay within the evidence.

Code or analysis produced with AI assistance is accepted as project evidence only after the project
owner can explain it, run it, inspect its outputs, and reproduce the result.

## Standard experiment workflow

| Gate | Work | Required evidence | Status |
| --- | --- | --- | --- |
| 0. Question | Define the question, hypothesis, variables, controls, and stop conditions | Reviewed protocol | Pass |
| 1. Environment | Install exact dependencies and run the minimum environment loop | Commands, versions, seed, output, clean-process rerun | Pass |
| 2. Interface | Inspect observation, action, reward, termination, and rendering | Annotated trace and owner explanation | Pass |
| 3. Baseline | Define and run a simple baseline under a fixed evaluation protocol | Metrics across declared seeds/episodes | Pass |
| 4. Controlled comparison | Change one motion-resistance factor while preserving controls | Comparable result table and diagnostics | Pass |
| 5. Interpretation | Analyze failures, uncertainty, and limitations | Feasibility memo and next decision | Pass — project Pivot |

Every gate ends with one of three decisions:

- **Pass** — the evidence and understanding checks both pass;
- **Patch** — a specific gap is repaired, then checked again;
- **Repeat** — the procedure is rerun because the evidence is not reliable.

The detailed contract is in [`PROTOCOL.md`](PROTOCOL.md). New experiment records start from
[`experiments/TEMPLATE.md`](experiments/TEMPLATE.md).

## Verified pilot

The first recorded pilot uses a deliberately limited block-chasing heuristic: each action targets
the block center. It is useful for checking the interface and logging path, but it does not use the
goal pose or constitute a learned policy.

For seed 0 with `damping=1.0`, the 20-step pilot finished with `success=False`, final coverage `0.0`,
and return `1.555275312143944`. Two new processes produced the same discrete trace and numerically
equivalent floating-point values (`max_abs_diff=5.551e-17`, declared `atol=1e-12`). The environment
procedure passed; the heuristic did not solve the task. See the
[`experiment record`](experiments/2026-08-20-stage1-heuristic-pilot.md) for the full evidence and
limitations.

## Verified baseline and controlled comparison

The Gate 3 baseline compared seeded random target-position actions with a block-chasing heuristic
across paired seeds `0..9`, `damping=1.0`, and 300-step episodes. Both policies had mean final
coverage and success rate `0.0`, so the directional hypothesis was not supported. See the
[`baseline record`](experiments/2026-08-25-random-vs-block-chasing-baseline.md).

The Gate 4 comparison replayed identical seeded open-loop action arrays across explicit
`damping={1.0, 0.7, 0.4}` settings for seeds `0..9`. All settings again had mean final coverage and
success rate `0.0`, while independently validated trajectory diagnostics differed from the
`damping=1.0` reference. This supports a simulator motion-response difference, not improved task
performance, adaptation, or real-world friction claims. See the
[`motion-resistance record`](experiments/2026-08-26-motion-resistance-comparison.md).

## Historical Gate 5 decision

The first cycle ended with a project-level **Pivot**, not a direct Proceed or Stop. The execution,
pairing, logging, and simulator-manipulation chain is credible, but the predeclared final-coverage
metric remained at a complete floor. The next bounded step is to predeclare net block displacement
and trajectory path length, then repeat the matched comparison. This measurement pivot does not
yet test probe-based adaptation or controller effectiveness. See the
[`feasibility memo`](experiments/2026-08-27-feasibility-memo.md).

## Reproduce the historical heuristic pilot

These historical installation/run commands were not re-executed in the 2026-09-05 audit.
`requirements-lock.txt` covers the heuristic environment, not the Diffusion Policy controller.
Use Python 3.11 and create the virtual environment outside Git tracking:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python experiments/heuristic_pilot.py
```

The exact two-process comparison commands are preserved in the experiment record. Raw runs belong
under `runs/`, which is intentionally ignored by Git.

## Initial scope

The first cycle covers environment validation, interface inspection, a simple baseline, and one
controlled motion-resistance comparison. Simulator `damping` will be described as **motion
resistance**, not measured real-world tabletop friction.

The initial cycle excludes large-scale training, real-robot experiments, world models, broad
hyperparameter sweeps, and claims of algorithmic novelty.

## Repository structure

```text
README.md                Project purpose, scope, and current stage
PROTOCOL.md              Stage gates and research-quality requirements
requirements-lock.txt    Exact Python dependency snapshot for the verified pilot
experiments/TEMPLATE.md  Blank record used for each experiment
experiments/heuristic_pilot.py
                         Fixed-seed 20-step interface pilot
experiments/2026-08-20-stage1-heuristic-pilot.md
                         Reviewed pilot record and evidence boundary
experiments/baseline_evaluator.py
experiments/2026-08-25-random-vs-block-chasing-baseline.md
                         Gate 3 paired baseline implementation and record
experiments/motion_resistance_evaluator.py
experiments/2026-08-26-motion-resistance-comparison.md
                         Gate 4 paired motion-resistance implementation and record
experiments/2026-08-27-feasibility-memo.md
                         Gate 5 evidence audit, project decision, and next-step boundary
tests/compare_numeric_logs.py
                         Numeric-tolerance comparator for independent logs
tests/test_motion_resistance_evaluator.py
tests/validate_motion_resistance_run.py
                         Unit and independent raw-output validation for Gate 4
```

Small, reviewed code and evidence records are tracked. Large datasets, checkpoints, and raw runs
remain outside Git.

## Research integrity

- A successful command is not automatically a successful experiment.
- A single run is not automatically a reproducible result.
- A smoke test is not evidence of policy quality.
- Plans, hypotheses, and expected behavior are labeled separately from observations.
- Failed and blocked runs remain part of the record.
- Conclusions state what the evidence supports and what remains unknown.

## License

This repository is licensed under the Apache License 2.0. Upstream projects retain their own
licenses and attribution.
