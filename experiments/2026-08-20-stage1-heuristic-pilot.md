# Experiment: Fixed-seed heuristic PushT pilot

## Metadata

- Date: 2026-08-20 (Asia/Shanghai)
- Stage / gate: Gate 1 environment and Gate 2 interface pilot
- Status: pass
- Related commit: the commit that adds this record

## Question

Can the clean PushT environment execute a fixed 20-step heuristic pilot from two new processes,
produce the expected interface outputs, render correctly, close cleanly, and reproduce its numeric
trace within a predeclared tolerance?

## Expected outcome and rationale

The environment and logging procedure were expected to run reproducibly because the environment,
software versions, seed, step limit, and deterministic heuristic were fixed. The heuristic was not
expected to solve the task reliably because it chases the block center without using the goal pose
or planning a useful contact direction.

## Variables and controls

- Independent variable: none; this was a fixed-configuration pilot.
- Dependent metrics / observables: reward, return, coverage, coverage change, maximum coverage,
  contact steps, success, termination flags, render shape, and render dtype.
- Constants and controls: environment ID, dependency versions, seed, 20-step pilot limit,
  `damping=1.0`, state observations, RGB-array rendering, and the block-chasing policy.
- Seeds: `0`.
- Known confounders: native physics and geometry calculations produced a floating-point difference
  of about `5.55e-17` between processes. The specific low-level cause was not isolated.

## Success and stopping criteria

- Pass requires: two new processes finish without warnings or exceptions; both close the
  environment; observations, actions, rewards, coverage, flags, and render metadata are present;
  non-numeric log structure is identical; and corresponding numbers agree with
  `atol=1e-12, rtol=0`.
- Patch when: output is incomplete, cleanup fails, configuration drifts, or numeric differences
  exceed the declared tolerance.
- Repeat when: the processes use different code, dependencies, seeds, or configuration.
- Stop when: success, environment truncation, or the 20-step pilot limit is reached.

## Environment

- OS / architecture: macOS 26.5.2, arm64.
- Python: 3.11.1 in the repository-local `.venv`.
- Packages: exact transitive versions are in [`../requirements-lock.txt`](../requirements-lock.txt).
  Key packages were `gym-pusht==0.1.6`, `gymnasium==1.3.0`, `pymunk==6.11.1`,
  `numpy==2.2.6`, and `opencv-python==4.12.0.88`.
- Environment ID and arguments: `gym_pusht/PushT-v0`, `obs_type="state"`,
  `render_mode="rgb_array"`, `damping=1.0`.
- Source version: installed `gym-pusht` package release 0.1.6; project source is the commit
  containing this record.

## Procedure

1. Create and activate the repository-local Python 3.11 virtual environment.
2. Install the exact dependency lock and run `pip check`.
3. Run [`heuristic_pilot.py`](heuristic_pilot.py) twice in separate processes with the same fixed
   configuration, saving raw logs under the Git-ignored `runs/` directory.
4. Compare both logs with [`compare_numeric_logs.py`](../tests/compare_numeric_logs.py), using the
   declared absolute tolerance.
5. Confirm that raw logs remain ignored and inspect the files intended for Git.

Exact commands:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check

mkdir -p runs/stage1-seed0-final
.venv/bin/python experiments/heuristic_pilot.py > runs/stage1-seed0-final/run-1.txt
.venv/bin/python experiments/heuristic_pilot.py > runs/stage1-seed0-final/run-2.txt
.venv/bin/python tests/compare_numeric_logs.py \
  runs/stage1-seed0-final/run-1.txt \
  runs/stage1-seed0-final/run-2.txt
git status --short
```

## Raw observations

The raw 41-line logs remain local under `runs/stage1-seed0-final/` and are intentionally ignored by
Git. Selected observations from both runs were:

```text
initial_observation [390. 304. 241.54255641 268.51697354 3.3990369]
initial_coverage 0.2657831144772234
step 2 ... reward 0.3312595545887966 coverage 0.31469657685935676 contacts 1
step 20 ... reward 0.0 coverage 0.0 contacts 0 terminated False truncated False
episode_return 1.555275312143944
final_coverage 0.0
max_coverage 0.31469657685935676
contact_steps 12
success False
stop_reason pilot_step_limit
render_shape (680, 680, 3)
render_dtype uint8
closed
```

The independent-process comparison returned:

```text
PASS lines=41 max_abs_diff=5.551e-17 atol=1.0e-12 rtol=0.0
```

## Checks

- [x] Inputs and configuration match the design.
- [x] Outputs have expected shapes, types, and ranges.
- [x] A clean-process rerun was completed and compared.
- [x] Failures and floating-point differences were preserved.
- [x] No private or oversized artifact is intended for Git.

## Results

| Metric | Value |
| --- | ---: |
| Steps | 20 |
| Episode return | 1.555275312143944 |
| Initial coverage | 0.2657831144772234 |
| Final coverage | 0.0 |
| Coverage change | -0.2657831144772234 |
| Maximum coverage | 0.31469657685935676 |
| Contact steps | 12 |
| Success | False |

## Interpretation

The clean environment, selected interface, rendering path, cleanup path, and numeric reproduction
procedure passed for this configuration. The block-chasing heuristic failed on seed 0: it briefly
increased coverage at step 2, then ended with zero coverage and no success. A positive return did
not indicate task completion because reward accumulated from transient overlap.

## Limitations

- This was one 20-step pilot with one seed, not a baseline evaluation.
- The heuristic did not use the goal pose, contact geometry, or T-block orientation.
- Initial coverage used the version-specific private method `env.unwrapped._get_coverage()` because
  reset info in `gym-pusht==0.1.6` does not expose coverage.
- The declared tolerance was adopted after an earlier exact-text comparison exposed a
  `5.55e-17` floating-point difference; it was then validated on two new processes.
- The low-level source of the floating-point difference was not isolated.
- No claim is made about other seeds, damping values, learned policies, or real-world friction.

## Understanding check

The project owner explained that the five-dimensional state observation contains the agent
position, block position, and block angle; the two-dimensional action is the agent target position;
the seed fixes random initial-state sampling; RGB-array rendering returns image data to the calling
program; and `finally` ensures the environment closes after normal execution or an error. She also
distinguished a successful procedure from an effective policy and explained why positive return
alone does not prove success.

## Gate decision

**Pass for Gate 1 environment and Gate 2 interface pilot.** The next stage is to design a baseline
evaluation across declared seeds. This record does not establish baseline policy quality.
