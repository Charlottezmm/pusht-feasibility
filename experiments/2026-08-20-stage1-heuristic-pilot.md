# Experiment: Fixed-seed heuristic PushT pilot

## Metadata

- Date: 2026-08-20 (Asia/Shanghai)
- Stage / gate: Gate 1 environment and Gate 2 interface pilot
- Status: Gate 1 environment pass; Gate 2 interface and reproducibility pass; heuristic pilot
  criterion not met; Gate 3 baseline evaluation not started
- Related commit: the commit that adds this record

## Question

Under the fixed configuration of `gym-pusht==0.1.6`, `seed=0`, `damping=1.0`, state observations,
RGB-array rendering, and a 20-step limit:

1. Can the PushT environment reproducibly complete the `reset → step → render → close`
   workflow and return the expected observations, rewards, termination flags, and diagnostic
   information?
2. In this episode, does the block-chasing heuristic increase the final coverage relative to the
   initial coverage or reach the environment's success condition?

## Expected outcome and rationale

The environment was expected to accept valid two-dimensional actions, update its state, return
complete interface outputs, produce an RGB image array, and close cleanly. Because the environment
configuration, dependency versions, seed, step limit, and policy were fixed, two independent
processes were expected to produce numerically equivalent traces within the declared tolerance.

The block-chasing policy always selected the current center of the T-block as the agent's next
target position. Therefore, the agent was expected to move toward and potentially contact the
block. However, the policy did not use the goal position, block orientation, or contact geometry,
so it was not expected to place the block reliably inside the target region.

For this pilot, evidence supporting policy effectiveness would require either `success=True` or a
positive final coverage change.

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

The results support the conclusion that the selected PushT environment, interface, rendering path,
logging procedure, cleanup procedure, and fixed-seed reproduction workflow functioned correctly
under the tested configuration.

The results also show that the block-chasing heuristic did not meet the task criterion in this
episode. Although the agent contacted the block and briefly increased coverage, the episode ended
with zero coverage and `success=False`.

The positive episode return does not demonstrate task completion because it accumulated rewards
from temporary overlap during intermediate steps. Likewise, the maximum coverage being greater
than the initial coverage only shows a temporary improvement, not a successful final placement.

Therefore, this pilot provides evidence that the experiment can run reproducibly, but it does not
establish that the policy is generally effective or ineffective across different seeds and initial
states.

## Limitations

- The experiment included only one episode with one seed and 20 action steps. It therefore cannot
  establish whether the policy is stable or effective across different initial states.
- The current policy generated actions using only the current T-block center. It did not consider
  the two-dimensional goal position, block orientation, contact geometry, or a useful pushing
  direction.
- The experiment did not compare the heuristic against a random, no-op, or other declared baseline
  policy. Therefore, observed changes in coverage and return cannot be attributed confidently to
  the heuristic.
- Only `damping=1.0` was tested, so the experiment cannot compare the effects of different damping
  values on coverage, return, contact, or success.
- Initial coverage used the version-specific private method `env.unwrapped._get_coverage()` because
  reset info in `gym-pusht==0.1.6` does not expose coverage.
- The declared tolerance was adopted after an earlier exact-text comparison exposed a
  `5.55e-17` floating-point difference; it was then validated on two new processes.
- The low-level source of the floating-point difference was not isolated.
- No conclusions can be made about learned policies, other environment versions, other seeds,
  different damping values, or real-world friction.

## Understanding check

The project owner explained that the five-dimensional state observation contains the agent
position, block position, and block angle; the two-dimensional action is the agent target position;
the seed fixes random initial-state sampling; RGB-array rendering returns image data to the calling
program; and `finally` ensures the environment closes after normal execution or an error. She also
distinguished a successful procedure from an effective policy and explained why positive return
alone does not prove success.

## Gate decision

- **Gate 1 — Environment execution: PASS.** The environment initialized, accepted actions, updated
  its state, rendered correctly, and closed cleanly.
- **Gate 2 — Interface and reproducibility: PASS.** The required measurements were recorded, and two
  independent processes produced equivalent numeric traces within the declared tolerance.
- **Current heuristic pilot criterion: NOT MET.** For `seed=0` and the 20-step limit, final coverage
  was lower than initial coverage and `success=False`.
- **Gate 3 — Baseline evaluation: NOT STARTED.** One episode is insufficient to evaluate general
  policy effectiveness.

Stage 1 can now be closed as a successful environment and measurement pilot. The next experiment
should keep the environment and dependency configuration fixed, compare the heuristic with a
declared baseline across multiple seeds, and only then begin a controlled damping comparison.
