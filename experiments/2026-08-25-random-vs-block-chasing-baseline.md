# Experiment: Seeded random vs block-chasing PushT baseline

## Metadata

- Date: 2026-08-25 (Asia/Shanghai)
- Stage / gate: Gate 3 baseline evaluation
- Status: `PASS` — Gate 3 baseline complete; no heuristic-effectiveness claim
- Pre-run project HEAD: `71d3dd6`

## Question

Under `gym-pusht==0.1.6`, `damping=1.0`, paired environment seeds `0..9`, and a 300-step
episode limit, is the mean final target-area coverage of the block-chasing heuristic greater than
that of a seeded random target-position baseline?

## Expected outcome and rationale

Charlotte expects H1 to receive support because the block-chasing heuristic uses the observed block
position, whereas the random baseline does not use block or goal information. This is only a
pre-run prediction. The heuristic still ignores the goal pose, block orientation, contact geometry,
and useful pushing direction, so using more information does not guarantee higher final coverage.

## Hypotheses

- H0: `mean_final_coverage(block_chasing) <= mean_final_coverage(random)`.
- H1: `mean_final_coverage(block_chasing) > mean_final_coverage(random)`.

The means are descriptive statistics over the declared paired seeds, not population estimates or a
formal significance test.

## Variables and controls

- Independent variable: action-generation method.
  - `random`: seeded uniform sampling from the valid two-dimensional target-position action space;
    it does not read block or goal position.
  - `block_chasing`: reads `observation[2:4]`, the block center position, and uses it as the action
    target; if the center is outside the legal action space, each coordinate is projected to the
    nearest legal boundary value and the projection is logged. It does not read the goal position.
- Primary dependent metric: mean final coverage across the declared seeds. Coverage is the area of
  overlap between the block and goal geometry divided by the goal area.
- Auxiliary metrics: success rate, mean maximum coverage, mean episode return, mean contact steps,
  initial coverage, and coverage change.
- Constants and controls: `gym_pusht/PushT-v0`, `gym-pusht==0.1.6`, state observations,
  RGB-array render mode, `damping=1.0`, paired environment seeds, 300-step limit, environment
  termination, metric definitions, policy implementations, and logging format.
- Environment seeds: `0..9`; both policies receive the same initial state within each seed.
- Random-action seed rule: `100000 + environment_seed`, fixed before any result is inspected.
- Known confounders and limitations: initial states differ across seeds; the random baseline adds
  action stochasticity; the heuristic is deliberately incomplete; initial overlap can affect raw
  final coverage; native physics may produce tiny floating-point differences; initial coverage
  uses the installed-version private `_get_coverage()` method because reset info omits coverage.
  In the first smoke, `gym-pusht==0.1.6` transiently returned a finite block x-coordinate above
  its declared observation-space upper bound; the evaluator now records such declaration
  violations instead of silently clipping the physical state.
  The first full run showed that an out-of-bounds block center can also make the original
  block-chasing action illegal. Evaluator v0.2 projects only the action target to the nearest legal
  boundary and records every projection; it does not alter the observed physical state.

## Metric boundaries

- Final coverage measures target overlap at the episode end; it does not establish trajectory
  smoothness, robustness, learned intelligence, or hidden-dynamics adaptation.
- Maximum coverage can show temporary overlap but not final placement.
- Return accumulates intermediate rewards and can be positive without final task success.
- Contact steps diagnose interaction but do not show that the block moved toward the goal.
- Success uses the installed environment's strict `coverage > 0.95` rule.

## Success and stopping criteria

- Pass requires: the frozen evaluator completes the declared paired runs; observations have the
  expected shape and finite values; any disagreement with the environment's declared observation
  space is explicitly logged; actions remain inside their declared space; per-step and per-episode
  logs are preserved; configuration, seeds, metrics, termination, and cleanup are consistent;
  summary calculations match raw records; and Charlotte can explain the result and its boundary.
  H1 does not need to receive support.
- Patch when: a bounded logging, summary, cleanup, or documentation defect can be repaired without
  changing the research question or intended comparison.
- Repeat when: policy implementations, seeds, environment arguments, step budgets, or metric
  definitions differ between groups; outputs are missing or unreliable; or the procedure cannot be
  reproduced.
- Stop when: the environment reaches success, the 300-step time limit is reached, or evaluator and
  measurement-path blockers exceed the session's 45-minute implementation timebox.

## Staged procedure

1. Validate the evaluator statically and inspect its command-line interface.
2. Run a targeted paired smoke on environment seeds `0` and `3`. Seed `0` checks the ordinary
   path; seed `3` checks the projected-action path that blocked v0.1. This checks only interface,
   logging, pairing, metrics, termination, and cleanup.
3. Inspect the smoke's config, random actions, ordinary and projected heuristic actions, episode
   summaries, and aggregate summary. Do not interpret comparative effectiveness from the smoke.
4. If the smoke passes without changing the scientific contract, run paired seeds `0..9` in a new
   output directory.
5. Preserve raw step and episode logs, verify the summary against them, then interpret H0/H1 within
   the declared scope.

Smoke command:

```bash
.venv/bin/python experiments/baseline_evaluator.py \
  --seeds 0,3 \
  --max-steps 300 \
  --output-dir runs/baseline-v0.2-smoke-seeds0-and3
```

Planned paired-run command, allowed only after smoke readback:

```bash
.venv/bin/python experiments/baseline_evaluator.py \
  --seeds 0-9 \
  --max-steps 300 \
  --output-dir runs/baseline-v0.2-seeds0-9
```

## Raw observations

- Failed smoke retained at `runs/baseline-v0.1-smoke-seed0/`. It stopped during random seed `0`
  before step `183` was written, when the environment returned
  `[362.27774814, 282.25021059, 513.592125, 445.63565542, 1.61467139]`.
- A bounded diagnostic replay found exactly one declared-space violation in 300 random-policy
  steps: block x was `513.592125` at step `183`, then returned inside the declared range. The
  episode itself reached the wrapper's 300-step truncation normally.
- Repeat smoke retained at `runs/baseline-v0.1-smoke-seed0-patch1/`. Both policies started from
  exactly the same observation and completed 300 logged steps. All 300 block-chasing actions
  exactly equal `observation[2:4]` after the action space's required `float32` conversion; all
  random actions remained inside the action space. The one random-policy observation declaration
  violation at step `183` was preserved in both the step and episode logs.
- Failed full v0.1 run retained at `runs/baseline-v0.1-seeds0-9/`. It contains seven complete
  episode rows and stopped before block-chasing seed `3` step `62`, when the observed block center
  `[451.416246, -1.341951]` would have produced an action outside the legal action space. It is an
  incomplete, invalid comparison and will not be combined with v0.2 outputs.
- Targeted v0.2 smoke retained at `runs/baseline-v0.2-smoke-seeds0-and3/`. All four episodes and
  1,200 step rows completed. Seed `0` reproduced the v0.1 metrics exactly. On block-chasing seed
  `3`, step `62` projected `[451.416246, -1.341951]` to `[451.416260, 0.0]`; every action remained
  legal and all raw-to-summary checks passed.
- Full v0.2 run retained at `runs/baseline-v0.2-seeds0-9/`. It contains 20 complete episode rows
  and 6,000 complete step rows. Both policies received exactly paired initial observations for all
  ten seeds. All actions followed the frozen rules and remained legal; raw episode values,
  aggregate means, counts, paired differences, and the evaluator hash were independently read back.

## Results

The v0.1 repeat smoke passed its interface and measurement-path checks. On seed `0`, both policies had
final coverage `0.0` and no success. Random had return `0.296878`, maximum coverage `0.282034`, and
82 contact steps; block-chasing had return `1.555275`, maximum coverage `0.314697`, and 275 contact
steps. These are smoke observations only, not the planned ten-seed comparison.
The v0.1 full run is invalid and has no aggregate result. Evaluator v0.2 produced a technically
validated result; its scientific interpretation remains pending below.

The validated v0.2 full run produced:

| Metric | Seeded random | Block-chasing v0.2 |
|---|---:|---:|
| Episodes | 10 | 10 |
| Mean initial coverage | 0.051792 | 0.051792 |
| Mean final coverage | 0.000000 | 0.000000 |
| Success rate | 0.000000 | 0.000000 |
| Mean maximum coverage | 0.177348 | 0.105511 |
| Mean episode return | 0.553766 | 0.468451 |
| Mean contact steps | 127.6 | 270.7 |
| Observation-space violation count | 9 | 296 |
| Projected action count | 0 | 295 |

Every paired final-coverage difference was `0.0`. Three block-chasing episodes used projected
actions; seed `9` accounted for 293 of the 295 projections. All 20 episodes ended through the
environment's 300-step time limit, and none reached success.

## Interpretation

The failed smoke is an invalid run, not evidence for or against H1. The failure data disposition is
`Repeat`; the bounded evaluator defect disposition is `Patch`. A valid negative result will be
recorded as `H1 not supported`, not rerun merely because it differs from the pre-run expectation.
The repeat smoke establishes that the paired execution and raw-to-summary path work for seed `0`;
it does not establish comparative effectiveness.
For the declared seeds and configuration, H1 is not supported: both policies have mean final
coverage `0.0`, so block-chasing is not greater than seeded random on the primary metric. This does
not prove a population-level null result or behavior under other seeds, environment parameters,
policies, or episode budgets.

Block-chasing produced more contact steps (`270.7` versus `127.6`), meaning that contact was
detected during more time steps. It does not show that contacts pushed the block toward the target.
Its lower mean maximum coverage and return also do not independently prove a general performance
ordering; the primary predeclared result remains the final-coverage tie.

The 295 projected block-chasing actions, including 293 on seed `9`, expose an important failure
mode: chasing the center can keep following a block at the workspace boundary without solving the
task. Because the projection rule was frozen before the complete v0.2 run and every projection was
logged, this is a documented policy/design limitation rather than missing data.

## Limitations

This bounded baseline cannot establish population-level performance, learned-policy quality,
robustness to different motion resistance, probe-response adaptation, OSI, or real-world behavior.

## Understanding check

Charlotte independently explained the action generators, paired seeds, primary metric, auxiliary
metric boundaries, raw-to-summary path, failed-run disposition, and conclusion limits. Her final
interpretation was: H1 is not supported because both final coverages are zero; more contact steps
show more time steps with contact but not target-directed block motion; and the experiment is a
`Pass` because the frozen protocol completed reliably.

## Gate decision

`PASS` for Gate 3 baseline evidence. This means the comparison is reproducible and honestly
interpreted; it does not mean the block-chasing heuristic is effective. No `Proceed / Pivot / Stop`
feasibility decision is made from this baseline alone.
