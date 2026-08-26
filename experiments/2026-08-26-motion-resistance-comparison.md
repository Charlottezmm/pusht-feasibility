# Experiment: Paired low / medium / high PushT motion resistance

## Metadata

- Date: 2026-08-26 (Asia/Shanghai)
- Stage / gate: Gate 4 controlled comparison
- Status: `PASS` — controlled comparison complete; no effectiveness or adaptation claim
- Pre-implementation project HEAD: `95dc922`

## Question

Under `gym-pusht==0.1.6`, paired environment seeds, identical open-loop target-position action
arrays, and a 300-step episode limit, does changing only simulator `damping` among `1.0`, `0.7`,
and `0.4` produce reproducible differences in PushT task metrics or block trajectories?

## Expected outcome and rationale

Before seeing any three-setting result, Charlotte predicted that `damping=0.4` will make the block
hardest to move because it loses the largest fraction of velocity per second and therefore has the
strongest simulator motion resistance. She predicted that block position / trajectory will expose
a difference before final coverage.

This is a motion-response prediction, not a claim that lower damping will monotonically improve or
worsen final task coverage.

## Variables and controls

- Independent variable: explicit `pymunk.Space.damping` through the environment constructor.
  - `low_resistance`: `damping=1.0`;
  - `medium_resistance`: `damping=0.7`;
  - `high_resistance`: `damping=0.4`.
- Primary dependent metric: mean final coverage for each setting across the declared paired seeds.
- Auxiliary task metrics: mean maximum coverage, mean coverage change, mean return, mean contact
  steps, success rate, and observation-space violation count.
- Motion-response diagnostics: per-seed block-trajectory distance and final block-position distance
  relative to `damping=1.0` under the same planned action sequence.
- Constants and controls: `gym_pusht/PushT-v0`, state observation, RGB-array render mode, installed
  package versions, environment seeds, initial observations, generated target-position actions,
  action timing, 300-step limit, environment termination, PD gains `k_p=100` and `k_v=20`, control
  frequency, metric definitions, evaluator version, and logging format.
- Full-run seeds: `0..9`; smoke seeds: `0,3`.
- Action rule: generate one legal float32 Box action sequence per environment seed using
  `action_seed = 100000 + environment_seed`, hash it, then replay the exact arrays in all settings.
- Known confounders: `damping` affects simulated bodies in the shared Pymunk space; collisions can
  amplify small trajectory differences; early success can cause different realized trajectory
  lengths despite the same maximum budget; coverage can have a floor effect; observations can be
  finite yet transiently outside the environment's declared Box in `gym-pusht==0.1.6`.

## Metric boundaries

- Final coverage measures target overlap at the episode end; it does not measure adaptation.
- Block-position and trajectory distance show different simulated responses to identical planned
  actions; they do not by themselves establish useful task performance.
- Return, maximum coverage, and contact steps are auxiliary and cannot replace the declared primary
  metric after results are visible.
- `damping` is simulator motion resistance, not measured real-world friction.

## Success and stopping criteria

- Pass requires: all declared episodes complete; each seed has identical initial observations and
  planned action hashes across settings; actions match step by step over the realized shared prefix;
  configuration and evaluator hash are frozen; raw episode and step records are retained; an
  independent validator reproduces every aggregate; and Charlotte explains the evidence boundary.
- Patch when: a bounded logging, validation, summary, cleanup, or documentation defect can be fixed
  without changing the question, settings, seeds, actions, budget, or metric definitions.
- Repeat when: initial observations, actions, seeds, budgets, metrics, or evaluator versions differ;
  outputs are missing or cannot be independently recomputed; rerun from smoke after repair.
- Stop when: the implementation or measurement blocker exceeds 45 minutes, or safe cleanup would
  risk the fixed 08:50 course-registration action.

## Staged procedure

1. Run syntax and unit checks without creating experimental results.
2. Run seeds `0,3` at all three settings as an interface, pairing, logging, and summary smoke.
3. Validate the smoke independently. Do not interpret comparative effectiveness from it.
4. Freeze settings, seeds, action rule, metrics, evaluator hash, and commands.
5. Run seeds `0..9` at all three settings in a new directory.
6. Validate raw outputs independently, then let Charlotte interpret the declared metrics.

Static and unit checks:

```bash
.venv/bin/python -m py_compile \
  experiments/motion_resistance_evaluator.py \
  tests/test_motion_resistance_evaluator.py \
  tests/validate_motion_resistance_run.py
.venv/bin/python -m unittest tests/test_motion_resistance_evaluator.py
```

Smoke command, allowed only after the checks pass:

```bash
.venv/bin/python experiments/motion_resistance_evaluator.py \
  --seeds 0,3 \
  --max-steps 300 \
  --output-dir runs/motion-resistance-v0.1-smoke-seeds0-and3
```

Independent smoke validation:

```bash
.venv/bin/python tests/validate_motion_resistance_run.py \
  runs/motion-resistance-v0.1-smoke-seeds0-and3
```

Planned full-run command, allowed only after smoke readback:

```bash
.venv/bin/python experiments/motion_resistance_evaluator.py \
  --seeds 0-9 \
  --max-steps 300 \
  --output-dir runs/motion-resistance-v0.1-seeds0-9
```

## Raw observations

- Static compilation and all 3 unit tests passed.
- Smoke output: `runs/motion-resistance-v0.1-smoke-seeds0-and3/`.
- Frozen evaluator SHA-256:
  `d43ca76da61fc9dcc2d30dd94896bc7a88d1bc19e40325b3844189f99b5e5790`.
- Seeds `0,3` completed at all three settings: 6/6 episodes and 1,800 total step rows.
- Every episode reached the installed environment's 300-step truncation; no episode succeeded.
- For each seed, all three settings had exactly equal initial observations, one common planned
  action SHA-256, and step-by-step equal actions across all 300 realized steps.
- Independent validator returned `status=valid`, `episode_count=6`, `seed_count=2`.
- Smoke final coverage was `0.0` in all six episodes. This is retained as an interface-path
  observation and is not used to choose settings, change metrics, or claim comparative performance.
- Charlotte independently judged the smoke `Pass`: environment seeds, initial states, every
  target-position action and its timing, 300-step budget, control frequency, PD parameters,
  termination rules, and metrics were fixed. Different block positions are allowed responses to
  the single changed environment variable and do not make the pairing unfair.
- Full output: `runs/motion-resistance-v0.1-seeds0-9/`.
- The frozen evaluator SHA-256 matched the smoke exactly before the full command ran.
- Seeds `0..9` completed at all three settings: 30/30 episodes and 9,000 total step rows.
- All 30 episodes reached the installed environment's 300-step truncation; none succeeded.
- All 10 seeds had exactly one planned-action SHA-256 shared across their three settings, exactly
  equal initial observations, and step-by-step equal actions for all 300 realized steps.
- Independent validator returned `status=valid`, `episode_count=30`, `seed_count=10`; every
  persisted aggregate matched recomputation from raw episode and step rows.

## Checks

- [x] Inputs and configuration match the design.
- [x] Outputs have expected shapes, types, and ranges.
- [x] Initial observations and action arrays are paired across settings.
- [x] Independent raw-to-summary validation passes.
- [x] Failures and warnings are preserved.
- [x] No private or oversized artifact is staged for Git.

## Results

Validated task metrics:

| Metric | Low resistance `1.0` | Medium resistance `0.7` | High resistance `0.4` |
| --- | ---: | ---: | ---: |
| Episodes | 10 | 10 | 10 |
| Mean initial coverage | 0.051792 | 0.051792 | 0.051792 |
| Mean final coverage | 0.000000 | 0.000000 | 0.000000 |
| Success rate | 0.000000 | 0.000000 | 0.000000 |
| Mean maximum coverage | 0.177348 | 0.214388 | 0.083741 |
| Mean episode return | 0.553766 | 1.406317 | 0.280227 |
| Mean contact steps | 127.6 | 99.0 | 127.2 |
| Observation-space violations | 9 | 0 | 0 |

Validated motion-response diagnostics relative to low resistance:

| Setting | Mean block-trajectory distance | Mean final block-position distance |
| --- | ---: | ---: |
| Medium resistance `0.7` | 196.080298 | 287.053630 |
| High resistance `0.4` | 222.402751 | 366.443529 |

The primary metric has a complete floor tie: all three mean final coverages are zero. The
diagnostics show that identical planned actions produced different simulated block trajectories,
with the high-resistance setting farther from the low-resistance reference than the medium setting
under these declared seeds.

The pre-run phrase “hardest to move” is not directly resolved by distance from the low-resistance
trajectory. Absolute block displacement or path length was not predeclared as a metric. It may be
calculated later only as explicitly exploratory analysis or predeclared in a new configuration; it
must not be retroactively presented as the primary result of this run.

## Interpretation

For the declared fixed open-loop actions, settings, and seeds, the primary metric detected no final
task-performance difference: every setting had mean final coverage `0.0`, success rate `0.0`, and
all episodes ended at the 300-step environment truncation. This is a bounded floor result, not
evidence that the simulator settings are dynamically equivalent.

The paired diagnostics establish that changing only `damping` produced different block
trajectories under identical initial states and action arrays. High resistance was farther from the
low-resistance reference than medium resistance on the declared trajectory-distance diagnostics.
These distances do not show that high resistance moved the block less, and the non-monotonic
auxiliary metrics do not support a task-performance ordering.

The controlled manipulation and measurement path are feasible, so Gate 4 execution passes. The
primary metric's floor means this run does not yet justify implementing or claiming an adaptive
controller. The later feasibility Gate should distinguish this experimental `Pass` from a research
`Proceed`: a controller/metric path with useful task-performance variation may need to be
established before adding `probe-adjust` complexity.

### Guided interpretation checkpoint

Charlotte correctly explained that the high-resistance trajectory was farther from the
low-resistance reference, while the run did not measure a predeclared absolute block movement
distance. She therefore correctly rejected the claim that high resistance had been proven to move
the block less. Local `Patch`: define and interpret the predeclared primary metric, mean final
coverage, before the final gate.

Primary-metric retest: `Pass`. Charlotte independently explained that all three mean final
coverages being zero supports only “no detected difference in final task performance; all three
failed on that metric.” She explicitly rejected “motion resistance had no effect,” because the
paired trajectory diagnostics differ, and she did not use those diagnostics to claim successful
task performance or less absolute block movement.

Final transfer check, first attempt: `Patch`. Charlotte initially judged a hypothetical run with
equal initial states but different action hashes as `Pass`. Correction: equal initial states are
necessary but insufficient for this open-loop causal comparison. Different action hashes introduce
a second changed input, so trajectory differences can no longer be attributed only to `damping`;
the run must be `Repeat`. The remaining transfer and recap checks stay pending until this point is
independently restated.

Action-hash retest: `Pass`. Charlotte independently explained that different actions create a
second possible cause of trajectory differences, so equal initial states alone do not preserve the
single-variable attribution. Terminology patch: `primary metric`, not `primary matrix`.

Closed-loop transfer check: Charlotte correctly explained that the same frozen controller can emit
different actions because different motion-resistance settings produce different observation
histories, and correctly reframed the question as behavior and task performance of one closed-loop
controller across motion-resistance settings. Local `Patch`: lack of policy-weight optimization is
not sufficient to rule out adaptation. Execution-time adaptation may use a fixed rule to extract a
probe-response feature and alter later actions; ordinary feedback replanning alone does not prove
hidden-resistance adaptation.

Adaptation-control check: Charlotte correctly identified that `probe-adjust` alone cannot separate
ordinary observation feedback, the probe itself, and use of probe-response feature `x`. Local
`Patch`: “the first path does not execute a probe” describes `fixed-continuation`, not the primary
control `probe-no-adjust`. Both probe paths must execute the identical probe; only `probe-adjust`
may use `x` to alter the comparison-window actions.

Primary-control restatement, first attempt: Charlotte correctly fixed initial state, observation
history, identical probe actions, and total step budget, and correctly identified use of `x` as the
single intended difference. Local `Patch`: the base controller must also be the same, not different;
otherwise controller identity is a confounder. Terminology remains `primary metric`, not `primary
matrix`.

Primary-control retest: `Pass`. Charlotte independently fixed initial state, observation history,
base controller, identical probe actions, total step budget, metrics, and evaluation seeds; she
identified use of response feature `x` in the comparison-window adjustment as the only intended
difference and correctly linked the comparison to the predeclared primary metric. Obvious
speech-to-text substitutions such as “matrix” for “metric” are treated as transcription noise when
her surrounding explanation makes the intended concept unambiguous.

Final closed-book recap: `Pass`, confidence `100/100`. Charlotte stated that motion resistance was
the single changed variable; environment seeds, PD controller parameters, control frequency, and
target-position actions were fixed; all final coverages were zero; and block trajectories differed
across settings. She correctly rejected any claim that motion resistance improved the primary
metric.

## Limitations

This bounded comparison cannot establish adaptation, controller quality, population-level
robustness, the true hidden resistance value, real-world friction, or real-robot behavior.

## Understanding check

Passed after targeted patches and independent retests. Charlotte distinguished the primary metric
from trajectory diagnostics, identified unequal action hashes as a reason to repeat the open-loop
comparison, separated ordinary closed-loop feedback from adaptation, and completed the final
closed-book recap with confidence `100/100`.

## Gate decision

`PASS` for Gate 4 controlled-comparison execution and understanding. This means the frozen
single-variable procedure completed, raw results independently validated, and Charlotte interpreted
the evidence within its limits. It does not mean motion resistance improved task performance,
high resistance was proven hardest to move, a controller is robust, adaptation occurred, or the
project has reached its later `Proceed / Pivot / Stop` feasibility decision.

Next delayed recall: Why can all three final coverages be zero while motion resistance still has a
verified effect on the simulated trajectories?

## Optional post-task continuation discussion

Charlotte correctly ruled out `Stop` because the environment, intended variable manipulation,
paired execution, logging, and measurement path all worked. She ruled out immediate `Proceed`
because the evidence chain does not yet support implementing and comparing controllers when the
declared task-performance metric remains at a complete floor.

Her proposed smallest next step is a measurement `Pivot`: add an explicit block-movement measure
such as net displacement from the initial block position or trajectory path length, then summarize
it across the paired seeds. This is scientifically relevant to her pre-run “hardest to move”
prediction, but it was selected after seeing the current result. Therefore it must be labeled
exploratory when calculated from the existing raw logs, or predeclared as a secondary metric in a
new frozen configuration before a confirmatory rerun. It cannot be retroactively promoted to this
run's primary metric.

This discussion was an optional extension after Gate 4 had already passed, not an additional
completion requirement. Its tentative status was `provisional Pivot` for a possible next action.
The scheduled feasibility Gate remains separate and has not yet issued the project's formal
`Proceed / Pivot / Stop` decision.
