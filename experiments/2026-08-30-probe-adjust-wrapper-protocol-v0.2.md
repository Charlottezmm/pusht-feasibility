# Probe-adjust three-path wrapper protocol v0.2

- Date: 2026-08-30
- Status: `DESIGN FROZEN FOR IMPLEMENTATION; NO CALIBRATION OR EVALUATION RESULT`
- Scope: simulator-instrumented wrapper, pure-core tests, and an optional debug-only logging smoke
- Predecessor: Academic causal-comparison protocol v0.1
- Controller evidence: `experiments/2026-08-29-diffusion-controller-gate.md`

## 1. Today’s answerable question

Can a three-path wrapper be implemented and tested so that, under a frozen controller, probe,
step budget, schema, and paired initial conditions, the primary `probe-adjust` versus
`probe-no-adjust` comparison differs only in whether the declared probe-response feature enters a
frozen later-action rule, without exposing true simulator `damping` to the wrapper?

Passing this design/implementation gate does not show that `probe-adjust` improves task
performance. A future paired evaluation will test

```text
delta = mean_final_coverage(probe-adjust)
      - mean_final_coverage(probe-no-adjust)

H0: delta <= 0
H1: delta > 0
```

## 2. Frozen controller and environment roles

- Environment: `gym_pusht/PushT-v0`, `obs_type=pixels_agent_pos`.
- Frozen base controller: pinned `lerobot/diffusion_pusht` checkpoint and compatibility config from
  the 2026-08-29 controller gate.
- Controller nominal condition: `damping=0.0`.
- Declared dynamics-shift diagnostic: `damping=1.0`.
- Total episode budget: `T=300`; all probe steps count toward the same budget.
- Probe start: `t_probe=20`, unless the episode has already ended.
- Probe length: `P=5`.
- Probe target offset: `L_probe=20` position units.
- Action semantics: absolute 2-D agent target, clamped to the environment action bounds.
- Base policy weights and configuration remain unchanged.

The true `damping` value is an evaluator-only grouping field. It must not appear in a protocol-core
or controller action-decision API.

## 3. Information boundary

| Field | Source | Action-decision access | Role |
|---|---|---:|---|
| `pixels`, `agent_pos` | ordinary policy observation | frozen policy, both probe paths | ordinary feedback |
| true `damping` | environment construction | prohibited | hidden grouping field |
| `info["block_pose"]` | simulator info | measurement adapter only | privileged simulator measurement |
| `x` | privileged block pose plus probe direction | `probe-adjust` only | response feature |
| path length, final coverage, hashes | evaluator/logging | prohibited | metrics and diagnostics |

Both probe paths calculate and log `x`; `probe-no-adjust` must not use it to select or transform an
action. Because the current `x` comes from `info["block_pose"]`, this version is explicitly a
**simulator-instrumented response-conditioned wrapper**. It is not an observation-only deployable
adaptation method.

## 4. Shared probe and response

At the probe boundary:

```text
d_probe = normalize(p_block_before - p_agent_before)
target  = clamp(p_block_before + L_probe * d_probe, action_bounds)
probe   = repeat(target, P steps)
```

The two probe paths must persist exactly the same probe array and SHA-256 hash for a paired
seed/setting. A zero-length direction is invalid.

For block positions `b_0, ..., b_P` observed over the probe window:

```text
delta_block = b_P - b_0
x = dot(delta_block, d_probe)
path_length = sum(norm(b_t - b_(t-1)) for t=1..P)
```

`x` is signed net block displacement along the probe direction. It is not task performance, true
motion resistance, or an estimated dynamics parameter. Path length is an evaluator diagnostic and
does not enter the adjustment rule.

The probe is valid only when all of the following hold:

- the probe direction is defined;
- all inputs and derived values are finite;
- at least one probe-window step reports agent-block contact;
- the measurement source is explicitly `simulator_info_block_pose`.

An invalid probe remains in the logs. It must not be interpreted as high motion resistance and must
fall back to `scale=1.0`.

## 5. Three paths

### `fixed`

Run the frozen base policy without inserting a dedicated probe. This is an auxiliary reference and
cannot isolate the effect of using `x` because it also differs in whether a probe occurred.

### `probe-no-adjust`

Run the shared prefix, execute the shared probe, calculate and log `x`, clear any stale queued
pre-probe policy actions, and resume ordinary replanning from the latest policy observation. `x`
must not enter the action rule.

### `probe-adjust`

Run the identical shared prefix and probe, calculate the same `x`, clear stale queued actions, and
resume ordinary replanning. Transform every later base target with the frozen rule:

```text
if probe_valid and x < tau:
    scale = 1.25
else:
    scale = 1.0

v = a_base - p_agent
a_executed = clamp(p_agent + scale * v, action_bounds)
```

The `1.25` scale is a bounded exploratory rule, not a theoretically derived or empirically optimal
value. A negative result can reject only this frozen rule in the tested scope. It cannot reject all
probe-based adjustment.

## 6. Calibration and evaluation separation

- Calibration seeds: `100..109`.
- Future evaluation seeds: `20..29`.
- The two seed sets are disjoint and must remain disjoint.
- Calibration represents `damping=0.0` and `damping=1.0` equally.
- `tau` is the median of pooled valid calibration `x` values.
- Fewer than five valid calibration probes is `Patch`; evaluation data must not supply or tune
  `tau`.
- `tau`, valid/invalid counts, raw calibration paths, config hash, wrapper hash, and controller
  identity must be persisted before evaluation starts.

Unit tests inject explicit `tau` values to exercise both branches. They do not estimate `tau`.
An optional single-seed logging smoke may verify interfaces and logging only; it must be labelled
debug-only and cannot enter calibration, evaluation, or an effectiveness claim.

## 7. Pairing and fairness invariants

For every paired seed/setting in the primary comparison:

- environment seed, policy seed, simulator setting, initial state, controller/checkpoint/config,
  device, dependencies, prefix behavior, and protocol version match;
- probe arrays and hashes match exactly;
- pre-adjustment measurement records match within a predeclared numeric tolerance;
- both paths keep ordinary policy feedback;
- probe steps count inside the same `T=300` budget;
- termination, action bounds, metrics, schema, and logging rules match;
- only `probe-adjust` may consume `x` when producing a later executed action.

Closed-loop action divergence after the declared adjustment point is an expected causal path. It is
not by itself adaptation-effectiveness evidence.

## 8. Required implementation tests

Before any environment smoke, tests must cover:

1. signed response for positive, negative, zero, and non-finite inputs;
2. path length for stationary, straight, and backtracking trajectories;
3. probe direction/target construction and zero-distance invalidity;
4. exact shared probe-array/hash equality;
5. `x` isolation from `probe-no-adjust` and use only by `probe-adjust`;
6. invalid-probe fallback to `scale=1.0`;
7. target scaling and action-bound clamping;
8. shared total-budget and phase accounting;
9. matching per-path result schemas and separation of controller-visible/evaluator-only fields;
10. absence of true `damping` from protocol-core action APIs.

## 9. Evidence decisions and stop rules

- **Pass:** protocol-core tests and readback pass; this supports only a design/implementation gate.
- **Patch:** a bounded protocol, measurement, wrapper, logging, clamp, or calibration defect is
  repaired before trustworthy evaluation evidence is produced.
- **Repeat:** an existing pair differs in seed, setting, initial state, controller, probe/hash,
  budget, metric, or protocol version; that pair cannot enter the primary comparison.
- **Stop current run:** a non-finite action/state, invalid environment transition, unsafe output, or
  timebox boundary occurs. Preserve the partial failure and do not resume or aggregate it.
- A reliable run with lower `probe-adjust` coverage is a credible negative result, not an automatic
  `Repeat`.
- At the 150-minute task boundary, if protocol/tests are not frozen, stop without an environment
  smoke and classify the remaining implementation gap as `Patch`.

## 10. Claim boundary

Tests or a single-seed logging smoke can show only that the declared response-conditioned action
path and evidence schema operate as designed. They cannot establish improved final coverage,
adaptation effectiveness, robustness, true-friction identification, or real-robot transfer.

A future valid paired evaluation can at most state whether this frozen `1.25` response-conditioned
rule improved the predeclared task metric for the tested simulator, controller, settings, seeds,
budget, and software configuration.
