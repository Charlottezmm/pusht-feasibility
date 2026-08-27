# PushT Feasibility Memo

**Date:** 2026-08-27
**Decision:** **PIVOT — refine motion-response measurement before implementing adaptation**

## Question and scope

This first research cycle asked whether a small, reproducible PushT protocol could support a
controlled comparison under different simulator motion-resistance settings. The longer-term
interest is whether a controller can use a short probe and the observed response to adjust later
actions under hidden dynamics. This cycle did not implement that adaptive controller. It tested the
environment, measurement path, simple baselines, and a single-variable open-loop comparison.

## Environment and reproducibility

The experiments used Python 3.11.1, `gym-pusht==0.1.6`, and the compatible Pymunk 6.x interface.
Simulator `damping` is treated only as a motion-resistance setting, not as measured real-world
friction. The environment/interface pilot recorded observations, actions, rewards, coverage,
contacts, termination flags, rendering metadata, and cleanup behavior. Two independent processes
produced numerically equivalent traces under the declared tolerance, supporting Gates 1–2 for the
execution and logging path. [E1]

## Baseline observation

The Gate 3 baseline compared seeded random target-position actions with a block-chasing heuristic
at `damping=1.0`, using paired environment seeds `0..9` and a 300-step limit. The run contained 20
episodes and 6,000 step rows. Both policies had mean final coverage `0.0` and success rate `0.0`.
Therefore, the directional hypothesis that block-chasing would exceed random on the predeclared
primary metric was not supported for this configuration. This is a valid negative result, not
evidence that block-chasing is ineffective under every seed, configuration, or real condition.
[E2]

## Controlled motion-resistance comparison

The Gate 4 comparison changed only `damping={1.0,0.7,0.4}` across paired seeds `0..9`. Within each
seed, the three settings used equal initial observations, identical planned target-position action
arrays, the same 300-step budget, and the same evaluator and metric definitions. The validated run
contained 30 episodes and 9,000 step rows. Mean final coverage and success rate were `0.0` in all
three settings. However, relative to `damping=1.0`, the mean block-trajectory distances were
`196.080298` for `damping=0.7` and `222.402751` for `damping=0.4`. Thus, the manipulation changed
the simulated block response under matched planned actions, while the primary task metric remained
at a complete floor. [E3–E4]

## Limitations

The current evidence does not establish controller effectiveness, adaptation, robustness,
population-level performance, a true hidden resistance value, or real-world friction. Trajectory
distance from the low-resistance reference does not show that the block was harder to move. Net
block displacement and trajectory path length were not predeclared metrics, so they cannot be
retroactively promoted to confirmatory results. The complete final-coverage floor also means that
the present task metric cannot yet distinguish useful controller behavior across settings.

## Decision and next step

The project should **Pivot**, rather than Proceed directly to `probe-adjust` or Stop. The execution,
pairing, logging, and simulator-manipulation chain is credible, so abandoning the question would be
premature. Directly adding adaptation would also be premature because no controller has yet shown
useful variation on the primary task metric.

The smallest next step is a measurement pivot: before inspecting new comparison results,
predeclare net block displacement and block-trajectory path length, then repeat the matched
low/medium/high motion-resistance comparison with the same seeds, initial-state checks, planned
actions, budget, and evaluator controls. This step can test whether motion resistance changes how
far or along what path the block moves. It cannot test probe-based adaptation, because it includes
neither a probe-response feature nor response-conditioned later actions. A later decision must
separately determine whether a controller and task-performance metric with non-floor behavior can
be established before an adaptation comparison is justified.

## Evidence map

- **E1:** [Stage 1 heuristic pilot](2026-08-20-stage1-heuristic-pilot.md)
- **E2:** [Seeded random vs block-chasing baseline](2026-08-25-random-vs-block-chasing-baseline.md)
- **E3–E4:** [Paired motion-resistance comparison](2026-08-26-motion-resistance-comparison.md)
