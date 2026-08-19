# PushT Robot-Learning Feasibility Project

An undergraduate, learning-first research project built around the
[`gym-pusht`](https://github.com/huggingface/gym-pusht) environment.

**Current status: Stage 0 — protocol design and environment setup.** No experiment has been run or
validated in this clean project history.

## Working research direction

The long-term direction is to study how a robot-learning policy behaves when object or environment
dynamics change. The first bounded question is:

> What is the smallest reproducible PushT evaluation protocol that can later support a controlled
> comparison across different simulator motion-resistance settings?

This question is provisional. It will be narrowed using implementation evidence, baseline results,
and faculty feedback.

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
| 0. Question | Define the question, hypothesis, variables, controls, and stop conditions | Reviewed protocol | In progress |
| 1. Environment | Install exact dependencies and run the minimum environment loop | Commands, versions, seed, output, clean-process rerun | Not started |
| 2. Interface | Inspect observation, action, reward, termination, and rendering | Annotated trace and owner explanation | Not started |
| 3. Baseline | Define and run a simple baseline under a fixed evaluation protocol | Metrics across declared seeds/episodes | Not started |
| 4. Controlled comparison | Change one motion-resistance factor while preserving controls | Comparable result table and plots | Not started |
| 5. Interpretation | Analyze failures, uncertainty, and limitations | Feasibility memo and next decision | Not started |

Every gate ends with one of three decisions:

- **Pass** — the evidence and understanding checks both pass;
- **Patch** — a specific gap is repaired, then checked again;
- **Repeat** — the procedure is rerun because the evidence is not reliable.

The detailed contract is in [`PROTOCOL.md`](PROTOCOL.md). New experiment records start from
[`experiments/TEMPLATE.md`](experiments/TEMPLATE.md).

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
experiments/TEMPLATE.md  Blank record used for each experiment
```

Code, dependency locks, and small evidence artifacts will be added during the guided workflow only
after their purpose and acceptance criteria are understood. Large datasets, checkpoints, and raw
runs remain outside Git.

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
