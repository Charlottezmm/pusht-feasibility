# Standard experimental protocol

This protocol is the quality gate for every experiment in the project. Completing a command or
producing a figure does not bypass the required understanding and evidence checks.

## 1. Frame the experiment

Before running code, write:

- one question that the experiment can actually answer;
- the expected outcome and the reason for that expectation;
- the independent variable, if any;
- the dependent metric or observable output;
- the factors held constant;
- known confounders and uncertainties;
- a success criterion and a stopping condition.

If these cannot be stated precisely, the experiment remains at the design gate.

## 2. Freeze the environment

Record:

- operating system and hardware architecture when relevant;
- Python version;
- direct and transitive dependency versions;
- environment ID and constructor arguments;
- installation commands;
- source version or commit when behavior depends on implementation details.

Secrets, personal paths, private course materials, large datasets, and model weights do not enter
the public repository.

## 3. Define the procedure

The procedure must specify:

- exact commands or entry points;
- seed handling for every relevant random source;
- episode and step limits;
- policy or action-generation method;
- recorded fields and file locations;
- error handling and cleanup;
- what will trigger Pass, Patch, Repeat, or Stop.

Change one experimental factor at a time unless the design explicitly studies an interaction.

## 4. Run a pilot

Use the smallest run that can expose interface or logging errors. A pilot checks the procedure; it
does not establish the final result.

Inspect:

- observation and action validity;
- reward and termination behavior;
- output shapes, types, and ranges;
- whether logs contain enough information to diagnose failure;
- whether the environment closes cleanly.

## 5. Reproduce before scaling

Repeat the pilot in a new process with the same configuration and seed. Compare machine-readable
outputs. If deterministic equality is not expected, define an appropriate tolerance before looking
at the result.

Only then expand to multiple seeds, episodes, or parameter settings.

## 6. Analyze without overclaiming

Separate the record into:

- observations: values and events directly produced by the run;
- interpretation: explanations consistent with those observations;
- limitations: alternative explanations and missing evidence;
- decision: continue, patch, repeat, narrow, or stop.

A difference in simulator performance does not by itself establish a real-world physical effect.

## 7. Understanding gate

Before an experiment is marked complete, the project owner should be able to explain:

1. what question the run answered;
2. what each major code block did;
3. which variables changed and which stayed fixed;
4. how the seed affected reproducibility;
5. what the metric measured;
6. what the result did not prove;
7. how to reproduce the run from a clean process.

## 8. Gate decision

- **Pass:** procedure, evidence, and understanding checks all pass.
- **Patch:** a bounded gap has a clear repair and the relevant check will be rerun.
- **Repeat:** the run is unreliable or cannot be compared with the intended protocol.
- **Stop:** the time limit, safety boundary, or feasibility criterion has been reached.

The decision and its evidence belong in the experiment record.

## 9. Operational entry and evidence record

Use [the current workflow](experiments/WORKFLOW.md) and the latest raw config/manifest to establish
scope. Historical next-step prose does not authorize a new question or a full evaluator run.
Before execution, fill the experiment template's baseline, independent-unit, data-role, provenance,
resource-limit and failure-ledger fields. Keep run success, measurement validity, scientific
interpretation and learner understanding as separate decisions.

A command may be called verified only with a current execution record containing exact argv,
executable, source/config identity, exit status and persisted output readback. `--help` verifies only
the CLI/import path. Preserve failed and partial runs; never aggregate surviving episodes as the
complete planned design. Negative outcomes are not exclusion reasons.
