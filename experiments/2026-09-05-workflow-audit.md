# Experiment workflow audit — 2026-09-05

## Scope and current evidence

This work improves experiment start and result review, without selecting a new intervention.
Repository HEAD at inspection: `d1545e99880186a541886c0fd0026d827ab4d31c`; worktree initially clean.
No repository-specific AGENTS.md was found; the supplied global instructions and PROTOCOL.md apply.
README's Gate 5 headline was stale relative to the actual August 31 calibration/formal-evaluation
records, runners and local raw directories. README now points to the completed three-path study.

Current question: with the frozen controller/probe/tau/budget, does allowing probe response into
later actions improve final coverage over `probe-no-adjust`? `fixed` is an auxiliary reference.
No newer experimental result or predeclared intervention was found in this repository.

## First calculation, then raw readback

Before implementation, a stdlib calculation from `episodes.jsonl` recomputed the primary deltas.
The new independent audit then derived final coverage from raw terminal step rows and compared the
saved summaries at absolute tolerance `1e-12`:

| damping | paired observations | mean adjust − no-adjust | sample SD | positive / zero / negative |
|---:|---:|---:|---:|---:|
| 0.0 | 10 | -0.1003153571181648 | 0.23918454896785302 | 5 / 0 / 5 |
| 1.0 | 10 | -0.02858323320770843 | 0.09038811982811934 | 0 / 9 / 1 |

There are ten seed blocks with repeated paths and conditions. The twenty seed/condition pairs are
not twenty independent seeds. Pooled mean `-0.06444929516293661` is descriptive only.
60/60 planned episodes and 15,406 finite step rows were present; 22 episodes stopped early with
success and 38 ended at environment truncation. No episodes/pairs were excluded. All twenty saved
pair validation files passed their eighteen recorded checks. The audit additionally checked
primary shared-state/probe fields and response isolation counts; it did not independently replay
all action decisions or reconstruct coverage geometry.

## Current execution evidence

Commands are in [WORKFLOW.md](WORKFLOW.md). The following are current local execution artifacts,
not a claim that historical commands were re-executed:

| Command / purpose | This execution | Persisted evidence under ignored `runs/` |
|---|---|---|
| `experiment_preflight.py` with the documented controller config | Exit 0; all five bounded subprocesses exit 0 | `2026-09-05-controller-preflight.json` |
| `validate_three_path_run.py` with the documented raw directory | Exit 0; persisted `succeeded` read back | `2026-09-05-three-path-audit-final.json` |
| `unittest discover -s tests -p test_validate_three_path_run.py -v` via `.venv/bin/python` | 5 tests passed | `2026-09-05-audit-tests.json` |
| Preflight with intentional `PUSHT_CONTROLLER_PYTHON=/nonexistent/pusht-python` override | Expected exit 1; retained `failed` with `FileNotFoundError` | `2026-09-05-preflight-failure-check.json` |

Audit tests check usable negative evidence, duplicate identity rejection even after updating its
hash, semantically incorrect summary rejection even after updating its hash, step corruption and
failed-run rejection. They use isolated temporary copies of the original logs.

Current environment is macOS 26.6.2 arm64, Python 3.11.1. The repository `.venv` has no torch/LeRobot;
its heuristic lock is not the controller environment. The historical controller interpreter still
exists and passed `pip check`, full dependency freeze and both runner `--help` imports offline.
Current key versions: torch 2.6.0, torchvision 0.21.0, diffusers 0.32.2, gymnasium 0.29.1,
gym-pusht 0.1.5, numpy 2.1.3. LeRobot is a local source installation; its local path is preserved in
the ignored dependency snapshot. This is a current environment record, not a clean-install proof.
Checkpoint and compatibility-config hashes match the controller record. Current wrapper, frozen
protocol and formal evaluator hashes match the formal raw config. The historical base commit was
`3c73d1c340fef63c9c5be0521bb1116c5694f51f` with untracked evaluator files: the base commit alone is
insufficient. New preflight snapshots include helper-source hashes; missing historical helper
hashes cannot be recovered as if they had been recorded at runtime.

## Changes and decision

Reused the existing README, protocol, experiment template, runner and `tests/validate_*` structure.
Added a stdlib offline preflight and independent raw audit, regression checks, exact command
instructions, baseline/independent-unit definitions, data-role and provenance fields, failure and
exclusion ledger, and a proposed bounded debug launcher. Frozen scientific runners/configs and
original raw results were not changed. The optional 120-second CPU debug pilot is explicitly
unexecuted; no new simulation, inference, training, paid compute or dependency installation occurred.

- Run status: historical run reports success; current audit/preflight commands succeeded.
- Measurement: scoped raw readback passed; full action-rule replay, simulator geometry and upstream
  training-data provenance were not revalidated here.
- Science: the directional improvement hypothesis remains unsupported in these two settings;
  this does not establish general adaptation, universal harm or real-robot performance.
- Understanding: not assessed by this automated workflow task.
- Stop decision: workflow goal met using existing evidence; no runtime pilot needed.

The workflow audit ended without a commit, push or publication. The owner subsequently authorized
committing and pushing these workflow changes. Raw evidence remains local and ignored by Git.
