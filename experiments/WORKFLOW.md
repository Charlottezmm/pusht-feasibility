# Experiment start and result review

## Scope and decision before execution

Read README, PROTOCOL, the latest dated experiment record, then its actual `config.json`,
`manifest.json`, raw episode/step files and any `failure.json`. The latest completed question is the
frozen three-path response-use comparison, not the earlier heuristic measurement pivot. No new
scientific design is approved by this workflow. Do not launch the formal evaluator merely to check
setup: its CLI has a fixed 60-episode, 300-step-per-episode manifest and no wall-time limit.

For that design, `probe-no-adjust` is the primary control: it preserves the probe, controller,
feedback and total budget while withholding response use. `fixed` also removes the probe and is
only auxiliary. Calibration uses seeds 100–109; evaluation uses 20–29. Once these evaluation results
inform tuning, their reuse is exploratory, not untouched confirmation.

The independent randomization block is the environment/policy seed pair. There are ten seed blocks,
ten paired deltas per damping, and repeated conditions within each seed. Frames, steps, three paths
and the twenty seed/condition pairs are not twenty independent seeds. Report per-condition mean,
median, sample SD, range and signs; the pooled result is descriptive. Any future confidence interval
must respect seed blocks, including dependence across conditions.

## Offline preflight (no inference)

Run from the repository root. The local `.venv` is sufficient for the stdlib audit utilities; it
lacks torch and LeRobot. The root dependency lock belongs to the historical heuristic environment.
Do not install it over the controller environment. The helper below reads the existing controller
config's interpreter and checkpoint locations; optional `PUSHT_CONTROLLER_PYTHON` and
`PUSHT_CHECKPOINT_DIR` override them explicitly. Missing files are a setup blocker, not permission
to download or rebuild the environment.

```bash
.venv/bin/python tests/experiment_preflight.py \
  --controller-config runs/2026-08-29-controller-gate/official-default-d0-seed1000/config.json \
  --output runs/2026-09-05-controller-preflight.json
```

This exact command passed on 2026-09-05. Use a fresh report filename on every subsequent invocation;
reports refuse overwrite. Read `status`, every subprocess exit/stdout/stderr, actual interpreter,
full `pip freeze --all`, checkpoint/config hashes, Git HEAD/dirty paths and source-file hashes.
It runs offline, with one OpenMP/MKL thread and at most 60 seconds for each of five subprocesses
(300 seconds total subprocess allowance). It imports the runner via `--help`, checks dependencies,
and hashes local files; it does not load policy weights for inference or verify controller behavior.
Editable dependency locations in the freeze are local evidence, not portable installation recipes.

The existing runner saves manifest/config, controller/wrapper/protocol hashes, seeds, episode logs,
pair validations and failure artifacts. Its base Git commit can describe a dirty worktree; use the
source hashes as well. Preflight also snapshots helper modules omitted by historical configs.
Do not retroactively claim those missing historical hashes were captured at runtime.

## Independent result readback

```bash
.venv/bin/python tests/validate_three_path_run.py \
  --run-dir runs/2026-08-31-three-path-formal-evaluation-v0.1 \
  --output runs/2026-09-05-three-path-audit-final.json
.venv/bin/python -m unittest discover -s tests -p test_validate_three_path_run.py -v
```

The exact commands above were executed in this audit; see the dated record. Use a fresh output name
when repeating. The audit is stdlib-only, does not import the evaluator's aggregation function, and
recomputes deltas from each raw step log's last coverage. It checks raw hashes, exact frozen manifest
and episode identities, finite values, ordered steps, valid early stops, return/max/final coverage,
primary pairing and the saved 18-check validator files. It compares aggregates at absolute tolerance
`1e-12`, including negative and zero outcomes. Exit 0 plus persisted `status=succeeded` is required.
Failures return nonzero and leave a failed report. The regression tests need the ignored local raw
run; without it they explicitly skip, which is not raw-evidence validation.

This is a scoped audit, not independent simulator geometry reconstruction or a replay of all action
rules. Saved validator flags are read back, not treated as a new independent action-rule proof.
Current source matches to historical hashes are reported separately from raw-data validity.
Historical calibration/checkpoint identities remain in the formal record; this auditor does not
recompute calibration tau or verify upstream training-data provenance.

## If a runtime pilot later becomes necessary

No runtime pilot was needed or executed on 2026-09-05. First fill TEMPLATE.md and preflight the exact
source/config snapshot. Reuse `three_path_logging_smoke.py`, not the formal evaluator. A suitable
**proposed, not runtime-verified here** debug budget is one CPU process, one thread, seed 1000,
damping 0.0, three paths × at most 30 steps, 120 seconds total, offline, zero spend. Use the existing
20-step prefix and 5-step probe; debug tau is an interface input, never calibration evidence.

After setting `PUSHT_CONTROLLER_PYTHON` and `PUSHT_CHECKPOINT_DIR` to the verified local locations,
this proposed launcher enforces the wall limit and retains stdout, stderr and timeout status:

```bash
.venv/bin/python - <<'PY'
import json, os, pathlib, subprocess
out = pathlib.Path('runs/debug-pilot-new-attempt')
log = pathlib.Path('runs/debug-pilot-new-attempt-launch.json')
assert not out.exists() and not log.exists(), 'choose a new attempt name'
cmd = [os.environ['PUSHT_CONTROLLER_PYTHON'], 'experiments/three_path_logging_smoke.py',
       '--checkpoint-dir', os.environ['PUSHT_CHECKPOINT_DIR'],
       '--compat-config-dir', 'runs/2026-08-29-controller-gate/compat-config',
       '--device', 'cpu', '--damping', '0.0', '--seed', '1000', '--max-steps', '30',
       '--probe-start', '20', '--probe-length', '5', '--target-offset', '20',
       '--debug-tau', '73.6528373524437', '--debug-only', '--output-dir', str(out)]
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
record = dict(command=cmd, status='started', wall_limit_seconds=120)
with log.open('x') as ledger, open(str(log)+'.stdout', 'x') as stdout, open(str(log)+'.stderr', 'x') as stderr:
    try:
        result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, timeout=120)
        record.update(status='succeeded' if result.returncode == 0 else 'failed', returncode=result.returncode)
    except (subprocess.TimeoutExpired, OSError) as error:
        record.update(status='failed', error=str(error))
    finally:
        json.dump(record, ledger, indent=2)
if record['status'] != 'succeeded':
    raise SystemExit(1)
PY
```

A timeout kills/waits for this single runner; preserve its partial directory and launcher ledger.
Never resume or pool partial output. Read its persisted config, per-path logs and validation before
calling the measurement usable. This debug output cannot enter calibration or formal evaluation.
A command that exits successfully is only run success; an import/help check is less than that.
No budget or pilot command here authorizes training, paid compute, scaling or a changed question.

## Failure, exclusion and closure

Record attempt ID, full argv, data role, planned identity, status, error/timeout, inclusion decision
and replacement link in the template ledger. Setup/environment failures stop before science;
measurement defects require Patch; incomplete/mismatched evidence requires Repeat after repair.
Preserve failed directories and create new ones for retries. A killed runner may retain `started`:
launcher failure/partial manifest overrides it. Never average only surviving episodes as a complete
run. Invalid calibration responses are retained but excluded from tau by the frozen rule; invalid
formal probes retain fallback and stay in the primary set. Zero/negative deltas are not exclusions.

Close with separate run status, scoped measurement validity, hypothesis result and understanding
status. A valid negative result does not trigger Repeat. Keep raw logs/local paths under ignored
`runs/`; changes to tracked files still require separate commit/push/publication authorization.
