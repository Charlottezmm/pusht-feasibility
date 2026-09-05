"""Read-only, stdlib audit of frozen v0.1 three-path raw evidence (no policy load)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHS = ('fixed', 'probe-no-adjust', 'probe-adjust')
ATOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'non-finite raw value')
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def close(a, b):
    require(math.isclose(a, b, rel_tol=0, abs_tol=ATOL), f'value mismatch: {a} != {b}')


def identity(row):
    return row['environment_seed'], row['configured_damping'], row['path']


def distribution(values):
    return dict(count=len(values), mean_paired_difference=statistics.mean(values),
                median_paired_difference=statistics.median(values),
                standard_deviation=statistics.stdev(values), minimum=min(values),
                maximum=max(values), positive_count=sum(v > ATOL for v in values),
                zero_count=sum(abs(v) <= ATOL for v in values),
                negative_count=sum(v < -ATOL for v in values))


def audit(directory):
    config, manifest, summary = [read(directory / name) for name in
                                 ('config.json', 'manifest.json', 'summary.json')]
    require(config['status'] == summary['status'] == 'succeeded', 'run did not succeed')
    require(not (directory / 'failure.json').exists(), 'failure artifact present')
    for name in ('manifest', 'episodes', 'summary'):
        suffix = '.jsonl' if name == 'episodes' else '.json'
        require(sha256(directory / (name + suffix)) == config[name + '_sha256'],
                f'{name} hash mismatch')
    episodes = lines(directory / 'episodes.jsonl')
    expected = Counter((s, d, p) for s in range(20, 30) for d in (0., 1.) for p in PATHS)
    require(Counter(map(identity, manifest['episodes'])) == expected, 'unexpected manifest')
    require(Counter(map(identity, episodes)) == expected, 'missing/duplicate/unplanned episode')
    values, step_count, stops = {}, 0, Counter()
    for episode in episodes:
        finite(episode)
        key = identity(episode)
        require(episode['status'] == 'succeeded' and episode['formal_evaluation']
                and not episode['debug_only'], f'invalid episode {key}')
        require(episode['policy_seed'] == 100000 + key[0], 'policy seed mismatch')
        close(episode['space_damping'], key[1])
        close(episode['tau'], config['tau'])
        require(episode['calibration_summary_sha256'] == config['frozen_calibration']['summary_sha256'],
                'calibration identity mismatch')
        path = (directory / episode['step_log']).resolve()
        require(path.is_relative_to(directory.resolve()), 'step log outside run directory')
        require(sha256(path) == episode['step_log_sha256'], f'step hash mismatch {key}')
        rows = lines(path)
        require(0 < len(rows) == episode['steps'] == episode['logged_step_count'] <= 300,
                f'incomplete steps {key}')
        require(episode['max_steps'] == 300, 'budget mismatch')
        for step, row in enumerate(rows, 1):
            finite(row)
            require(identity(row) == key and row['step'] == step, 'step identity/order mismatch')
            require(row['policy_seed'] == episode['policy_seed'], 'step policy seed mismatch')
            coverage = row['evaluator_only']['coverage']
            require(0 <= coverage <= 1, 'coverage outside [0,1]')
            close(coverage, row['evaluator_only']['state_after']['coverage'])
            if step < len(rows):
                require(not row['terminated'] and not row['truncated'], 'steps after termination')
        last = rows[-1]
        require(last['terminated'] or last['truncated'], 'missing terminal/truncated final step')
        if len(rows) < 300:
            require(last['terminated'] and episode['success']
                    and episode['stop_reason'] == 'terminated_success', 'invalid early stop')
        close(last['evaluator_only']['coverage'], episode['final_coverage'])
        close(sum(r['reward'] for r in rows), episode['episode_return'])
        close(max([episode['initial_coverage']] + [r['evaluator_only']['coverage'] for r in rows]),
              episode['max_coverage'])
        values[key] = last['evaluator_only']['coverage']
        step_count += len(rows)
        stops[episode['stop_reason']] += 1
    indexed = {identity(r): r for r in episodes}
    pairs, by_condition = [], {}
    for d in (0., 1.):
        deltas = []
        for seed in range(20, 30):
            a, b = [indexed[seed, d, p] for p in PATHS[1:]]
            for field in ('initial_state_sha256', 'prefix_action_sha256', 'pre_probe_state_sha256',
                          'probe_action_sha256', 'post_probe_state_sha256', 'probe_measurement'):
                require(a[field] == b[field], f'pair mismatch: {field}')
            require(a['response_used_count'] == 0, 'control consumed response')
            validation = read(directory / 'pair-validations' /
                              f'seed-{seed}-damping-{str(d).replace(".", "p")}.json')
            require(validation['passed'] and len(validation['checks']) == 18
                    and all(validation['checks'].values()) and not validation['errors'],
                    'saved pair validation failed')
            delta = values[seed, d, PATHS[2]] - values[seed, d, PATHS[1]]
            deltas.append(delta)
            pairs.append(dict(seed=seed, damping=d, delta=delta))
        by_condition[str(d)] = distribution(deltas)
        for k, v in by_condition[str(d)].items():
            close(v, summary['primary_by_condition'][str(d)][k])
    pooled = distribution([p['delta'] for p in pairs])
    for k, v in pooled.items():
        close(v, summary['primary_pooled'][k])
    source_matches = {}
    for field, name in [('formal_evaluator', 'three_path_evaluator.py'),
                        ('wrapper', 'probe_adjust_wrapper.py'),
                        ('protocol', '2026-08-30-probe-adjust-wrapper-protocol-v0.2.md')]:
        source_matches[name] = sha256(ROOT / 'experiments' / name) == config[field + '_sha256']
    return dict(run_status=config['status'], measurement_status='passed_scoped_raw_audit',
                scientific_claim=('directional improvement not supported in either condition'
                                  if all(v['mean_paired_difference'] <= 0 for v in by_condition.values())
                                  else 'inspect per-condition descriptive results; no general claim'),
                episodes=len(episodes), step_rows=step_count, primary_pairs=pairs,
                primary_by_condition=by_condition, pooled_descriptive=pooled,
                independent_units='10 seed blocks; conditions and paths repeat within seed',
                exclusions=[], stop_reasons=dict(stops), historical_git=config['git'],
                current_source_matches_historical=source_matches,
                scope='Hashes, identities, terminal coverage/return/max, pairing, saved validator readback; '
                      'does not re-simulate geometry, policy, or independently revalidate all action rules.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve a new report before auditing; failed audits are retained too.
    with args.output.open('x') as handle:
        report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      command=[sys.executable, *sys.argv], python=platform.python_version(),
                      platform=platform.platform(), auditor_sha256=sha256(Path(__file__)),
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                      inputs={p.name: sha256(p) for p in args.run_dir.glob('*.json*')})
        try:
            report.update(audit(args.run_dir))
            report['status'] = 'succeeded'
        except (ValueError, KeyError, OSError, TypeError) as error:
            report.update(status='failed', measurement_status='unusable_pending_review',
                          error=f'{type(error).__name__}: {error}')
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['status'] == 'succeeded' else 1


if __name__ == '__main__':
    sys.exit(main())
