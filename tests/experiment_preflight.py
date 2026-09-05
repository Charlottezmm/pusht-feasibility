"""Offline controller preflight; imports/help only, no inference or downloads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-config', type=Path, required=True,
                        help='Existing controller gate config with checkpoint/config locations')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as handle:
        report = dict(status='started', command=[sys.executable, *sys.argv],
                      created_at_utc=datetime.now(timezone.utc).isoformat(),
                      platform=platform.platform(), limits='offline, no inference, 60 seconds per command',
                      checks=[])
        try:
            config = json.loads(args.controller_config.read_text())
            python = os.environ.get('PUSHT_CONTROLLER_PYTHON', config['python_executable'])
            checkpoint = Path(os.environ.get('PUSHT_CHECKPOINT_DIR', config['checkpoint_dir']))
            report['controller_python'] = python
            report['checkpoint_dir'] = str(checkpoint)
            report['source_sha256'] = {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for pattern in ('experiments/*.py', 'experiments/*.md', 'tests/*.py', '*.md', '*lock.txt')
                for p in ROOT.glob(pattern)
            }
            report['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=10).strip()
            report['git_status'] = subprocess.check_output(
                ['git', 'status', '--porcelain'], cwd=ROOT, text=True, timeout=10)
            weights = checkpoint / 'model.safetensors'
            actual = hashlib.sha256(weights.read_bytes()).hexdigest()
            report['checkpoint_sha256'] = actual
            if actual != config['checkpoint_weights_sha256']:
                raise ValueError('checkpoint weights differ from controller record')
            compat = ROOT / 'runs/2026-08-29-controller-gate/compat-config/config.json'
            report['compat_config_sha256'] = hashlib.sha256(compat.read_bytes()).hexdigest()
            if report['compat_config_sha256'] != config['compat_config_sha256']:
                raise ValueError('compatibility config differs from controller record')
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            commands = [[python, '--version'], [python, '-m', 'pip', 'check'],
                        [python, '-m', 'pip', 'freeze', '--all'],
                        [python, 'experiments/three_path_evaluator.py', '--help'],
                        [python, 'experiments/three_path_logging_smoke.py', '--help']]
            for command in commands:
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=60)
                report['checks'].append(dict(command=command, returncode=result.returncode,
                                             stdout=result.stdout, stderr=result.stderr))
                if result.returncode:
                    raise ValueError(f'preflight command failed: {command}')
            report['status'] = 'succeeded'
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(f"status={report['status']} output={args.output}")
    return 0 if report['status'] == 'succeeded' else 1


if __name__ == '__main__':
    sys.exit(main())
