"""Audit regressions using the existing local raw run; never mutate original evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('validate_three_path_run.py'))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
RAW = AUDIT.ROOT / 'runs/2026-08-31-three-path-formal-evaluation-v0.1'


@unittest.skipUnless(RAW.exists(), 'local ignored raw evidence required')
class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'
        shutil.copytree(RAW, self.run)

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def refresh_hash(self, name, key):
        config = AUDIT.read(self.run / 'config.json')
        config[key] = AUDIT.sha256(self.run / name)
        self.write('config.json', config)

    def test_valid_negative_result_is_usable(self):
        report = AUDIT.audit(self.run)
        self.assertEqual(report['episodes'], 60)
        self.assertEqual(report['step_rows'], 15406)
        self.assertAlmostEqual(report['primary_by_condition']['0.0']['mean_paired_difference'],
                               -0.1003153571181648)

    def test_duplicate_episode_rejected_even_with_updated_hash(self):
        rows = AUDIT.lines(self.run / 'episodes.jsonl')
        rows[-1] = rows[0]
        (self.run / 'episodes.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        self.refresh_hash('episodes.jsonl', 'episodes_sha256')
        with self.assertRaisesRegex(ValueError, 'missing/duplicate'):
            AUDIT.audit(self.run)

    def test_summary_drift_rejected_even_with_updated_hash(self):
        summary = AUDIT.read(self.run / 'summary.json')
        summary['primary_by_condition']['0.0']['mean_paired_difference'] = 0.5
        self.write('summary.json', summary)
        self.refresh_hash('summary.json', 'summary_sha256')
        with self.assertRaisesRegex(ValueError, 'value mismatch'):
            AUDIT.audit(self.run)

    def test_raw_step_corruption_rejected(self):
        episode = AUDIT.lines(self.run / 'episodes.jsonl')[0]
        with (self.run / episode['step_log']).open('a') as handle:
            handle.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'step hash mismatch'):
            AUDIT.audit(self.run)

    def test_failed_run_not_aggregated(self):
        self.write('failure.json', {'status': 'failed', 'error': 'time limit'})
        with self.assertRaisesRegex(ValueError, 'failure artifact'):
            AUDIT.audit(self.run)


if __name__ == '__main__':
    unittest.main()
