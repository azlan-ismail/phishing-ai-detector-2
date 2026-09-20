import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_research_benchmark import file_hash, run


class DiscountSelectionTests(unittest.TestCase):
    def check_configuration(self, ablation, gamma, batch=64):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'manifest.json').write_text('{}')
            selection = {'preparation_manifest_sha256': file_hash(root/'manifest.json'),
                         'gamma': .99, 'batch_size': 64, 'weighting': 'none',
                         'feature_condition': 'all', 'trees': 200}
            (root/'selection.json').write_text(json.dumps(selection))
            args = SimpleNamespace(prepared=str(root), output=str(root),
                tuning_selection=str(root/'selection.json'), budget_selection=None,
                discount_ablation_from_tuning=ablation, gamma=gamma,
                batch_size=batch, weighting='none', feature_condition='all', trees=200)
            run(args)

    def test_explicit_ablation_accepts_inherited_gamma_only(self):
        # Existing output is checked after configuration validation, before training.
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            self.check_configuration(True, 0)

    def test_unmarked_gamma_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Tuning configuration mismatch: gamma'):
            self.check_configuration(False, 0)

    def test_ablation_does_not_bypass_other_configuration_guards(self):
        with self.assertRaisesRegex(ValueError, 'Tuning configuration mismatch: batch_size'):
            self.check_configuration(True, 0, batch=32)
        with self.assertRaisesRegex(ValueError, 'requires tuning selection and gamma zero'):
            self.check_configuration(True, .5)


if __name__ == '__main__':
    unittest.main()
