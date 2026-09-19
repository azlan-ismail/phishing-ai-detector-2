import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_research_data.py"
spec = importlib.util.spec_from_file_location("prepare_research_data", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def test_invalid_counts_never_enter_ratios(self):
        data = pd.DataFrame({"length_url": [-1, 0, 10], "domain_length": [-1, 2, 5], "qty_dot_url": [-1, 1, 2]})
        x = prep.candidates(data, "mendeley")
        self.assertTrue(x.loc[:1, "domain_url_ratio"].isna().all())
        self.assertEqual(x.loc[2, "domain_url_ratio"], .5)
        self.assertNotIn("Extension_LetterCount", x)

    def test_source_scaler_does_not_fit_or_clip_target(self):
        train = pd.DataFrame({c: [1., 3., np.nan] for c in prep.FEATURES})
        state = prep.fit_transformer(train)
        target = pd.DataFrame({c: [101., np.nan] for c in prep.FEATURES})
        actual = prep.transform(target, state)
        self.assertEqual(actual.loc[0, "url_length"], 50.)
        self.assertEqual(actual.loc[1, "url_length"], .5)
        self.assertEqual(actual.loc[1, "url_length_missing"], 1)
        self.assertEqual(state["maximum"]["url_length"], 3.)
        with self.assertRaises(ValueError):
            prep.transform(target[list(reversed(prep.FEATURES))], state)

    def test_all_missing_training_column_rejected(self):
        train = pd.DataFrame({c: [1., 2.] for c in prep.FEATURES})
        train["url_length"] = np.nan
        with self.assertRaises(ValueError):
            prep.fit_transformer(train)

    def test_duplicate_groups_stay_together_even_with_conflicting_labels(self):
        groups = np.array([str(i) for i in range(100)] * 2)
        labels = np.array([0] * 100 + [1] * 100)
        split = prep.grouped_split(labels, groups, 42)
        np.testing.assert_array_equal(split[:100], split[100:])
        np.testing.assert_array_equal(split, prep.grouped_split(labels, groups, 42))
        for part in range(3):
            self.assertEqual(set(labels[split == part]), {0, 1})

    def test_end_to_end_exports_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            n = 100
            a = pd.DataFrame({"urlLen": np.arange(n) + 10, "domainlength": np.arange(n) + 5,
                              "NumberofDotsinURL": np.arange(n) % 3 + 1,
                              "URL_Type_obf_Type": ["benign", "phishing"] * 50})
            b = a.rename(columns={"urlLen": "length_url", "domainlength": "domain_length", "NumberofDotsinURL": "qty_dot_url", "URL_Type_obf_Type": "phishing"})
            b["phishing"] = [0, 1] * 50
            paths = {"iscx": root / "a.csv", "mendeley": root / "b.csv"}
            a.to_csv(paths["iscx"], index=False); b.to_csv(paths["mendeley"], index=False)
            original = {k: p.read_bytes() for k, p in paths.items()}
            output = root / "run"
            report = prep.prepare(paths, output)
            self.assertEqual(len(report["exports"]), 8)
            for name, metadata in report["exports"].items():
                frame = pd.read_csv(output / name)
                self.assertEqual(len(frame), metadata["rows"])
                self.assertTrue(frame["sample_id"].is_unique)
                self.assertTrue(np.isfinite(frame.drop(columns="sample_id").to_numpy()).all())
            for k, path in paths.items():
                self.assertEqual(original[k], path.read_bytes())
            candidate_report = prep.prepare(paths, root / "candidate-run", group_by="candidate")
            for entry in candidate_report["datasets"].values():
                self.assertEqual(entry["candidate_test_rows_matching_training"], 0)
            with self.assertRaises(ValueError):
                prep.prepare(paths, output)


if __name__ == "__main__":
    unittest.main()
