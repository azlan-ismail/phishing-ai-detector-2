import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset.py"
spec = importlib.util.spec_from_file_location("audit_dataset", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.csv"

    def write(self, text):
        self.path.write_bytes(text.encode("utf-8"))

    def test_duplicates_conflicts_and_identifier_exclusion(self):
        self.write("id,x,label\na,1,-1\na,1,-1\nb,1,1\nc,2,1\n")
        result = audit.audit_dataset(self.path, "label", ["id"])
        self.assertEqual(result["label_counts_raw"], {"-1": 2, "1": 2})
        self.assertEqual(result["exact_duplicate_rows_beyond_first"], 1)
        self.assertEqual(result["repeated_feature_rows_beyond_first"], 2)
        self.assertEqual(result["conflicting_feature_groups"], 1)
        self.assertEqual(result["rows_in_conflicting_feature_groups"], 3)
        self.assertEqual(result["feature_count"], 1)
        self.assertEqual(result["label_semantics"], "unverified")

    def test_missing_nonfinite_and_categorical_values(self):
        self.write("x,empty,label\nNA,,-1\ninf,?,1\nword,null,\n0,nan,1\n")
        result = audit.audit_dataset(self.path, "label")
        self.assertEqual(result["columns"]["x"]["missing_count"], 1)
        self.assertEqual(result["columns"]["x"]["nonfinite_numeric_count"], 1)
        self.assertEqual(result["columns"]["x"]["nonnumeric_count"], 1)
        self.assertEqual(result["columns"]["x"]["numeric_min"], 0)
        self.assertEqual(result["all_missing_features"], ["empty"])
        self.assertEqual(result["missing_label_count"], 1)
        json.dumps(result, allow_nan=False)

    def test_quoted_csv_bom_crlf_and_blank_records(self):
        self.write('\ufeffx,label\r\n"a,b",-1\r\n\r\n"a\nb",1\r\n')
        before = self.path.read_bytes()
        result = audit.audit_dataset(self.path, "label")
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["blank_records_skipped"], 1)
        self.assertEqual(result["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.path.read_bytes(), before)

    def test_rejects_invalid_schema_and_rows(self):
        for text in ["", "x,label\n", "x,x,label\n1,2,1\n", "x,label\n1,2,3\n",
                     "x,label\n1\n", " ,label\n1,1\n"]:
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaises(ValueError):
                    audit.audit_dataset(self.path, "label")

    def test_rejects_invalid_column_requests(self):
        self.write("x,label\n1,1\n")
        for label, identifiers in [("missing", []), ("label", ["missing"]),
                                   ("label", ["label"]), ("label", ["x", "x"]),
                                   ("label", ["x"])]:
            with self.subTest(label=label, identifiers=identifiers):
                with self.assertRaises(ValueError):
                    audit.audit_dataset(self.path, label, identifiers)

    def test_cli_outputs_json_and_preserves_input(self):
        self.write("x,label\n1,-1\n2,1\n")
        before = self.path.read_bytes()
        output = Path(self.temp.name) / "reports" / "audit.json"
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.path),
                                 "--label-column", "label", "--output", str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["row_count"], 2)
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.path),
                                 "--label-column", "label", "--output", str(self.path)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
