import json
import tempfile
import unittest
from pathlib import Path
from lesson11_merge import check_adapter, BASE_ID

class TestAdapter(unittest.TestCase):
    def test_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                check_adapter(Path(d))
    def test_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path":BASE_ID}))
            (p/"adapter_model.safetensors").write_bytes(b"fixture")
            self.assertEqual(check_adapter(p)["base_model_name_or_path"],BASE_ID)
    def test_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path":"wrong"}))
            (p/"adapter_model.safetensors").write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                check_adapter(p)
if __name__=="__main__": unittest.main()