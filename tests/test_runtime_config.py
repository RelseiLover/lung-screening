"""Configuration precedence and literal parsing without real credentials."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runtime_config


class RuntimeConfigTest(unittest.TestCase):
    def test_system_values_win_and_quotes_are_literal(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            config = Path(folder) / ".env"
            config.write_text(
                'DEEPSEEK_MODEL="from-file"\nDEEPSEEK_API_KEY=\nHF_HUB_OFFLINE=0\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DEEPSEEK_MODEL": "from-system"}, clear=True):
                runtime_config.load_environment(config)
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "from-system")
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
                self.assertNotIn("DEEPSEEK_API_KEY", os.environ)

    def test_unknown_settings_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            config = Path(folder) / ".env"
            config.write_text("UNSUPPORTED_KEY=value\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid configuration key"):
                runtime_config.load_environment(config)

    def test_default_configuration_is_at_repository_root(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root / "backend").mkdir()
            (root / ".env").write_text("DEEPSEEK_MODEL=root-model\n", encoding="utf-8")
            with (
                patch.object(runtime_config, "__file__", str(root / "backend/runtime_config.py")),
                patch.dict(os.environ, {}, clear=True),
            ):
                runtime_config.load_environment()
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "root-model")
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
