import importlib
import os
import tempfile
import unittest
from pathlib import Path

import src.config as config


class ConfigTests(unittest.TestCase):
    def test_get_api_key_reads_env_file_from_project_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

            importlib.reload(config)
            os.environ.pop("OPENAI_API_KEY", None)
            config.PROJECT_ROOT = Path(tmp_dir)
            config.load_dotenv_config()

            self.assertEqual(config.get_api_key(), "from-dotenv")

    def test_groq_provider_uses_groq_key(self):
        os.environ["AI_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "groq-test-key"
        os.environ.pop("OPENAI_API_KEY", None)

        importlib.reload(config)
        self.assertEqual(config.get_provider_name(), "groq")
        self.assertEqual(config.get_api_key(), "groq-test-key")


if __name__ == "__main__":
    unittest.main()
