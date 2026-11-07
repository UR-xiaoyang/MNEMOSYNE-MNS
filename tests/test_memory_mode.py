import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from memory_mode import get_memory_mode, is_experimental_global_enabled, should_expose_proposals_tool


class MemoryModeTest(unittest.TestCase):
    def setUp(self):
        self.old_mode = os.environ.get("MNEMOSYNE_MEMORY_MODE")
    
    def tearDown(self):
        if self.old_mode is None:
            os.environ.pop("MNEMOSYNE_MEMORY_MODE", None)
        else:
            os.environ["MNEMOSYNE_MEMORY_MODE"] = self.old_mode
    
    def test_defaults_to_session_when_unset(self):
        os.environ.pop("MNEMOSYNE_MEMORY_MODE", None)
        self.assertEqual(get_memory_mode(), "session")
    
    def test_session_mode_is_recognized(self):
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        self.assertEqual(get_memory_mode(), "session")
        self.assertFalse(is_experimental_global_enabled())
        self.assertFalse(should_expose_proposals_tool())
    
    def test_experimental_global_mode_is_recognized(self):
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
        self.assertEqual(get_memory_mode(), "experimental_global")
        self.assertTrue(is_experimental_global_enabled())
        self.assertTrue(should_expose_proposals_tool())
    
    def test_unknown_mode_raises_value_error(self):
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "invalid_mode"
        with self.assertRaises(ValueError) as ctx:
            get_memory_mode()
        self.assertIn("invalid_mode", str(ctx.exception))
        self.assertIn("session", str(ctx.exception))
        self.assertIn("experimental_global", str(ctx.exception))
    
    def test_is_experimental_global_returns_false_on_unknown_mode(self):
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "invalid"
        self.assertFalse(is_experimental_global_enabled())
    
    def test_mode_string_is_case_insensitive(self):
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "SESSION"
        self.assertEqual(get_memory_mode(), "session")
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "Experimental_Global"
        self.assertEqual(get_memory_mode(), "experimental_global")


if __name__ == "__main__":
    unittest.main()
