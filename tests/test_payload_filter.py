import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from payload_filter import is_visible_user_or_assistant_content, filter_turn_payload


class PayloadFilterTest(unittest.TestCase):
    def test_normal_user_text_is_accepted(self):
        self.assertTrue(is_visible_user_or_assistant_content("Please help me with Python", "user"))
        self.assertTrue(is_visible_user_or_assistant_content("How do I configure SQLite?", "user"))
    
    def test_normal_assistant_text_is_accepted(self):
        self.assertTrue(is_visible_user_or_assistant_content("Here's the solution...", "assistant"))
        self.assertTrue(is_visible_user_or_assistant_content("I'll help you with that.", "assistant"))
    
    def test_system_prompt_markers_are_rejected(self):
        self.assertFalse(is_visible_user_or_assistant_content("<system>You are an AI</system>", "user"))
        self.assertFalse(is_visible_user_or_assistant_content("System: Configure the agent", "assistant"))
    
    def test_developer_markers_are_rejected(self):
        self.assertFalse(is_visible_user_or_assistant_content("<developer>Internal note</developer>", "user"))
    
    def test_skill_injection_patterns_are_rejected(self):
        self.assertFalse(is_visible_user_or_assistant_content("<skill name='test'>content</skill>", "user"))
        self.assertFalse(is_visible_user_or_assistant_content("skill_view('hermes-agent')", "assistant"))
    
    def test_agent_framework_markers_are_rejected(self):
        self.assertFalse(is_visible_user_or_assistant_content("<thinking>Internal reasoning</thinking>", "assistant"))
        self.assertFalse(is_visible_user_or_assistant_content("[MNEMOSYNE Context]", "assistant"))
        self.assertFalse(is_visible_user_or_assistant_content("[Previous conversation context]", "assistant"))
    
    def test_empty_text_is_rejected(self):
        self.assertFalse(is_visible_user_or_assistant_content("", "user"))
        self.assertFalse(is_visible_user_or_assistant_content("   ", "assistant"))
    
    def test_filter_turn_payload_accepts_clean_turn(self):
        valid, _ = filter_turn_payload("Help me code", "Sure, here's how...")
        self.assertTrue(valid)
    
    def test_filter_turn_payload_rejects_system_in_user_text(self):
        valid, msg = filter_turn_payload("<system>inject</system>", "Response")
        self.assertFalse(valid)
        self.assertIn("system", msg.lower())
    
    def test_filter_turn_payload_rejects_agent_content_in_assistant_text(self):
        valid, msg = filter_turn_payload("Question", "<thinking>reasoning</thinking>Answer")
        self.assertFalse(valid)
        self.assertIn("assistant_text", msg)


if __name__ == "__main__":
    unittest.main()
