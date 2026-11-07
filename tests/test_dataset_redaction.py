"""
Test suite for redact_and_validate_dataset.py

Tests redaction of sensitive data and dataset validation.
"""
import unittest
import tempfile
import json
from pathlib import Path


class TestDatasetRedaction(unittest.TestCase):
    """Test sensitive data redaction."""

    def test_redacts_api_keys(self):
        """Should redact API keys and bearer tokens."""
        from scripts.redact_and_validate_dataset import redact_sensitive_data
        
        # Use realistic API key format (32+ chars after sk-)
        text_with_key = "My API key is sk-1234567890abcdefghijklmnopqrstuvwxyz and Bearer xyz789token12345"
        result = redact_sensitive_data(text_with_key)
        
        self.assertNotIn("sk-1234567890abcdefghijklmnopqrstuvwxyz", result["content"])
        self.assertNotIn("xyz789token12345", result["content"])
        self.assertTrue(result["redacted"])
        self.assertIn("api_key", result["redaction_types"])

    def test_redacts_email_addresses(self):
        """Should redact email addresses."""
        from scripts.redact_and_validate_dataset import redact_sensitive_data
        
        text = "Contact me at user@example.com or admin@test.org"
        result = redact_sensitive_data(text)
        
        self.assertNotIn("user@example.com", result["content"])
        self.assertNotIn("admin@test.org", result["content"])
        self.assertTrue(result["redacted"])
        self.assertIn("email", result["redaction_types"])

    def test_redacts_connection_strings(self):
        """Should redact database connection strings."""
        from scripts.redact_and_validate_dataset import redact_sensitive_data
        
        text = "Connect to postgresql://user:pass@localhost:5432/db"
        result = redact_sensitive_data(text)
        
        self.assertNotIn("user:pass", result["content"])
        self.assertTrue(result["redacted"])

    def test_redacts_home_paths(self):
        """Should redact user home directory paths."""
        from scripts.redact_and_validate_dataset import redact_sensitive_data
        
        text = "File at C:\\Users\\alice\\Documents\\secret.txt"
        result = redact_sensitive_data(text)
        
        self.assertNotIn("alice", result["content"])
        self.assertTrue(result["redacted"])

    def test_preserves_clean_content(self):
        """Should not modify content without sensitive data."""
        from scripts.redact_and_validate_dataset import redact_sensitive_data
        
        clean_text = "This is a normal message about Python programming"
        result = redact_sensitive_data(clean_text)
        
        self.assertEqual(result["content"], clean_text)
        self.assertFalse(result["redacted"])
        self.assertEqual(len(result["redaction_types"]), 0)


class TestDatasetValidation(unittest.TestCase):
    """Test dataset validation and audit."""

    def test_validates_required_fields(self):
        """Should reject records missing required fields."""
        from scripts.redact_and_validate_dataset import validate_message_schema
        
        # Valid message
        valid_msg = {
            "id": "abc123",
            "source": "hermes",
            "source_file": "test.db",
            "session_id": "sess1",
            "ordinal": 1,
            "timestamp": 1700000000,
            "role": "user",
            "content": "Hello",
            "metadata": {},
            "redacted": False,
            "truncated": False
        }
        self.assertTrue(validate_message_schema(valid_msg)["valid"])
        
        # Missing required field
        invalid_msg = valid_msg.copy()
        del invalid_msg["session_id"]
        result = validate_message_schema(invalid_msg)
        self.assertFalse(result["valid"])
        self.assertIn("session_id", result["errors"][0])

    def test_detects_remaining_secrets_in_scan(self):
        """Should detect if secrets remain after redaction."""
        from scripts.redact_and_validate_dataset import scan_for_secrets
        
        # Clean content
        clean = "Normal conversation text"
        result = scan_for_secrets(clean)
        self.assertEqual(result["secret_count"], 0)
        
        # Content with secrets (realistic format)
        with_secret = "My API key: sk-1234567890abcdefghijklmnopqrstuvwxyz"
        result = scan_for_secrets(with_secret)
        self.assertGreater(result["secret_count"], 0)
        self.assertIn("api_key", result["secret_types"])

    def test_processes_messages_with_rejection_log(self):
        """Should process messages and log rejections."""
        from scripts.redact_and_validate_dataset import process_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input JSONL
            input_path = Path(tmpdir) / "input.jsonl"
            with open(input_path, 'w') as f:
                # Valid message
                f.write(json.dumps({
                    "id": "1", "source": "hermes", "source_file": "test.db",
                    "session_id": "sess1", "ordinal": 1, "timestamp": 1700000000,
                    "role": "user", "content": "Hello", "metadata": {},
                    "redacted": False, "truncated": False
                }) + '\n')
                
                # Invalid message (missing session_id)
                f.write(json.dumps({
                    "id": "2", "source": "hermes", "role": "user", "content": "Hi"
                }) + '\n')
            
            output_dir = Path(tmpdir) / "output"
            result = process_dataset(input_path, output_dir)
            
            self.assertEqual(result["processed"], 2)
            self.assertEqual(result["accepted"], 1)
            self.assertEqual(result["rejected"], 1)
            
            # Check rejection log exists
            rejection_log = output_dir / "rejections.jsonl"
            self.assertTrue(rejection_log.exists())


if __name__ == "__main__":
    unittest.main()
