import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from api_security import authorization_is_valid


class ApiSecurityTest(unittest.TestCase):
    def test_authentication_is_disabled_when_no_key_is_configured(self):
        self.assertTrue(authorization_is_valid("", None))
        self.assertTrue(authorization_is_valid("", "Bearer arbitrary"))

    def test_exact_bearer_token_is_required_when_key_is_configured(self):
        self.assertTrue(authorization_is_valid("secret", "Bearer secret"))
        self.assertFalse(authorization_is_valid("secret", None))
        self.assertFalse(authorization_is_valid("secret", "secret"))
        self.assertFalse(authorization_is_valid("secret", "Bearer wrong"))


if __name__ == "__main__":
    unittest.main()
