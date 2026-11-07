"""Test train_rank_hybrid.py can be imported both as script and as package."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_import_from_clean_interpreter():
    """Test that `from scripts.train_rank_hybrid import ...` works from repo root.
    
    This verifies the module can be imported as a package without relying on
    sys.path side effects from other tests or direct script execution.
    """
    # Run in a clean child interpreter with cwd=repo root
    # Do NOT add scripts/ to sys.path - test pure package import
    code = """
import sys
from pathlib import Path
# Verify scripts/ is NOT in sys.path (clean interpreter)
scripts_path = str(Path.cwd() / 'scripts')
assert scripts_path not in sys.path, f"scripts/ should not be in sys.path: {sys.path}"
# Now test package import
from scripts.train_rank_hybrid import set_seed, get_git_sha
print('import ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0, (
        f"Package import failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "import ok" in result.stdout


def test_direct_script_help_does_not_trigger_training():
    """Test that `python scripts/train_rank_hybrid.py --help` exits cleanly without E5/training."""
    result = subprocess.run(
        [sys.executable, "scripts/train_rank_hybrid.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,  # Module-level imports (torch/transformers) are slow
    )
    
    assert result.returncode == 0, (
        f"Direct script --help failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    # Verify help text appears (not training output)
    assert "usage:" in result.stdout.lower() or "--help" in result.stdout.lower()
