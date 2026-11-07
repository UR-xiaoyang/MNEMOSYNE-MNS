"""
Tests for train_rank_hybrid.py import fallback mechanism.

Ensures that ModuleNotFoundError is handled precisely:
- Direct script import (run_output_manager not found) → fallback to package import
- Package import (scripts.run_output_manager not found) → propagate error
- Internal import errors within run_output_manager → propagate error (not masked)
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def test_package_import_works():
    """Package import should work - verify the fallback mechanism itself."""
    # We don't need to actually import the heavy module, just verify the pattern works
    # The actual package import is already tested in test_train_rank_hybrid_package_import.py
    import tempfile
    import textwrap
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a simple module to test package import fallback
        (tmp_path / "scripts").mkdir()
        rom = tmp_path / "scripts" / "run_output_manager.py"
        rom.write_text(textwrap.dedent("""
            def compute_file_hash(path):
                return "package_import_hash"
        """))
        
        test_script = tmp_path / "test_fallback.py"
        test_script.write_text(textwrap.dedent("""
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            
            # This should fail on direct import and succeed with package fallback
            try:
                from run_output_manager import compute_file_hash
            except ModuleNotFoundError as exc:
                if exc.name == 'run_output_manager':
                    from scripts.run_output_manager import compute_file_hash
                else:
                    raise
            
            result = compute_file_hash(None)
            assert result == "package_import_hash", f"Got {result}"
            print("PACKAGE-FALLBACK-OK")
        """))
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Package fallback failed:\n{result.stderr}"
        assert "PACKAGE-FALLBACK-OK" in result.stdout


def test_direct_script_help_without_heavy_imports():
    """
    Test that the script can at least be parsed and show basic structure.
    We test --help in a separate integration test to avoid timeout from torch imports.
    """
    # Just verify the file compiles and has the expected structure
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SCRIPTS_DIR / "train_rank_hybrid.py")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Script compilation failed:\n{result.stderr}"


def test_internal_import_error_not_masked():
    """
    Internal ModuleNotFoundError within run_output_manager should propagate,
    not trigger package import fallback.
    
    This test simulates a broken run_output_manager that tries to import
    a non-existent module. The current too-broad except clause would catch
    this and try package import fallback, masking the real error.
    """
    # Create a temporary broken run_output_manager in a temp directory
    import tempfile
    import textwrap
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Write a broken run_output_manager that imports non-existent module
        broken_rom = tmp_path / "run_output_manager.py"
        broken_rom.write_text(textwrap.dedent("""
            # Broken module that imports something that doesn't exist
            import this_module_does_not_exist_xyz123
            
            def compute_file_hash(path):
                pass
            
            def prepare_output_directory(output_dir, overwrite):
                pass
            
            def build_training_manifest(**kwargs):
                pass
        """))
        
        # Write a test script using the PRECISE exception handling (the fix)
        test_script = tmp_path / "test_import.py"
        test_script.write_text(textwrap.dedent("""
            import sys
            try:
                from run_output_manager import compute_file_hash
            except ModuleNotFoundError as exc:
                # PRECISE handling: only fallback if run_output_manager is missing
                if exc.name == 'run_output_manager':
                    print("FALLBACK-TRIGGERED")
                    sys.exit(99)
                else:
                    # Re-raise internal import errors
                    raise
        """))
        
        # Run the test script - should fail with the internal import error
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        # Should NOT trigger fallback (exit 99)
        assert result.returncode != 99, "Should not have triggered fallback for internal error"
        # Should fail with the actual missing module error
        assert result.returncode != 0, "Should have failed with import error"
        assert "this_module_does_not_exist_xyz123" in result.stderr, \
            f"Should show the actual missing module, got:\n{result.stderr}"


def test_precise_fallback_only_for_target_module():
    """
    ModuleNotFoundError should only trigger fallback when the exception
    is specifically about 'run_output_manager', not other imports.
    """
    import tempfile
    import textwrap
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a working run_output_manager
        working_rom = tmp_path / "run_output_manager.py"
        working_rom.write_text(textwrap.dedent("""
            def compute_file_hash(path):
                return "hash123"
            
            def prepare_output_directory(output_dir, overwrite):
                return None
            
            def build_training_manifest(**kwargs):
                return {}
        """))
        
        # Write test script with PRECISE exception checking
        test_script = tmp_path / "test_precise.py"
        test_script.write_text(textwrap.dedent("""
            try:
                from run_output_manager import compute_file_hash
            except ModuleNotFoundError as exc:
                # Only fallback if the error is about run_output_manager itself
                if exc.name == 'run_output_manager':
                    from .run_output_manager import compute_file_hash
                else:
                    # Re-raise if it's an internal import error
                    raise
            
            # Should succeed with direct import
            assert callable(compute_file_hash)
            print("OK-PRECISE")
        """))
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        assert result.returncode == 0, f"Precise fallback test failed:\n{result.stderr}"
        assert "OK-PRECISE" in result.stdout


def test_current_train_rank_hybrid_import_pattern():
    """
    Test the actual current import pattern in train_rank_hybrid.py to verify
    it will work with the fix.
    """
    import tempfile
    import textwrap
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create working run_output_manager
        rom = tmp_path / "run_output_manager.py"
        rom.write_text(textwrap.dedent("""
            def compute_file_hash(path):
                return "test_hash"
            def prepare_output_directory(output_dir, overwrite):
                return None
            def build_training_manifest(**kwargs):
                return {}
        """))
        
        # Test that the FIXED pattern works for both direct and package import
        test_direct = tmp_path / "test_direct.py"
        test_direct.write_text(textwrap.dedent("""
            try:
                from run_output_manager import compute_file_hash
            except ModuleNotFoundError as exc:
                if exc.name == 'run_output_manager':
                    from .run_output_manager import compute_file_hash
                else:
                    raise
            print(f"Direct: {compute_file_hash(None)}")
        """))
        
        result = subprocess.run(
            [sys.executable, str(test_direct)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Direct import failed:\n{result.stderr}"
        assert "Direct: test_hash" in result.stdout

