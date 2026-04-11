"""Unit test for main.py - compares output to sanctioned output.

Run from project root with 

        python -m unittest test.test_main

Or run the test directly as a script with
 
        python test/test_main.py
        

Run all tests with

        python -m unittest discover -s test -p "test_*.py" -v


"""

import unittest
import subprocess
import sys
from pathlib import Path


class TestMainOutput(unittest.TestCase):
    """Test that main.py produces the expected output."""

    def test_main_output_matches_sanctioned(self):
        """Run main.py and compare its output to the sanctioned output file."""
        # Get project root directory (parent of test folder)
        test_dir = Path(__file__).parent
        project_root = test_dir.parent
        
        # Path to main.py and sanctioned output
        main_script = project_root / "main.py"
        sanctioned_output_file = (
            project_root / "data" / "sanctioned-output" / "AAPL_MSFT_GOOGL_AMZNSPY.txt"
        )
        
        # Verify files exist
        self.assertTrue(main_script.exists(), f"main.py not found at {main_script}")
        self.assertTrue(
            sanctioned_output_file.exists(),
            f"Sanctioned output file not found at {sanctioned_output_file}",
        )
        
        # Run main.py and capture output
        result = subprocess.run(
            [sys.executable, str(main_script)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        
        # Check that the script ran successfully
        self.assertEqual(
            result.returncode,
            0,
            f"main.py exited with code {result.returncode}\nStderr: {result.stderr}",
        )
        
        # Read the sanctioned output
        with open(sanctioned_output_file, "r") as f:
            expected_output = f.read()
        
        # Compare outputs (strip trailing whitespace from each line for robustness)
        actual_lines = [line.rstrip() for line in result.stdout.splitlines()]
        expected_lines = [line.rstrip() for line in expected_output.splitlines()]
        
        self.assertEqual(
            actual_lines,
            expected_lines,
            f"Output mismatch.\nExpected:\n{expected_output}\n\nActual:\n{result.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
