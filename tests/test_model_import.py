# tests/test_model_import.py
"""
Basic import test for the rrs3c package.

This prepends the repository 'src/' directory to sys.path so the local
package can be imported when running pytest from the repository root
without requiring an editable install (pip install -e .).
"""

import sys
from pathlib import Path

# Ensure repo/src is on sys.path for tests run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Import after sys.path modification (intentional); silence E402 linter rule.
from rrs3c.model import rrs_model_3C  # noqa: E402


def test_model_import():
    """Construct the model and check it exposes the expected API."""
    data_folder = str(DATA_DIR)  # use src/ as a simple data-folder placeholder in tests
    model = rrs_model_3C(data_folder=data_folder)
    assert hasattr(model, "fit_LtEs")
