# tests/test_model_import.py
"""
Simple import test that ensures the rrs3c package is importable in CI/local dev
without requiring a packaged install. This prepends the repository's `src/`
directory to sys.path so `import rrs3c` works when running pytest from the repo root.
"""
import sys
from pathlib import Path
from rrs3c.model import rrs_model_3C

# Insert repo/src at front of sys.path so tests can import the local package.
# This keeps tests independent from having to 'pip install -e .' first.
ROOT = Path(__file__).resolve().parents[1]  # repo root
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Now import the package and construct the model


def test_model_import():
    data_folder = str(SRC)  # or point to a test-data folder if you have one
    model = rrs_model_3C(data_folder=data_folder)
    assert hasattr(model, "fit_LtEs")
