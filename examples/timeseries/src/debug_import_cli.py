# debug_import_cli.py
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

here = Path(__file__).resolve().parent
print("DEBUG: script dir:", here)
print("DEBUG: cwd:", Path.cwd())
print("DEBUG: python executable:", sys.executable)
print("\nDEBUG: sys.path (first 12 entries):")
for i, p in enumerate(sys.path[:12]):
    print(f"  {i:2d}: {p}")

# Find repo root by looking for src/rrs3c
repo_root = None
for p in here.parents:
    if (p / "src" / "rrs3c").exists():
        repo_root = p
        break

print("\nDEBUG: detected repo_root:", repo_root)
if repo_root is not None:
    print(
        "DEBUG: src path exists:", (repo_root / "src").exists(), "->", repo_root / "src"
    )
    try:
        print("DEBUG: listing src/rrs3c:")
        for x in sorted((repo_root / "src" / "rrs3c").iterdir()):
            print("   ", x.name)
    except Exception as e:
        print("  (listing failed):", e)

# Try to import package normally
print("\nTrying `import rrs3c.model` ...")
try:
    import rrs3c.model as m

    print(
        "OK: imported rrs3c.model as", m.__name__, "from", getattr(m, "__file__", None)
    )
except Exception:
    print("FAILED to import rrs3c.model:")
    traceback.print_exc()

# Try direct file import fallback (point to expected file)
candidate = None
if repo_root is not None:
    candidate = repo_root / "src" / "rrs3c" / "model.py"
print(
    "\nCandidate model.py:",
    candidate,
    "exists?:",
    candidate.exists() if candidate is not None else None,
)

if candidate and candidate.exists():
    print("Trying importlib to load model.py directly ...")
    try:
        spec = importlib.util.spec_from_file_location(
            "rrs3c_model_fallback", str(candidate)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(
            "OK: loaded module name:",
            module.__name__,
            "file:",
            getattr(module, "__file__", None),
        )
        print("Has rrs_model_3C:", hasattr(module, "rrs_model_3C"))
    except Exception:
        print("FAILED to load model.py directly:")
        traceback.print_exc()
else:
    print("No candidate model.py to try.")
