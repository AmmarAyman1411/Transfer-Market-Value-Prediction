"""Refresh data and retrain the model."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import build_training_dataset
import subprocess

if __name__ == "__main__":
    # Force rebuild
    import os as _os
    tp = _os.path.join(_os.path.dirname(__file__), "data", "training.parquet")
    if _os.path.exists(tp):
        _os.remove(tp)
    df = build_training_dataset()
    print("Dataset rows:", len(df))
    # retrain
    subprocess.run([sys.executable, "src/train.py"], check=True)
    print("Refresh complete.")
