import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_matcher_fresh_process_prediction():
    code = """
import joblib
import numpy as np
pack = joblib.load('models/matcher.joblib')
model = pack['model']
features = np.ascontiguousarray(np.random.default_rng(7).random((2, len(pack['feature_names']))), dtype=np.float32)
first = model.predict_proba(features)
second = model.predict_proba(features)
assert first.shape == (2, 2)
assert np.isfinite(first).all()
assert np.array_equal(first, second)
print('PASS')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
