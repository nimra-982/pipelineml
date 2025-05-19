
import pytest
from src.preprocess import load_data, preprocess_data

def test_no_missing_values():
    df = load_data()
    assert df.isna().sum().sum() == 0

def test_feature_scaling():
    df = load_data()
    X_scaled, _ = preprocess_data(df)
    assert X_scaled.mean() == pytest.approx(0, abs=0.1)
    assert X_scaled.std() == pytest.approx(1, abs=0.1)

