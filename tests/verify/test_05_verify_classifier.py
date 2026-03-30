"""Tests for verify/05_verify_classifier.py — geographic overfitting check."""
import importlib.util
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest
import rioxarray  # noqa: F401

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_prob_raster(path: Path, values: np.ndarray):
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    H, W = values.shape
    transform = from_bounds(700000, -1650000, 750000, -1600000, W, H)
    with rasterio.open(
        str(path), "w",
        driver="GTiff", dtype="float32", count=1,
        width=W, height=H, transform=transform,
        crs=CRS.from_epsg(7854),
    ) as dst:
        dst.write(values[np.newaxis])


def _write_model_cache(cache_path: Path, top_feature: str, cv_mean: float = 0.90):
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.rand(100, 8)
    y = (X[:, 0] > 0.5).astype(int)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    # Override feature importances to make desired feature top
    feature_names = [
        "ndvi_anomaly", "flowering_index", "vv_db", "vh_db",
        "ndvi_median", "glcm_contrast", "glcm_homogeneity", "dist_to_watercourse",
    ]
    idx = feature_names.index(top_feature)
    importances = np.zeros(len(feature_names))
    importances[idx] = 1.0
    cv_scores = np.array([cv_mean] * 5)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_names": feature_names,
            "cv_scores": cv_scores,
            "feature_importances": importances,
        }, f)


def test_geographic_top_feature_fails(tmp_dirs):
    """dist_to_watercourse as top feature must fail with geographic overfitting message."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    prob_path = config.probability_raster_path(config.YEAR)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    _write_prob_raster(prob_path, np.full((10, 10), 0.5, dtype=np.float32))

    cache_path = Path(config.CACHE_DIR) / f"rf_model_{config.YEAR}.pkl"
    _write_model_cache(cache_path, top_feature="dist_to_watercourse", cv_mean=0.92)

    script = PROJECT_ROOT / "verify" / "05_verify_classifier.py"
    mod = _load_module(script, "verify05")

    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 2


def test_good_top_feature_passes(tmp_dirs):
    """ndvi_anomaly as top feature should pass."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    prob_path = config.probability_raster_path(config.YEAR)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    _write_prob_raster(prob_path, np.full((10, 10), 0.5, dtype=np.float32))

    cache_path = Path(config.CACHE_DIR) / f"rf_model_{config.YEAR}.pkl"
    _write_model_cache(cache_path, top_feature="ndvi_anomaly", cv_mean=0.92)

    script = PROJECT_ROOT / "verify" / "05_verify_classifier.py"
    mod = _load_module(script, "verify05_good")
    mod.main()  # should not raise SystemExit(2)
