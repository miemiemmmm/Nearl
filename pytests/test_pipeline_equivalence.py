"""
End-to-end pipeline equivalence tests.

These tests run the *full* Nearl featurization pipeline (query -> run -> dump)
and capture the resulting output arrays. Their purpose is to verify that
optimizations to the CPU-side phases (query, cache, dump) do not change the
numerical output of the GPU compute.

Workflow
--------
1. Before making any changes, generate a baseline:

       python -m pytest pytests/test_pipeline_equivalence.py -k baseline

   This runs the pipeline with the current code and saves the output arrays to
   a baseline file (``.npz``) next to this test.

2. Make your optimization changes.

3. Re-run the equivalence tests:

       python -m pytest pytests/test_pipeline_equivalence.py

   The tests re-run the pipeline and compare every output array against the
   baseline with ``np.allclose``. Any numerical drift caused by the changes
   will fail the test.

The tests cover both a DensityFlow feature and a MarchingObservers feature, so
both GPU code paths are exercised. They therefore need the compiled CUDA
extension and are skipped without it -- a GPU-less runner cannot say anything
about numerical equivalence.
"""

import os
import warnings

import numpy as np
import pytest

import nearl
import nearl.features
import nearl.featurizer
import nearl.io

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Small grid + small time window keep the test fast while still exercising the
# full pipeline (query, GPU compute, dump).
DIMS = 8
LENGTHS = 8
TIME_WINDOW = 5
SIGMA = 1.5
CUTOFF = 3.5


def _extension_built():
    try:
        from nearl import all_actions  # noqa: F401
    except ImportError:
        return False
    return True


# Every test here runs real kernels; on the CPU-only CI runner there are none.
pytestmark = pytest.mark.skipif(
    not _extension_built(),
    reason="nearl.all_actions is not built (no nvcc in this environment)",
)

# Baseline file lives next to this test module.
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "pipeline_baseline.npz")

# The example data directory (downloaded once by get_example_data).
EXAMPLE_DATA_DIR = os.environ.get("NEARL_TEST_DATA", "/tmp/nearl_test")


def _build_featurizer(outfile):
    """Build a Featurizer with a DensityFlow and a MarchingObservers feature."""
    loader = nearl.io.TrajectoryLoader(
        nearl.get_example_data(EXAMPLE_DATA_DIR)["MINI_TRAJSET"]
    )
    featurizer = nearl.featurizer.Featurizer(
        {
            "dimensions": DIMS,
            "lengths": LENGTHS,
            "time_window": TIME_WINDOW,
            "sigma": SIGMA,
            "cutoff": CUTOFF,
            "outfile": outfile,
        }
    )
    featurizer.register_features(
        [
            nearl.features.DensityFlow(
                selection="!:T3P",
                agg="standard_deviation",
                weight_type="mass",
                outkey="df",
            ),
            nearl.features.MarchingObservers(
                selection="!:T3P",
                obs="density",
                agg="mean",
                weight_type="mass",
                outkey="obs",
            ),
        ]
    )
    featurizer.register_trajloader(loader)
    featurizer.register_focus([":LIG"], "mask")
    return featurizer


def _run_pipeline(outfile):
    """Run the full pipeline and return a dict of {outkey: ndarray}."""
    if os.path.exists(outfile):
        os.remove(outfile)
    featurizer = _build_featurizer(outfile)
    featurizer.run()

    import h5py

    results = {}
    with h5py.File(outfile, "r") as f:
        for key in f:
            if isinstance(f[key], h5py.Dataset):
                results[key] = np.array(f[key])
    return results


# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------


@pytest.mark.baseline
def test_generate_baseline(tmp_path):
    """Run the pipeline and save the output arrays as the baseline."""
    outfile = str(tmp_path / "baseline.h5")
    results = _run_pipeline(outfile)

    assert "df" in results, "DensityFlow output missing"
    assert "obs" in results, "MarchingObservers output missing"

    np.savez(BASELINE_PATH, **results)
    print(f"\nSaved baseline to {BASELINE_PATH}")
    for key, arr in results.items():
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")


# ---------------------------------------------------------------------------
# Equivalence tests (run against the baseline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline():
    """Load the baseline output arrays, skipping if not generated yet."""
    if not os.path.exists(BASELINE_PATH):
        pytest.skip(
            "No baseline found. Generate it first with: "
            "python -m pytest pytests/test_pipeline_equivalence.py -k baseline"
        )
    return np.load(BASELINE_PATH)


def test_pipeline_output_matches_baseline(baseline, tmp_path):
    """The full pipeline output must match the baseline exactly."""
    outfile = str(tmp_path / "current.h5")
    results = _run_pipeline(outfile)

    assert set(results.keys()) == set(baseline.files), (
        f"Output keys changed: got {sorted(results.keys())}, "
        f"expected {sorted(baseline.files)}"
    )

    for key in baseline.files:
        current = results[key]
        expected = baseline[key]
        assert current.shape == expected.shape, (
            f"Shape mismatch for '{key}': got {current.shape}, "
            f"expected {expected.shape}"
        )
        np.testing.assert_allclose(
            current,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Numerical drift detected for output '{key}'",
        )


def test_pipeline_output_is_finite(baseline, tmp_path):
    """Output arrays must be finite (no NaN/Inf) and non-trivial."""
    outfile = str(tmp_path / "finite.h5")
    results = _run_pipeline(outfile)

    for key, arr in results.items():
        assert np.all(np.isfinite(arr)), f"Output '{key}' contains NaN/Inf"
        assert np.any(arr != 0), f"Output '{key}' is all zeros (suspicious)"


def test_pipeline_output_shape(baseline, tmp_path):
    """Output shape must be (n_slices, DIMS, DIMS, DIMS)."""
    outfile = str(tmp_path / "shape.h5")
    results = _run_pipeline(outfile)

    for key, arr in results.items():
        assert arr.ndim == 4, f"Output '{key}' should be 4-D, got {arr.ndim}-D"
        assert arr.shape[1:] == (DIMS, DIMS, DIMS), (
            f"Output '{key}' grid dims {arr.shape[1:]} != {(DIMS, DIMS, DIMS)}"
        )
