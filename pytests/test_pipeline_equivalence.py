"""
End-to-end pipeline equivalence tests.

These tests run the *full* Nearl featurization pipeline (query -> run -> dump)
and capture the resulting output arrays. Their purpose is to verify that
optimizations to the CPU-side phases (query, cache, dump) do not change the
numerical output of the GPU compute.

Workflow
--------
1. Before making any changes, generate a baseline:

       NEARL_GENERATE_BASELINE=1 python -m pytest pytests/test_pipeline_equivalence.py -k baseline

   This runs the pipeline with the current code and saves the output arrays to
   a baseline file (``.npz``) next to this test. The baseline test is skipped
   by default so a plain ``pytest`` run never regenerates (overwrites) the
   baseline.

2. Make your optimization changes.

3. Re-run the equivalence tests:

       python -m pytest pytests/test_pipeline_equivalence.py

   The tests re-run the pipeline and compare every output array against the
   baseline with ``np.allclose``. Any numerical drift caused by the changes
   will fail the test.

The tests cover both a DensityFlow feature and a MarchingObservers feature, so
both GPU code paths are exercised.
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

# Baseline file lives next to this test module.
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "pipeline_baseline.npz")

# The example data directory (downloaded once by get_example_data).
EXAMPLE_DATA_DIR = os.environ.get("NEARL_TEST_DATA", "/tmp/nearl_test")


def _build_featurizer(outfile, producer_threads=1):
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
            "producer_threads": producer_threads,
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


def _run_pipeline(outfile, producer_threads=1):
    """Run the full pipeline and return a dict of {outkey: ndarray}."""
    if os.path.exists(outfile):
        os.remove(outfile)
    featurizer = _build_featurizer(outfile, producer_threads=producer_threads)
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
    """Run the pipeline and save the output arrays as the baseline.

    Skipped unless ``NEARL_GENERATE_BASELINE=1`` is set, so a plain ``pytest``
    run never overwrites the committed baseline file.
    """
    if os.environ.get("NEARL_GENERATE_BASELINE") != "1":
        pytest.skip(
            "Baseline generation is opt-in. Set NEARL_GENERATE_BASELINE=1 to "
            "regenerate pytests/pipeline_baseline.npz."
        )

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


def test_pipeline_output_is_finite(tmp_path):
    """Output arrays must be finite (no NaN/Inf) and non-trivial."""
    outfile = str(tmp_path / "finite.h5")
    results = _run_pipeline(outfile)

    for key, arr in results.items():
        assert np.all(np.isfinite(arr)), f"Output '{key}' contains NaN/Inf"
        assert np.any(arr != 0), f"Output '{key}' is all zeros (suspicious)"


def test_pipeline_output_shape(tmp_path):
    """Output shape must be (n_slices, DIMS, DIMS, DIMS)."""
    outfile = str(tmp_path / "shape.h5")
    results = _run_pipeline(outfile)

    for key, arr in results.items():
        assert arr.ndim == 4, f"Output '{key}' should be 4-D, got {arr.ndim}-D"
        assert arr.shape[1:] == (DIMS, DIMS, DIMS), (
            f"Output '{key}' grid dims {arr.shape[1:]} != {(DIMS, DIMS, DIMS)}"
        )


@pytest.mark.parametrize("producer_threads", [2, 4])
def test_multiproducer_output_matches_baseline(baseline, tmp_path, producer_threads):
    """Multi-producer CPU scheduling must not change the numerical output.

    The R1 optimization parallelizes the CPU producer across trajectories by
    giving each producer thread a private clone of the feature set. The GPU
    consumer stays single-threaded and uses the original features. This test
    verifies that running with ``producer_threads > 1`` produces output
    identical to the single-producer baseline.
    """
    outfile = str(tmp_path / f"multi_{producer_threads}.h5")
    results = _run_pipeline(outfile, producer_threads=producer_threads)

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
            err_msg=f"Numerical drift detected for output '{key}' with "
            f"producer_threads={producer_threads}",
        )
