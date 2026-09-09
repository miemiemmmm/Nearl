"""
Hermetic multi-producer parity tests.

These tests verify that the R1 multi-producer concurrency path
(``producer_threads > 1``) schedules *exactly* the same set of GPU tasks as the
single-producer path (``producer_threads == 1``).

Why this matters
----------------
The multi-producer path splits the trajectories across several background
``_ProducerWorker`` threads, each with a private clone of the feature set and a
private copy of the trajectory-scoped state. A bug in that path (a dropped or
duplicated task, a clobbered focal point, a wrong trajectory index) would change
the *number* of tasks or the *payload* of the tasks handed to the GPU consumer,
even if the numerical output happened to look plausible.

These tests are deliberately hermetic:

* The trajectories are tiny in-memory stand-ins (no files on disk, no example
  data download).
* The feature's ``run``/``dump`` are stubbed out, so no CUDA extension and no
  GPU are required.
* There is no baseline file to regenerate.

What is asserted
----------------
For each ``producer_threads`` value, we run the full ``Featurizer.run()``
pipeline (producers -> prefetch buffer -> consumer) and capture every
``(feature, queried)`` task the consumer receives. We then assert:

1. The **task count** is identical to the single-producer run (and equals the
   analytically expected count), catching dropped/duplicated tasks.
2. The **multiset of payloads** is identical to the single-producer run. The
   *order* is not compared because concurrent producers interleave tasks on the
   buffer nondeterministically; only the set of tasks must match.
"""

import numpy as np
import pytest

import nearl.features
import nearl.featurizer
from nearl.io.trajloader import TrajectoryLoader

# ---------------------------------------------------------------------------
# In-memory stand-ins
# ---------------------------------------------------------------------------


class _FakeAtom:
    def __init__(self, index):
        self.resid = index
        self.atomic_number = 6


class _FakeTop:
    """Minimal topology: enough for ``Feature.cache`` and ``top.select``."""

    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.atoms = [_FakeAtom(i) for i in range(n_atoms)]

    def select(self, mask):
        # A mask that selects every atom, so focal parsing never fails.
        return np.arange(self.n_atoms)


class _FakeTraj:
    """Stand-in trajectory so the pipeline needs no real files on disk."""

    def __init__(self, traj_file, top_file, **kwargs):
        self.traj_file = traj_file
        self.top_file = top_file
        self.identity = kwargs.get("identity", traj_file)
        self.n_frames = 4
        self.n_atoms = 3
        self.top = _FakeTop(self.n_atoms)
        # Deterministic per-trajectory coordinates.
        rng = np.random.default_rng(
            int.from_bytes(self.identity.encode("utf-8"), "little") % (2**32)
        )
        self.xyz = rng.random((self.n_frames, self.n_atoms, 3))


class _FakeFeature(nearl.features.Feature):
    """A feature whose CPU phases run for real but whose GPU phases are stubs.

    ``cache``/``query`` exercise the producer-side scheduling logic; ``run``
    records the consumed task payload instead of launching a CUDA kernel, and
    ``dump`` is a no-op so no HDF5 file is written.
    """

    def __init__(self, outkey, consumed):
        super().__init__(
            dims=4,
            spacing=1.0,
            cutoff=2.0,
            sigma=1.0,
            outkey=outkey,
        )
        self._consumed = consumed

    def cache(self, trajectory):
        self.resids = np.arange(trajectory.n_atoms)
        self.atomic_numbers = np.full(trajectory.n_atoms, 6)
        self.selected = np.full(trajectory.n_atoms, True)

    def query(self, topology, frame_coords, focal_point):
        # Deterministic payload derived from the inputs, so any clobbering of
        # per-thread state would show up as a different payload.
        return (
            np.asarray(frame_coords, dtype=np.float32).copy(),
            np.asarray(focal_point, dtype=np.float32).copy(),
        )

    def run(self, coords, weights):
        # Record the task the consumer received.
        self._consumed.append((self.outkey, coords.copy(), weights.copy()))
        return np.zeros(self.dims, dtype=np.float32)

    def dump(self, result):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TRAJS = 3
N_FRAMES = 4
TIME_WINDOW = 2
N_FOCAL = 1
N_FEATURES = 1
# SLICENUMBER = N_FRAMES // TIME_WINDOW
EXPECTED_TASKS = N_TRAJS * (N_FRAMES // TIME_WINDOW) * N_FOCAL * N_FEATURES


def _build_featurizer(producer_threads, consumed):
    featurizer = nearl.featurizer.Featurizer(
        parms={
            "dimensions": 4,
            "spacing": 1.0,
            "time_window": TIME_WINDOW,
            "producer_threads": producer_threads,
            "device_context": False,
        }
    )
    featurizer.register_feature(_FakeFeature(outkey="fake", consumed=consumed))
    featurizer.register_focus(np.array([[0.0, 0.0, 0.0]]), "absolute")
    loader = TrajectoryLoader(
        [("t0", "t0"), ("t1", "t1"), ("t2", "t2")],
        trajtype=_FakeTraj,
        trajids=["t0", "t1", "t2"],
    )
    featurizer.register_trajloader(loader)
    return featurizer


def _run(producer_threads):
    """Run the full pipeline and return the list of consumed tasks."""
    consumed = []
    featurizer = _build_featurizer(producer_threads, consumed)
    featurizer.run()
    return consumed


def _canonical(task):
    """A sortable, order-independent key for a consumed task."""
    outkey, coords, weights = task
    return (outkey, coords.tobytes(), weights.tobytes())


def _normalize(consumed):
    """Return the consumed tasks as a sorted multiset of canonical keys."""
    return sorted(_canonical(t) for t in consumed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("producer_threads", [2, 3])
def test_multiproducer_task_count_matches_single(producer_threads):
    """Multi-producer scheduling must not drop or duplicate any task."""
    single = _run(1)
    multi = _run(producer_threads)

    assert len(single) == EXPECTED_TASKS, (
        f"single-producer produced {len(single)} tasks, expected {EXPECTED_TASKS}"
    )
    assert len(multi) == len(single), (
        f"producer_threads={producer_threads} produced {len(multi)} tasks, "
        f"single-producer produced {len(single)}"
    )


@pytest.mark.parametrize("producer_threads", [2, 3])
def test_multiproducer_payloads_match_single(producer_threads):
    """Multi-producer scheduling must hand the consumer the same task set."""
    single = _normalize(_run(1))
    multi = _normalize(_run(producer_threads))

    assert multi == single, (
        f"producer_threads={producer_threads} produced a different task set "
        f"than single-producer"
    )


def test_multiproducer_uneven_split_is_covered():
    """Sanity check that the parametrization actually exercises an uneven split.

    With 3 trajectories and producer_threads=2 the split is 2+1, which is the
    case most likely to expose an off-by-one in the chunking.
    """
    assert N_TRAJS % 2 != 0
