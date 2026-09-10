"""
Producer threads must not share feature state through a cloned feature.

``Featurizer._clone_features`` gives every producer thread its own
``copy.copy`` of each feature, because ``cache``/``query`` write
trajectory-scoped state (``selected``, ``resids``, ``cached_array``) onto the
feature. A shallow copy also copies *instance attributes*, and an instance
attribute that shadows a method -- what a profiler or any monkeypatch installs
-- is a closure bound to the original. The clone then dispatches straight back
into the original and every producer shares one feature's state.

The failure is a shape mismatch, because trajectories in a set rarely have
identical atom counts:

    ValueError: operands could not be broadcast together with
                shapes (14534,) (14488,)

raised from ``final_mask * self.selected`` once one thread has overwritten the
other's cache mid-query.

``test_multiproducer_parity`` cannot see this: its fake trajectories all have
the same atom count, so clobbered state still broadcasts, and it never patches
an instance. Everything here therefore uses *different* atom counts per
trajectory and installs an instance-level wrapper the way a profiler does.
"""

import copy
import threading

import numpy as np
import pytest

import nearl.featurizer
from nearl.features import Feature

# Deliberately unequal, like a real trajectory set: a clobbered `selected`
# cannot broadcast against the other topology's mask.
SIZE_A, SIZE_B = 40, 37


class _Atom:
    def __init__(self, index):
        self.resid = index
        self.atomic_number = 6


class _Top:
    """Enough topology for Feature.cache and top.select."""

    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.atoms = [_Atom(i) for i in range(n_atoms)]

    def select(self, mask):
        return np.arange(self.n_atoms)


class _Traj:
    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.top = _Top(n_atoms)


class _Featurizer:
    """The two attributes ``_clone_features`` actually touches."""

    classname = "Featurizer"

    def __init__(self, features):
        self.FEATURESPACE = list(features)


def clone_features(features):
    """Call the real ``Featurizer._clone_features`` on a bare holder."""
    return nearl.featurizer.Featurizer._clone_features(_Featurizer(features))


def make_feature():
    """A Feature that reaches ``final_mask * self.selected`` in query().

    center and lengths are read-only, derived when dims and spacing are both
    set, so they are re-assigned here to trigger that derivation. Without them
    query() short-circuits before the selected mask is ever applied.
    """
    feat = Feature(dims=4, spacing=1.0, cutoff=2.0, sigma=1.0, outkey="probe")
    feat.spacing = 1.0
    feat.dims = 4
    feat.padding = 0.0
    feat.byres = False
    feat.selection = "!:T3P"  # not None, so the selected mask is applied
    assert feat.center is not None and feat.lengths is not None
    return feat


def instance_wrap(feat, name, seen=None):
    """Patch an instance method the way benchmarks/profiling_host_device.py did.

    The closure captures the *bound* method of ``feat``, which is precisely what
    makes a shallow copy dangerous.
    """
    original = getattr(feat, name)

    def timed(*args, **kwargs):
        if seen is not None:
            seen.append(name)
        return original(*args, **kwargs)

    setattr(feat, name, timed)


def query_once(feat, traj, rng):
    coords = rng.random((traj.n_atoms, 3)) * 4.0
    return feat.query(traj.top, coords, np.zeros(3))


###############################################################################
# The mechanism: a clone must never execute on the original
###############################################################################


def test_a_clone_does_not_execute_on_the_original():
    """
    The precise defect. With an instance-level override in place, a shallow
    copy carries the wrapper across, and the wrapper's closure still points at
    the original -- so ``clone.cache(...)`` runs ``Feature.cache`` with
    ``self`` bound to the *original* feature.
    """
    ran_on = []

    class _Probe(Feature):
        def cache(self, trajectory):
            ran_on.append(self)

    original = _Probe(dims=4, spacing=1.0, cutoff=2.0, sigma=1.0, outkey="probe")
    instance_wrap(original, "cache")

    (clone,) = clone_features([original])
    assert clone is not original

    clone.cache(_Traj(SIZE_A))
    assert ran_on, "Feature.cache never ran"
    assert ran_on[-1] is clone, (
        "the clone dispatched into the original feature, so every producer "
        "thread would share one feature's trajectory state"
    )


def test_cloning_reports_the_overrides_it_drops(caplog):
    """Dropping a caller's instrumentation is silent data loss unless it warns."""
    feat = make_feature()
    instance_wrap(feat, "cache")
    instance_wrap(feat, "query")

    with caplog.at_level("WARNING"):
        clone_features([feat])

    dropped = [
        r.getMessage() for r in caplog.records if "instance-level" in r.getMessage()
    ]
    assert len(dropped) == 2, f"expected cache and query to be reported, got {dropped}"
    assert any("cache" in m for m in dropped)
    assert any("query" in m for m in dropped)


def test_the_original_keeps_its_own_override():
    """Cloning must not disturb the feature the caller still holds."""
    feat = make_feature()
    seen = []
    instance_wrap(feat, "cache", seen)

    clone_features([feat])

    feat.cache(_Traj(SIZE_A))
    assert seen == ["cache"], "cloning stripped the override from the original too"


###############################################################################
# The consequence: per-thread trajectory state stays consistent
###############################################################################


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain", "instance-wrapped"])
def test_clones_hold_independent_selection_state(wrapped):
    """Two clones cached on differently sized topologies must not share state."""
    feat = make_feature()
    if wrapped:
        instance_wrap(feat, "cache")

    clone_a, clone_b = clone_features([feat] * 2)
    clone_a.cache(_Traj(SIZE_A))
    clone_b.cache(_Traj(SIZE_B))

    assert len(clone_a.selected) == SIZE_A
    assert len(clone_b.selected) == SIZE_B
    assert len(clone_a.resids) == SIZE_A
    assert len(clone_b.resids) == SIZE_B


def test_concurrent_producers_do_not_corrupt_each_others_cache():
    """
    The regression test for the reported crash.

    Two threads, each with its own clone, repeatedly cache and query a
    differently sized topology -- the shape of the real producer loop. If the
    clones share state, one thread's ``selected`` lands on the other's feature
    and ``final_mask * self.selected`` cannot broadcast.
    """
    feat = make_feature()
    instance_wrap(feat, "cache")
    instance_wrap(feat, "query")
    clone_a, clone_b = clone_features([feat] * 2)

    errors = []
    start = threading.Barrier(2)

    def worker(clone, n_atoms, seed):
        traj = _Traj(n_atoms)
        rng = np.random.default_rng(seed)
        try:
            start.wait()
            for _ in range(60):
                clone.cache(traj)
                mask, _coords = query_once(clone, traj, rng)
                # Whatever the interleaving, this thread's own view must stay
                # self-consistent with its own topology.
                assert len(mask) == n_atoms
                assert len(clone.selected) == n_atoms
        except Exception as exc:  # surfaced from the main thread below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(clone_a, SIZE_A, 1)),
        threading.Thread(target=worker, args=(clone_b, SIZE_B, 2)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"producer threads corrupted each other: {errors[0]!r}"


def test_a_shallow_copy_alone_would_not_be_enough():
    """
    Pins why ``_clone_features`` cannot just be ``copy.copy``.

    This is the behaviour the guard exists to prevent, asserted directly so the
    guard cannot be removed and quietly replaced by a bare shallow copy.
    """
    ran_on = []

    class _Probe(Feature):
        def cache(self, trajectory):
            ran_on.append(self)

    original = _Probe(dims=4, spacing=1.0, cutoff=2.0, sigma=1.0, outkey="probe")
    instance_wrap(original, "cache")

    naive = copy.copy(original)
    naive.cache(_Traj(SIZE_A))
    assert ran_on[-1] is original, (
        "copy.copy no longer carries the instance override; if that changed, "
        "the guard in _clone_features may no longer be needed"
    )
