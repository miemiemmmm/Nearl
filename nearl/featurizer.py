import contextlib
import copy
import json
import logging
import threading
import time

import numpy as np

from . import config, constants, features, log, utils
from .pipeline import (
    _ERROR,
    _SENTINEL,
    AsyncWriter,
    PipelineCancelled,
    PrefetchBuffer,
)

__all__ = [
    "Featurizer",
]

logger = logging.getLogger(__name__)


class _ProducerWorker:
    """
    Per-thread state for a CPU producer thread.

    The featurizer's trajectory-scoped state (``traj``, ``FOCALPOINTS``,
    ``FRAMESLICES``, ...) and the per-feature caches must not be shared by
    concurrent producers, so each worker owns a private copy plus a private
    clone of the feature set.

    Attributes
    ----------
    traj : pytraj.Trajectory
      The trajectory currently being processed by this worker.
    FRAMENUMBER : int
      Number of frames in the active trajectory.
    SLICENUMBER : int
      Number of frame-slices in the active trajectory.
    FRAMESLICES : list of slice
      The frame-slices of the active trajectory.
    FOCALPOINTS : np.ndarray or None
      Parsed focal points for the active trajectory.
    FOCALNUMBER : int
      Number of focal points per frame-slice.
    frame_slice : slice
      The frame-slice currently being processed.
    features : list of :class:`nearl.features.Feature`
      Private clones of the featurizer's feature set, used for ``cache`` and
      ``query`` so that concurrent workers do not share mutable feature state.
    """

    __slots__ = (
        "FOCALNUMBER",
        "FOCALPOINTS",
        "FRAMENUMBER",
        "FRAMESLICES",
        "SLICENUMBER",
        "features",
        "frame_slice",
        "traj",
    )

    def __init__(self, features):
        self.traj = None
        self.FRAMENUMBER = 0
        self.SLICENUMBER = 0
        self.FRAMESLICES = []
        self.FOCALPOINTS = None
        self.FOCALNUMBER = 0
        self.frame_slice = None
        self.features = features


def wrapper_runner(func, args):
    """
    Take the feature.run methods and its input arguments for multiprocessing

    Parameters
    ----------
    func : function
      The function to be run
    args : list
      The arguments for the function

    """
    return func(*args)


def _run_prepared_tasks(featurizer, tasks, feature_map):
    """Prepare one input ahead while the previous CUDA action is in flight."""
    from .features import DensityFlow, Feature, MarchingObservers

    results = []
    pending = None
    try:
        for (feature, query_args), metadata in zip(tasks, feature_map):
            featurizer.frame_slice = featurizer.FRAMESLICES[metadata[1]]
            async_feature = type(feature).run in (
                Feature.run,
                DensityFlow.run,
                MarchingObservers.run,
            )
            # A custom feature may itself use CUDA during query().
            if pending is not None and not async_feature:
                collect, pending = pending, None
                results.append(collect())
            queried = feature.query(*query_args)
            if pending is not None:
                collect, pending = pending, None
                results.append(collect())
            # Preserve custom run() implementations rather than bypassing them.
            if async_feature:
                pending = feature._dispatch(*queried)
            else:
                results.append(feature.run(*queried))
        if pending is not None:
            collect, pending = pending, None
            results.append(collect())
    finally:
        if pending is not None:
            pending()  # Drain CUDA if preparation of the next input raises.
    return results


class Featurizer:
    """
    Featurizer aims to automate the process of featurization of multiple Features for a batch of structures or trajectories

    Parameters
    ----------
    parms : dict
      A dictionary of parameters for the featurizer

    Attributes
    ----------
    traj : :class:`nearl.Trajectory <nearl.io.traj.Trajectory>` or pytraj.Trajectory
      The trajectory to be processed
    top : pytraj.Topology
      The topology of the trajectory
    dims : np.ndarray
      The dimensions of the 3D grid
    lengths : np.ndarray
      The lengths of the 3D grid
    spacing : float
      The spacing of the 3D grid
    time_window : int, default = 1
      The time window for the trajectory (default is 1), Simple integer.
    frame_slice : slice
      The frame-slice being processed in the trajectory


    FRAMENUMBER : int
      The number of frames in the trajectory to be processed
    FRAMESLICENUMBER : int
      The number of slices of frames in the trajectory
    FRAMESLICES : list
      A list of slices of frames in the trajectory to be processed

    FEATURESPACE : list
      A list of features to be processed
    FEATURENUMBER : int
      The number of features to be processed

    FOCALPOINTS_PROTOTYPE
      The prototype of the focal points
    FOCALPOINTS : np.ndarray
      The focal points to be processed
    FOCALNUMBER : int
      The number of focal points for each frame slice

    Notes
    -----
    Required parameters:

    - **dimensions**: the dimensions of the 3D grid
    - lengths: the lengths of the 3D grid (optional)
    - spacing: the spacing of the 3D grid (optional)
    - time_window: the time window for the trajectory (default is 1), Simple integer.

    The following are optional parameters for features.
    If the initialization of the feature did not explicitly define the following parameters, the following parameters will be inherited from the featurizer:

    - outfile: The output file to dump the parameters and the results
    - sigma: The smoothness of the Gaussian-based feature distribution
    - cutoff: The cutoff distance for the grid-based feature calculation

    """

    def __init__(self, parms=None, **kwargs):
        """
        Initialize the featurizer with the given parameters
        """
        if parms is None:
            parms = {}
        # Check the essential parameters for the featurizer
        # assert "dimensions" in parms, "Please define the 'dimensions' in the parameter set"
        # assert ("lengths" in parms) or ("spacing" in parms), "Please define the 'lengths' or 'spacing' in the parameter set"

        # Basic parameters for the featurizer to communicate with cuda code
        self.__dims = None
        self.__lengths = None
        self.__spacing = None
        if parms.get("dimensions") is not None:
            self.dims = parms.get(
                "dimensions"
            )  # Normalize scalar/list/tuple into a 3-element array
        if "lengths" in parms:
            self.lengths = parms.get(
                "lengths", 16
            )  # Normalize scalar/list/tuple into a 3-element array
            self.__spacing = np.mean(self.lengths / self.dims)
        elif "spacing" in parms:
            self.__spacing = parms.get("spacing", 1.0)
            self.__lengths = (
                self.dims * self.__spacing
            )  # Directly assignment avoid the re-calculation of the spacing

        self.time_window = int(
            parms.get("time_window", 1)
        )  # The time window for the trajectory (default is 1), Simple integer.
        if self.time_window > constants.MAX_FRAME_NUMBER:
            logger.warning(
                f"{self.__class__.__name__}: the time window ({self.time_window}) exceeds "
                f"the {constants.MAX_FRAME_NUMBER} frames the CUDA kernels can aggregate; "
                f"only the first {constants.MAX_FRAME_NUMBER} frames of each slice "
                "will contribute to the dynamic features."
            )

        # Get common feature parameters to hook the features
        self.FEATURE_PARMS = {}
        for key in constants.COMMON_FEATURE_PARMS:
            if key in parms:
                self.FEATURE_PARMS[key] = parms[key]
            elif key in kwargs:
                self.FEATURE_PARMS[key] = kwargs[key]
            else:
                self.FEATURE_PARMS[key] = None

        self.OTHER_PARMS = {}
        for key in parms:
            if key not in constants.COMMON_FEATURE_PARMS:
                self.OTHER_PARMS[key] = parms[key]
            else:
                continue
        for key in kwargs:
            if key not in constants.COMMON_FEATURE_PARMS:
                self.OTHER_PARMS[key] = kwargs[key]
            else:
                continue

        # Derivative parameters from trajectory
        self._traj = None
        self.frame_slice = None
        self.FRAMENUMBER = 0
        self.FRAMESLICENUMBER = 0
        self.FRAMESLICES = []

        # Component I: Feature space
        self.FEATURESPACE = []
        self.FEATURENUMBER = 0

        # Component II: Focal point space
        self.FOCALPOINTS = []
        self.FOCALPOINTS_PROTOTYPE = None
        self.FOCALNUMBER = 0

        # Component III: Trajectory space
        self.TRAJLOADER = None
        self.TRAJECTORYNUMBER = 0

        # Pipelining knobs: capacity of the CPU->GPU prefetch buffer and the
        # async HDF5 writer queue. Larger values allow the CPU producer to run
        # further ahead of the GPU consumer, at the cost of more memory.
        self._prefetch_capacity = int(parms.get("prefetch_capacity", 2))
        self._writer_capacity = int(parms.get("writer_capacity", 16))

        # Background CPU producer threads; each owns a private feature-set clone
        # and processes a disjoint subset of trajectories.
        self._producer_threads = int(parms.get("producer_threads", 2))

        # Accumulate GPU busy time during run(); disable to skip the per-dispatch timing overhead.
        self._gpu_busy_capture = bool(
            parms.get("gpu_busy_capture", kwargs.get("gpu_busy_capture", True))
        )
        features.Feature.gpu_busy_capture = self._gpu_busy_capture

        self.classname = self.__class__.__name__
        if config.verbose():
            log(
                f"{self.classname}: Featurizer is initialized successfully with dimensions: {self.dims} and lengths: {self.lengths}"
            )

        # One CUDA stream and one set of device buffers shared by every kernel
        # call, rather than an allocation per task. Pass device_context=False to
        # measure against the per-call path.
        if parms.get("device_context", kwargs.get("device_context", True)):
            self.init_device_context()

        if "outfile" in parms:
            # Dump the parm dict to that hdf file
            log(
                f"{self.classname}: Dumping the parameters to {parms['outfile']} : {self.parms}"
            )
            utils.dump_dict(parms["outfile"], "featurizer_parms", self.parms)

    def init_device_context(self):
        """
        Bring up the persistent GPU context that the CUDA commands share.

        Failing is not fatal: without the compiled extension or a visible device
        the commands allocate per call, which is what a CPU-only machine does.
        """
        from . import commands

        try:
            commands.init_context()
            logger.debug(f"{self.classname}: persistent GPU context is active")
        except Exception as exc:
            logger.debug(f"{self.classname}: running without a GPU context ({exc})")

    def __str__(self):
        finalstr = f"Feature Number: {self.FEATURENUMBER}; \n"
        for feat in self.FEATURESPACE:
            finalstr += f"Feature: {feat.__str__()}\n"
        return finalstr

    @property
    def parms(self):
        """
        Return the parameters of the featurizer
        """
        return {
            k: v if v is not None else 0
            for k, v in {**self.FEATURE_PARMS, **self.OTHER_PARMS}.items()
        }

    # The most important attributes to determine the size of the 3D grid
    @property
    def dims(self):
        """
        The 3 dimensions of the 3D grid
        """
        return np.array(self.__dims) if self.__dims is not None else None

    @dims.setter
    def dims(self, newdims):
        if isinstance(
            newdims, (int, float, np.float32, np.float64, np.int32, np.int64)
        ):
            self.__dims = np.array([newdims, newdims, newdims], dtype=int)
        elif isinstance(newdims, (list, tuple, np.ndarray)):
            assert len(newdims) == 3, "length should be 3"
            self.__dims = np.array(newdims, dtype=int)
        else:
            raise TypeError(
                "Unexpected data type, either be a number or a list of 3 integers"
            )
        if self.__lengths is not None:
            self.__spacing = np.mean(self.lengths / self.dims)

    # The most important attributes to determine the size of the 3D grid
    @property
    def lengths(self):
        """
        The lengths of the 3D grid in Angstrom
        """
        return self.__lengths

    @lengths.setter
    def lengths(self, new_length):
        if isinstance(
            new_length, (int, float, np.float32, np.float64, np.int32, np.int64)
        ):
            self.__lengths = np.array([new_length] * 3, dtype=float)
        elif isinstance(new_length, (list, tuple, np.ndarray)):
            assert len(new_length) == 3, "length should be 3"
            self.__lengths = np.array(new_length, dtype=float)
        else:
            raise TypeError(
                "Unexpected data type, either be a number or a list of 3 floats"
            )
        if self.__dims is not None:
            self.__spacing = np.mean(self.lengths / self.dims)

    @property
    def spacing(self):
        """
        The spacing (also the resolution) between grid points of the 3D grid
        """
        return self.__spacing

    @property
    def traj(self):
        """
        The trajectory being processed.

        """
        return self._traj

    @traj.setter
    def traj(self, the_traj):
        self._traj = the_traj
        self.FRAMENUMBER = the_traj.n_frames
        self.SLICENUMBER = self.FRAMENUMBER // self.time_window
        if self.SLICENUMBER == 0:
            logger.warning(
                f"{self.classname}: No frame slice is available. The trajectory has {self.FRAMENUMBER} frames and the time window is {self.time_window}."
            )
        if self.FRAMENUMBER % self.time_window != 0 and self.FRAMENUMBER != 1:
            logger.warning(
                f"{self.classname}: the number of frames ({self.FRAMENUMBER}) is not divisible by the time window ({self.time_window}). The last few frames will be ignored."
            )
        logger.info(
            f"{self.classname}: Registered {self.SLICENUMBER} slices of frames with {self.time_window} as the time window (frames-per-slice)."
        )
        logger.debug(f"Having {self.SLICENUMBER} frame slices in the trajectory ")
        frame_array = np.array(
            [0, *np.cumsum([self.time_window] * self.SLICENUMBER).tolist()]
        )
        self.FRAMESLICES = [
            np.s_[frame_array[i] : frame_array[i + 1]] for i in range(self.SLICENUMBER)
        ]

    @property
    def top(self):
        """
        The topology of the trajectory being processed.
        """
        return self.traj.top

    def register_feature(self, feature):
        """
        Register a :class:`nearl.features.Feature` to the featurizer

        Parameters
        ----------
        feature : :class:`nearl.features.Feature`

        """
        feature.hook(self)  # Hook the featurizer to the feature
        self.FEATURESPACE.append(feature)
        self.FEATURENUMBER = len(self.FEATURESPACE)
        output_keys = [i.outkey for i in self.FEATURESPACE]
        if len(set(output_keys)) != len(output_keys):
            print(np.unique(output_keys, return_counts=True))
            raise ValueError("The output keys for the features should be unique.")

    def register_features(self, features):
        """
        Register multiple :class:`nearl.features.Feature` in a list or dictionary to the featurizer

        Parameters
        ----------
        features : list-like or dict-like

        """
        if isinstance(features, (list, tuple)):
            for feature in features:
                self.register_feature(feature)
        elif isinstance(features, dict):
            for _, feature in features.items():
                if config.verbose() or config.debug():
                    log(
                        f"{self.classname}: Registering the feature named: {_} from {feature.classname} class"
                    )
                self.register_feature(feature)

    def register_trajloader(self, trajloader):
        """
        Register a trajectory loader to the featurizer for further processing

        Parameters
        ----------
        trajloader : :class:`nearl.io.trajloader.TrajectoryLoader`

        """
        self.TRAJLOADER = trajloader
        self.TRAJECTORYNUMBER = len(trajloader)
        log(f"{self.classname}: Registered {self.TRAJECTORYNUMBER} trajectories")

    def register_focus(self, focus, format):
        """
        Register a set of focal points to the featurizer for further processing

        Parameters
        ----------
        focus : list_like
          The focal points to process
        format : str
          The format of the focal points

        Notes
        -----
        The following 4 formats of focuses are supported:

        - **mask**: provide a selection of atoms (`Amber's selection convention <https://amberhub.chpc.utah.edu/atom-mask-selection-syntax/>`_)
        - **index**: provide a list of atom indices (int)
        - **absolute**: provide a list of 3D coordina tes
        - **json**: provide a json file containing the indexes of the atoms for each trajectory (the key for each trajectory should match the ``feat.identity`` attribute)

        """
        if format == "mask":
            self.FOCALPOINTS_PROTOTYPE = focus
            self.FOCALPOINTS_TYPE = "mask"
            self.FOCALPOINTS = None

        elif format == "absolute":
            assert len(focus.shape) == 2, "The focus should be a 2D array"
            assert focus.shape[1] == 3, "The focus should be a 2D array with 3 columns"
            self.FOCALPOINTS_PROTOTYPE = focus
            self.FOCALPOINTS = focus
            self.FOCALPOINTS_TYPE = "absolute"

        elif format == "index":
            self.FOCALPOINTS_PROTOTYPE = focus
            self.FOCALPOINTS_TYPE = "index"
            self.FOCALPOINTS = None

        elif format == "json":
            self.FOCALPOINTS_PROTOTYPE = focus
            self.FOCALPOINTS_TYPE = "json"
            self.FOCALPOINTS = None

        else:
            raise ValueError(
                f"Unexpected focus format: {format}. Please choose from 'mask', 'absolute', 'index', 'function'"
            )

    def parse_focus(self):
        """
        After registering the active trajectory, parse the focal points for each frame-slice in the ``run`` method.
        The resulting shape will be a 3D array with the shape ``(slice_number, focus_number, 3)``

        Notes
        -----
        Thin wrapper around :meth:`_parse_focus` operating on the featurizer's
        own state; the multi-producer schedule passes a :class:`_ProducerWorker`
        instead.
        """
        return self._parse_focus(self)

    def _parse_focus(self, worker):
        """
        Parse the focal points for each frame-slice of the worker's active
        trajectory, with the shape ``(slice_number, focus_number, 3)``.

        Parameters
        ----------
        worker : :class:`_ProducerWorker` or :class:`Featurizer`
          The object holding the active trajectory and parsed focal points;
          the featurizer itself in the single-producer schedule.
        """
        # For "json" the prototype is a file path, not a focal list; it always writes one focal point per slice.
        focal_number = (
            1 if self.FOCALPOINTS_TYPE == "json" else len(self.FOCALPOINTS_PROTOTYPE)
        )
        worker.FOCALPOINTS = np.full(
            (worker.SLICENUMBER, focal_number, 3),
            99999,
            dtype=np.float32,
        )
        logger.debug(f"Shape of the focal points prototype: {worker.FOCALPOINTS.shape}")
        worker.FOCALNUMBER = focal_number
        if self.FOCALPOINTS_TYPE == "mask":
            # Get the center of geometry for the frames with self.interval
            for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE):
                selection = worker.traj.top.select(mask)
                if len(selection) == 0:
                    log.warning(
                        f"{self.classname}: The trajectory {worker.traj.identity} does not have any atoms in the selection {mask}"
                    )
                    return False
                for fidx in range(worker.SLICENUMBER):
                    frame = worker.traj.xyz[fidx * self.time_window]
                    worker.FOCALPOINTS[fidx, midx] = np.mean(frame[selection], axis=0)
            return 1
        elif self.FOCALPOINTS_TYPE == "index":
            for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE):
                for idx, frame in enumerate(worker.traj.xyz[:: self.time_window]):
                    if idx >= worker.SLICENUMBER:
                        break
                    worker.FOCALPOINTS[idx, midx] = np.mean(frame[mask], axis=0)
            return 1

        elif self.FOCALPOINTS_TYPE == "absolute":
            for focusidx, focus in enumerate(self.FOCALPOINTS_PROTOTYPE):
                assert len(focus) == 3, "The focus should be a 3D coordinate"
                logger.debug(f"Shape of the focus: {focus.shape}")
                for idx, _frame in enumerate(worker.traj.xyz[:: self.time_window]):
                    if idx >= worker.SLICENUMBER:
                        break
                    worker.FOCALPOINTS[idx, focusidx] = focus
            return 1

        elif self.FOCALPOINTS_TYPE == "json":
            with open(self.FOCALPOINTS_PROTOTYPE) as f:
                focus = json.load(f)
                indices = focus[utils.get_pdbcode(worker.traj.identity)]
            indices = np.array(indices, dtype=int)
            for idx, frame in enumerate(worker.traj.xyz[:: self.time_window]):
                if idx >= worker.SLICENUMBER:
                    break
                focus = np.mean(frame[indices], axis=0)
                worker.FOCALPOINTS[idx, 0] = focus
            return 1

        else:
            raise ValueError(f"Unexpected focus format: {self.FOCALPOINTS_TYPE}")

    @property
    def gpu_busy_time(self):
        """
        Total wall-clock time (seconds) the GPU was busy during the last
        ``run()``: the accumulated dispatch-to-collection window of every CUDA
        kernel (see ``features.Feature.gpu_busy_seconds``). A lower bound on
        GPU utilization; excludes idle time waiting for CPU work.
        """
        return features.Feature.gpu_busy_seconds

    def run(self):
        """
        Run the featurization for each iteration over trajectory, frame-slice, focal-point, and feature.

        The pipeline is overlapped with background threads:

        * **CPU producer** thread(s) run all the CPU preprocessing (trajectory
          loading, focal-point parsing, weight caching and coordinate cropping)
          and deposit the resulting GPU tasks onto a :class:`PrefetchBuffer`.
        * The **main process** consumes the GPU tasks from the buffer and
          launches the CUDA kernels (``Feature.run``).
        * An :class:`AsyncWriter` thread drains the results and writes them to
          HDF5 (``Feature.dump``) asynchronously.

        This mirrors a ``torch.utils.data.DataLoader`` prefetch buffer: the CPU
        preprocessing for task ``N + 1`` overlaps with the GPU compute of task
        ``N``, and the HDF5 writes overlap with the next kernel launch.

        When ``producer_threads > 1``, the CPU preprocessing is additionally
        parallelized across trajectories: each producer owns a private clone of
        the feature set and processes a disjoint subset of the trajectories.
        This helps when the CPU preprocessing is the bottleneck.

        The overlap depends on the CUDA extension releasing the GIL around each
        kernel launch (``py::gil_scoped_release`` in ``src/actions_py.cpp``);
        while it is held no background thread can run and this degenerates to
        the serial schedule.

        In exchange, that release makes the extension re-entrant, which the
        global device context is not. Only this loop may launch kernels: the
        producers confine themselves to ``cache``/``query`` and the writer to
        ``dump``, none of which enter the extension. Adding a second consumer
        thread, or a feature whose ``cache`` calls a kernel, would race on the
        shared device buffers.
        """
        # Reset the GPU busy-time accumulator for this run.
        features.Feature.gpu_busy_seconds = 0.0

        buffer = PrefetchBuffer(capacity=self._prefetch_capacity)
        writer = AsyncWriter(self._dump_result, capacity=self._writer_capacity)

        if self._producer_threads <= 1:
            # Single-producer schedule: the producer owns the featurizer's
            # mutable state and closes the buffer itself; no coordinator needed.
            producers = [
                threading.Thread(
                    target=self._produce_tasks,
                    args=(buffer,),
                    name="nearl-cpu-producer",
                    daemon=True,
                )
            ]
            coordinator = None
        else:
            # Multi-producer schedule: each worker gets a disjoint trajectory
            # chunk and a private feature-set clone.
            traj_indices = np.array_split(
                np.arange(self.TRAJECTORYNUMBER), self._producer_threads
            )
            producers = []
            for wid, indices in enumerate(traj_indices):
                if len(indices) == 0:
                    continue
                worker = _ProducerWorker(self._clone_features())
                producers.append(
                    threading.Thread(
                        target=self._produce_worker,
                        args=(worker, buffer, indices),
                        name=f"nearl-cpu-producer-{wid}",
                        daemon=True,
                    )
                )

            # Coordinator: join all producers, then enqueue the single sentinel
            # (a second one would stop the consumer early).
            coordinator = threading.Thread(
                target=self._coordinate_producers,
                args=(buffer, producers),
                name="nearl-cpu-producer-coordinator",
                daemon=True,
            )

        for producer in producers:
            producer.start()
        if coordinator is not None:
            coordinator.start()

        try:
            while True:
                item = buffer.get()
                if item is _SENTINEL:
                    break
                if item[0] is _ERROR:
                    raise item[1]
                if self._producer_threads <= 1:
                    # Single item: one ``(feature, queried)`` GPU task.
                    feature, queried = item
                    # Launch the GPU kernel on the main process
                    result = feature.run(*queried)
                    # Hand the result to the background writer (async HDF5 dump)
                    writer.submit(feature, result)
                else:
                    # Sample bundle: one GPU task per feature, in feature order;
                    # keeps output datasets row-aligned across producers.
                    for feature, queried in item:
                        result = feature.run(*queried)
                        writer.submit(feature, result)
        finally:
            # Cancel first: a producer parked in put() on a full buffer would
            # hang the join below.
            buffer.cancel()
            for producer in producers:
                producer.join()
            if coordinator is not None:
                coordinator.join()
            writer.close()
            # Close any persistent HDF5 file handles held by the features
            for feat in self.FEATURESPACE:
                feat.close()

        log(f"{self.classname}: All trajectories and tasks are finished. \n")

    def _clone_features(self):
        """
        Return a shallow copy of the feature set.

        ``cache``/``query`` write trajectory-scoped state (``cached_array``,
        ``selected``, ...) onto the feature, so each producer thread needs its
        own instances. A shallow copy shares the immutable configuration
        (dims, spacing, cutoff, ...) while giving each thread its own mutable
        slots, which are always reassigned, never mutated in place. The
        class-level ``_topology_cache`` stays shared, read-mostly.
        """
        clones = []
        for feat in self.FEATURESPACE:
            clone = copy.copy(feat)
            # A shallow copy also copies instance attributes that shadow a
            # method -- the pattern any profiler or monkeypatch uses. Those are
            # closures bound to the *original*, so the clone would silently
            # dispatch back to it and every worker would share one feature's
            # trajectory state. Drop them; the class method is correct here.
            for name, value in vars(feat).items():
                if callable(value) and callable(getattr(type(feat), name, None)):
                    delattr(clone, name)
                    logger.warning(
                        f"{self.classname}: dropped the instance-level override of "
                        f"{type(feat).__name__}.{name} when cloning for a producer "
                        f"thread; it is bound to the original feature and would "
                        f"make the workers share state. Patch the class instead."
                    )
            clones.append(clone)
        return clones

    def _setup_worker_trajectory(self, worker, traj):
        """
        Attach a trajectory to a worker and compute its frame-slices.

        Mirrors the :attr:`traj` setter, but writes the trajectory-scoped state
        onto the worker instead of the featurizer.

        Parameters
        ----------
        worker : :class:`_ProducerWorker`
          The worker to attach the trajectory to.
        traj : pytraj.Trajectory
          The trajectory to process.
        """
        worker.traj = traj
        worker.FRAMENUMBER = traj.n_frames
        worker.SLICENUMBER = worker.FRAMENUMBER // self.time_window
        if worker.SLICENUMBER == 0:
            logger.warning(
                f"{self.classname}: No frame slice is available. The trajectory has {worker.FRAMENUMBER} frames and the time window is {self.time_window}."
            )
        if worker.FRAMENUMBER % self.time_window != 0 and worker.FRAMENUMBER != 1:
            logger.warning(
                f"{self.classname}: the number of frames ({worker.FRAMENUMBER}) is not divisible by the time window ({self.time_window}). The last few frames will be ignored."
            )
        frame_array = np.array(
            [0, *np.cumsum([self.time_window] * worker.SLICENUMBER).tolist()]
        )
        worker.FRAMESLICES = [
            np.s_[frame_array[i] : frame_array[i + 1]]
            for i in range(worker.SLICENUMBER)
        ]

    def _coordinate_producers(self, buffer, producers):
        """
        Join all producer threads, then enqueue the single end-of-stream
        sentinel once every producer has finished.

        Parameters
        ----------
        buffer : :class:`nearl.pipeline.PrefetchBuffer`
          The buffer to close.
        producers : list of threading.Thread
          The producer threads to join.
        """
        for producer in producers:
            producer.join()
        buffer.close()

    def _produce_tasks(self, buffer):
        """
        Background CPU producer (single-producer schedule): run all CPU
        preprocessing and feed GPU tasks to the buffer.

        This method runs on a single background thread. It owns the mutable
        featurizer state (``self.traj``, ``self.FOCALPOINTS``, ``self.FRAMESLICES``
        and the per-feature caches) while the main process only consumes the
        queried data from the buffer and runs the GPU kernels, so there is no
        data race.

        Parameters
        ----------
        buffer : :class:`nearl.pipeline.PrefetchBuffer`
          The buffer onto which ``(feature, queried)`` GPU tasks are
          deposited, one per feature per sample. The buffer is closed (a
          single end-of-stream sentinel is enqueued) once every task has been
          produced.
        """
        try:
            for tid in range(self.TRAJECTORYNUMBER):
                # Attach the trajectory and compute its frame slices
                self.traj = self.TRAJLOADER[tid]
                msg = f"Processing the trajectory {tid + 1} ({self.traj.identity}) with {self.SLICENUMBER} frame slices"
                log(f"{self.classname}: {msg:=^80}")
                st = time.perf_counter()

                if self.FOCALPOINTS_PROTOTYPE is not None:
                    # Re-parse the focal points; shape is (SLICENUMBER, FOCALNUMBER, 3)
                    focus_state = self.parse_focus()
                    if focus_state == 0:
                        log.warning(
                            f"{self.classname}: Skipping the trajectory {self.traj.identity}(index {tid + 1}) because focal points parsing is failed. "
                        )
                        continue
                    if config.verbose() or config.debug():
                        log(
                            f"{self.classname}: Parsing of focal points on trajectory ({tid + 1}/{self.traj.identity}) yield the shape: {self.FOCALPOINTS.shape}. "
                        )

                # Cache the per-atom weights once per trajectory
                for feat in self.FEATURESPACE:
                    if config.verbose():
                        log(
                            f"{self.classname}: Caching the weights of feature {feat.classname} for the trajectory {tid + 1}"
                        )
                    feat.cache(self.traj)

                task_count = 0
                # Enqueue the tasks for each frame slice
                for bid in range(self.SLICENUMBER):
                    self.frame_slice = self.FRAMESLICES[bid]
                    frames = self.traj.xyz[self.FRAMESLICES[bid]]
                    if self.FOCALNUMBER > 0:
                        # After determineing each focus point, run the featurizer for each focus point
                        for pid in range(self.FOCALNUMBER):
                            focal_point = self.FOCALPOINTS[bid, pid]
                            # Crop the trajectory and send the coordinates/trajectory to the featurizer
                            for fidx in range(self.FEATURENUMBER):
                                queried = self.FEATURESPACE[fidx].query(
                                    self.top, frames, focal_point
                                )
                                buffer.put((self.FEATURESPACE[fidx], queried))
                                task_count += 1
                    else:
                        # No focal points: focal-point-independent features
                        for fidx in range(self.FEATURENUMBER):
                            # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
                            queried = self.FEATURESPACE[fidx].query(
                                self.top, frames, [0, 0, 0]
                            )
                            buffer.put((self.FEATURESPACE[fidx], queried))
                            task_count += 1

                log(
                    f"{self.classname}: Trajectory {tid + 1} yields {task_count} frame-slices (tasks) for the featurization. "
                )
                msg = f"Finished the trajectory {tid + 1} / {self.TRAJECTORYNUMBER} with {task_count} tasks in {time.perf_counter() - st:.6f} seconds"
                msg = f"{msg:=^80}"
                if tid < self.SLICENUMBER - 1:
                    msg += "\n"
                log(f"{self.classname}: {msg}")
        except PipelineCancelled:
            # The consumer is already raising its own exception.
            pass
        except Exception as exc:  # pragma: no cover - surfaced on the main thread
            with contextlib.suppress(PipelineCancelled):
                buffer.put((_ERROR, exc))
        finally:
            # No coordinator here, so the producer closes the buffer itself.
            buffer.close()

    def _produce_worker(self, worker, buffer, traj_indices):
        """
        Background CPU producer (multi-producer schedule): run all CPU
        preprocessing for a disjoint subset of trajectories and feed GPU tasks
        to the buffer.

        The GPU tasks carry the *original* feature (``self.FEATURESPACE[fidx]``)
        so the single consumer keeps using the original features for
        ``run``/``dump`` and their persistent HDF5 handles.

        Parameters
        ----------
        worker : :class:`_ProducerWorker`
          The per-thread state and cloned feature set for this producer.
        buffer : :class:`nearl.pipeline.PrefetchBuffer`
          The buffer onto which sample bundles (lists of ``(feature,
          queried)`` GPU tasks, one per feature for a single sample) are
          deposited.
        traj_indices : np.ndarray
          The indices of the trajectories this worker should process.
        """
        try:
            for tid in traj_indices:
                tid = int(tid)
                # Attach the trajectory and compute its frame slices
                self._setup_worker_trajectory(worker, self.TRAJLOADER[tid])
                msg = f"Processing the trajectory {tid + 1} ({worker.traj.identity}) with {worker.SLICENUMBER} frame slices"
                log(f"{self.classname}: {msg:=^80}")
                st = time.perf_counter()

                if self.FOCALPOINTS_PROTOTYPE is not None:
                    # Re-parse the focal points; shape is (SLICENUMBER, FOCALNUMBER, 3)
                    focus_state = self._parse_focus(worker)
                    if focus_state == 0:
                        log.warning(
                            f"{self.classname}: Skipping the trajectory {worker.traj.identity}(index {tid + 1}) because focal points parsing is failed. "
                        )
                        continue
                    if config.verbose() or config.debug():
                        log(
                            f"{self.classname}: Parsing of focal points on trajectory ({tid + 1}/{worker.traj.identity}) yield the shape: {worker.FOCALPOINTS.shape}. "
                        )

                # Cache the per-atom weights once per trajectory
                for feat in worker.features:
                    if config.verbose():
                        log(
                            f"{self.classname}: Caching the weights of feature {feat.classname} for the trajectory {tid + 1}"
                        )
                    feat.cache(worker.traj)

                task_count = 0
                # Enqueue the tasks for each frame slice
                for bid in range(worker.SLICENUMBER):
                    worker.frame_slice = worker.FRAMESLICES[bid]
                    frames = worker.traj.xyz[worker.FRAMESLICES[bid]]
                    if worker.FOCALNUMBER > 0:
                        # After determineing each focus point, run the featurizer for each focus point
                        for pid in range(worker.FOCALNUMBER):
                            focal_point = worker.FOCALPOINTS[bid, pid]
                            # One atomic bundle per sample keeps output rows in
                            # sample order across producers.
                            bundle = []
                            for fidx in range(self.FEATURENUMBER):
                                queried = worker.features[fidx].query(
                                    worker.traj.top, frames, focal_point
                                )
                                bundle.append((self.FEATURESPACE[fidx], queried))
                            buffer.put(bundle)
                            task_count += len(bundle)
                    else:
                        # No focal points: focal-point-independent features
                        bundle = []
                        for fidx in range(self.FEATURENUMBER):
                            # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
                            queried = worker.features[fidx].query(
                                worker.traj.top, frames, [0, 0, 0]
                            )
                            bundle.append((self.FEATURESPACE[fidx], queried))
                        buffer.put(bundle)
                        task_count += len(bundle)

                log(
                    f"{self.classname}: Trajectory {tid + 1} yields {task_count} frame-slices (tasks) for the featurization. "
                )
                msg = f"Finished the trajectory {tid + 1} / {self.TRAJECTORYNUMBER} with {task_count} tasks in {time.perf_counter() - st:.6f} seconds"
                msg = f"{msg:=^80}"
                if tid != int(traj_indices[-1]):
                    msg += "\n"
                log(f"{self.classname}: {msg}")
        except PipelineCancelled:
            # The consumer is already raising its own exception.
            pass
        except Exception as exc:  # pragma: no cover - surfaced on the main thread
            with contextlib.suppress(PipelineCancelled):
                buffer.put((_ERROR, exc))
        # No buffer.close() here: the coordinator enqueues the sentinel after
        # joining every producer.

    def _dump_result(self, feature, result):
        """
        Callback for the :class:`AsyncWriter`: dump a single result to HDF5.

        Parameters
        ----------
        feature : :class:`nearl.features.Feature`
          The feature that produced the result.
        result : np.ndarray
          The result array to dump.
        """
        feature.dump(result)

    def loop_by_residue(self, restype, tag_limit=0):
        """
        TO BE ADDED
        """
        for tid in range(self.TRAJECTORYNUMBER):
            # Setup the trajectory and its related parameters such as slicing of the trajectory
            self.traj = self.TRAJLOADER[tid]
            log(
                f"{self.classname}: Start processing the trajectory {tid + 1} with {self.SLICENUMBER} frames"
            )

            # Cache the weights for each atoms in the trajectory (run once for each trajectory)
            for feat in self.FEATURESPACE:
                feat.cache(self.traj)

            # Calculate the slices to pro cess based on the single / dual residue tag
            tasks = []
            feature_map = []
            for bid in range(self.SLICENUMBER):
                frames = self.traj.xyz[self.FRAMESLICES[bid]]
                if restype == "single":
                    for single_resname in constants.RES + [
                        i for i in constants.RES_PATCH
                    ]:
                        if single_resname in constants.RES_PATCH:
                            label = constants.RES2LAB[
                                constants.RES_PATCH[single_resname]
                            ]
                        else:
                            label = constants.RES2LAB[single_resname]
                        # Find all of the residue block in the sequence and iterate them
                        slices = utils.find_block_single(self.traj, single_resname)
                        for sidx, s_ in enumerate(slices):
                            if tag_limit > 0 and sidx >= tag_limit:
                                break
                            sliced_top = self.traj.top[s_]
                            sliced_coord = frames[:, s_, :]
                            focal_point = np.mean(sliced_coord[0], axis=0)
                            for fidx in range(self.FEATURENUMBER):
                                query_args = (
                                    sliced_top,
                                    sliced_coord.copy(),
                                    focal_point,
                                )
                                tasks.append([self.FEATURESPACE[fidx], query_args])
                                feature_map.append((tid, bid, fidx, label))

                elif restype == "dual":
                    # for label, dual_resname in constants.LAB2RES_DUAL.items():
                    for res1 in constants.RES + [i for i in constants.RES_PATCH]:
                        for res2 in constants.RES + [i for i in constants.RES_PATCH]:
                            tmp_key = ""
                            if res1 in constants.RES_PATCH:
                                tmp_key += constants.RES_PATCH[res1]
                            else:
                                tmp_key += res1
                            if res2 in constants.RES_PATCH:
                                tmp_key += constants.RES_PATCH[res2]
                            else:
                                tmp_key += res2
                            label = constants.RES2LAB_DUAL.get(tmp_key, "Unknown")
                            dual_resname = res1 + res2
                            # Find the residue block in the sequence.
                            slices = utils.find_block_dual(self.traj, dual_resname)
                            for s_ in slices:
                                sliced_top = self.traj.top[s_]
                                sliced_coord = frames[:, s_, :]
                                focal_point = np.mean(sliced_coord[0], axis=0)
                                for fidx in range(self.FEATURENUMBER):
                                    query_args = (
                                        sliced_top,
                                        sliced_coord.copy(),
                                        focal_point,
                                    )
                                    tasks.append([self.FEATURESPACE[fidx], query_args])
                                    feature_map.append((tid, bid, fidx, label))

            log(
                f"{self.classname}: Task set containing {len(tasks)} tasks are created for the trajectory {tid + 1}; "
            )
            results = _run_prepared_tasks(self, tasks, feature_map)

            log(
                f"{self.classname}: Tasks are finished, dumping the results to the feature space..."
            )

            # Dump to file for each feature
            for feat_meta, result in zip(feature_map, results):
                tid, bid, fidx, label = feat_meta
                self.FEATURESPACE[fidx].dump(result)

            if self.FEATURE_PARMS.get("outfile", None) is not None:
                # Dump the label to the file
                labels = np.array([i[-1] for i in feature_map], dtype=int)
                if self.FEATURE_PARMS.get("hdf_compress_level", 0) > 0:
                    utils.append_hdf_data(
                        self.FEATURE_PARMS["outfile"],
                        "label",
                        labels[: int(len(feature_map) / len(self.FEATURESPACE))],
                        dtype=int,
                        maxshape=(None,),
                        chunks=True,
                        compress_level=self.FEATURE_PARMS.get("hdf_compress_level", 0),
                    )
                else:
                    utils.append_hdf_data(
                        self.FEATURE_PARMS["outfile"],
                        "label",
                        labels[: int(len(feature_map) / len(self.FEATURESPACE))],
                        dtype=int,
                        maxshape=(None,),
                        chunks=True,
                    )

            if config.verbose() or config.debug():
                log(
                    f"{self.classname}: Finished the trajectory {tid + 1} with {len(tasks)} tasks"
                )
        log(f"{self.classname}: All trajectories and tasks are finished")
