import contextlib
import json
import logging
import threading
import time

import numpy as np

from . import config, constants, log, utils
from .pipeline import (
    _ERROR,
    _SENTINEL,
    AsyncWriter,
    PipelineCancelled,
    PrefetchBuffer,
)
from .profiling import annotate, nvtx_range, pop_range, push_range

__all__ = [
    "Featurizer",
]

logger = logging.getLogger(__name__)


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


@annotate("Featurizer.run_tasks", category="gpu")
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
                push_range("collect", category="gpu")
                results.append(collect())
                pop_range()
            push_range("query", category="query")
            queried = feature.query(*query_args)
            pop_range()
            if pending is not None:
                collect, pending = pending, None
                push_range("collect", category="gpu")
                results.append(collect())
                pop_range()
            # Preserve custom run() implementations rather than bypassing them.
            if async_feature:
                push_range("dispatch", category="gpu")
                pending = feature._dispatch(*queried)
                pop_range()
            else:
                push_range("run", category="gpu")
                results.append(feature.run(*queried))
                pop_range()
        if pending is not None:
            collect, pending = pending, None
            push_range("collect", category="gpu")
            results.append(collect())
            pop_range()
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
                f"{self.classname}: No frame slice is available. The trajectory have {self.FRAMENUMBER} frames and the time window is {self.time_window}."
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

    @annotate("Featurizer.parse_focus", category="focus")
    def parse_focus(self):
        """
        After registering the active trajectory, parse the focal points for each frame-slice in the ``run`` method.
        The resulting shape will be a 3D array with the shape ``(slice_number, focus_number, 3)``

        """
        # Parse the focus points to the correct format
        self.FOCALPOINTS = np.full(
            (self.SLICENUMBER, len(self.FOCALPOINTS_PROTOTYPE), 3),
            99999,
            dtype=np.float32,
        )
        logger.debug(f"Shape of the focal points prototype: {self.FOCALPOINTS.shape}")
        self.FOCALNUMBER = len(self.FOCALPOINTS_PROTOTYPE)
        if self.FOCALPOINTS_TYPE == "mask":
            # Get the center of geometry for the frames with self.interval
            for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE):
                selection = self.traj.top.select(mask)
                if len(selection) == 0:
                    log.warning(
                        f"{self.classname}: The trajectory {self.traj.identity} does not have any atoms in the selection {mask}"
                    )
                    return False
                for fidx in range(self.SLICENUMBER):
                    frame = self.traj.xyz[fidx * self.time_window]
                    self.FOCALPOINTS[fidx, midx] = np.mean(frame[selection], axis=0)
            return 1
        elif self.FOCALPOINTS_TYPE == "index":
            for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE):
                for idx, frame in enumerate(self.traj.xyz[:: self.time_window]):
                    if idx >= self.SLICENUMBER:
                        break
                    self.FOCALPOINTS[idx, midx] = np.mean(frame[mask], axis=0)
            return 1

        elif self.FOCALPOINTS_TYPE == "absolute":
            for focusidx, focus in enumerate(self.FOCALPOINTS_PROTOTYPE):
                assert len(focus) == 3, "The focus should be a 3D coordinate"
                logger.debug(f"Shape of the focus: {focus.shape}")
                for idx, _frame in enumerate(self.traj.xyz[:: self.time_window]):
                    if idx >= self.SLICENUMBER:
                        break
                    self.FOCALPOINTS[idx, focusidx] = focus
            return 1

        elif self.FOCALPOINTS_TYPE == "json":
            with open(self.FOCALPOINTS_PROTOTYPE) as f:
                focus = json.load(f)
                indices = focus[utils.get_pdbcode(self.traj.identity)]
            indices = np.array(indices, dtype=int)
            for idx, frame in enumerate(self.traj.xyz[:: self.time_window]):
                if idx >= self.SLICENUMBER:
                    break
                focus = np.mean(frame[indices], axis=0)
                self.FOCALPOINTS[0, idx] = focus
            return 1

        else:
            raise ValueError(f"Unexpected focus format: {self.FOCALPOINTS_TYPE}")

    @annotate("Featurizer.run", category="io")
    def run(self):
        """
        Run the featurization for each iteration over trajectory, frame-slice, focal-point, and feature.

        The pipeline is overlapped with background threads:

        * A **CPU producer** thread runs all the CPU preprocessing (trajectory
          loading, focal-point parsing, weight caching and coordinate cropping)
          and deposits the resulting GPU tasks onto a :class:`PrefetchBuffer`.
        * The **main process** consumes the GPU tasks from the buffer and
          launches the CUDA kernels (``Feature.run``).
        * An :class:`AsyncWriter` thread drains the results and writes them to
          HDF5 (``Feature.dump``) asynchronously.

        This mirrors a ``torch.utils.data.DataLoader`` prefetch buffer: the CPU
        preprocessing for task ``N + 1`` overlaps with the GPU compute of task
        ``N``, and the HDF5 writes overlap with the next kernel launch.

        The overlap depends on the CUDA extension releasing the GIL around each
        kernel launch (``py::gil_scoped_release`` in ``src/actions_py.cpp``);
        while it is held no background thread can run and this degenerates to
        the serial schedule.

        In exchange, that release makes the extension re-entrant, which the
        global device context is not. Only this loop may launch kernels: the
        producer confines itself to ``cache``/``query`` and the writer to
        ``dump``, none of which enter the extension. Adding a second consumer
        thread, or a feature whose ``cache`` calls a kernel, would race on the
        shared device buffers.
        """
        buffer = PrefetchBuffer(capacity=self._prefetch_capacity)
        writer = AsyncWriter(self._dump_result, capacity=self._writer_capacity)

        producer = threading.Thread(
            target=self._produce_tasks,
            args=(buffer,),
            name="nearl-cpu-producer",
            daemon=True,
        )
        producer.start()

        try:
            while True:
                # A long "wait" range means the CPU producer is the bottleneck
                with nvtx_range("Featurizer.wait", category="io"):
                    item = buffer.get()
                if item is _SENTINEL:
                    break
                if item[0] is _ERROR:
                    raise item[1]
                feature, queried = item
                # Launch the GPU kernel on the main process
                with nvtx_range(feature.classname, category="gpu"):
                    result = feature.run(*queried)
                # Hand the result to the background writer (async HDF5 dump)
                with nvtx_range("Featurizer.submit", category="dump"):
                    writer.submit(feature, result)
        finally:
            # Anything raised above (a kernel error, a missing extension) leaves
            # the producer parked in buffer.put() on a full buffer. Cancel it
            # first: joining a stranded producer would hang the process instead
            # of surfacing the exception.
            buffer.cancel()
            producer.join()
            writer.close()
            # Close any persistent HDF5 file handles held by the features
            for feat in self.FEATURESPACE:
                feat.close()

        log(f"{self.classname}: All trajectories and tasks are finished. \n")

    def _produce_tasks(self, buffer):
        """
        Background CPU producer: run all CPU preprocessing and feed GPU tasks to
        the buffer.

        This method runs on a single background thread. It owns the mutable
        featurizer state (``self.traj``, ``self.FOCALPOINTS``, ``self.FRAMESLICES``
        and the per-feature caches) while the main process only consumes the
        queried data from the buffer and runs the GPU kernels, so there is no
        data race.

        Parameters
        ----------
        buffer : :class:`nearl.pipeline.PrefetchBuffer`
          The buffer onto which ``(feature, queried)`` GPU tasks are deposited.
        """
        try:
            for tid in range(self.TRAJECTORYNUMBER):
                # Setup the trajectory and its related parameters such as slicing of the trajectory
                with nvtx_range("Featurizer.load_trajectory", category="io"):
                    self.traj = self.TRAJLOADER[tid]
                msg = f"Processing the trajectory {tid + 1} ({self.traj.identity}) with {self.SLICENUMBER} frame slices"
                log(f"{self.classname}: {msg:=^80}")
                st = time.perf_counter()
                with nvtx_range("Featurizer.trajectory", category="io"):
                    if self.FOCALPOINTS_PROTOTYPE is not None:
                        # NOTE: Re-parse the focal points for each trajectory
                        # Expected output shape is (self.SLICENUMBER, self.FOCALNUMBER, 3) array
                        focus_state = self.parse_focus()
                        if focus_state == 0:
                            log.warning(
                                f"{self.classname}: Skipping the trajectory {self.traj.identity}(index {tid + 1}) because focal points parsing is failed. "
                            )
                            continue
                        if config.verbose() or config.debug():
                            log(
                                f"{self.classname}: Parsing of focal points on trajectory ({tid + 1}/{self.traj.identity}) yeield the shape: {self.FOCALPOINTS.shape}. "
                            )

                    # Cache the weights for each atoms in the trajectory (run once for each trajectory)
                    with nvtx_range("Featurizer.cache", category="cache"):
                        for feat in self.FEATURESPACE:
                            if config.verbose():
                                log(
                                    f"{self.classname}: Caching the weights of feature {feat.classname} for the trajectory {tid + 1}"
                                )
                            with nvtx_range(feat.classname, category="cache"):
                                feat.cache(self.traj)

                    task_count = 0
                    # Pool the actions for each trajectory
                    for bid in range(self.SLICENUMBER):
                        self.frame_slice = self.FRAMESLICES[bid]
                        frames = self.traj.xyz[self.FRAMESLICES[bid]]
                        if self.FOCALNUMBER > 0:
                            # After determineing each focus point, run the featurizer for each focus point
                            for pid in range(self.FOCALNUMBER):
                                focal_point = self.FOCALPOINTS[bid, pid]
                                # Crop the trajectory and send the coordinates/trajectory to the featurizer
                                for fidx in range(self.FEATURENUMBER):
                                    # NOTE: Isolate the effect on the calculation of the next feature
                                    with nvtx_range("query", category="query"):
                                        queried = self.FEATURESPACE[fidx].query(
                                            self.top, frames, focal_point
                                        )
                                    buffer.put((self.FEATURESPACE[fidx], queried))
                                    task_count += 1
                        else:
                            # Without registeration of focal points: focal-point independent features such as label-generation
                            for fidx in range(self.FEATURENUMBER):
                                # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
                                with nvtx_range("query", category="query"):
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
            # The consumer stopped early and cancelled us; it is already
            # raising its own exception, so there is nothing to report.
            pass
        except Exception as exc:  # pragma: no cover - surfaced on the main thread
            with contextlib.suppress(PipelineCancelled):
                buffer.put((_ERROR, exc))
        finally:
            buffer.close()

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
