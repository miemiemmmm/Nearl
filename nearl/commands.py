import os

import numpy as np

from . import log, utils

try:
    from . import all_actions
except ImportError:
    log.warning(
        "Could not import all_actions submodule. Please check if the package is compiled correctly."
    )

    class _MissingExtension:
        """Stands in for the unbuilt extension so use-sites fail with a clear reason."""

        def __getattr__(self, name):
            raise ImportError(
                f"nearl.commands.{name} needs the nearl.all_actions CUDA extension, "
                "which is not built in this environment. Build it by running "
                "`make all_actions` in src/ (requires nvcc), or reinstall Nearl "
                "where the CUDA toolkit is available."
            )

    all_actions = _MissingExtension()

try:
    # Re-exported so callers can isinstance()-check what the *_dlpack commands
    # return without reaching into the extension module. A stale extension
    # raises AttributeError here, and must not take the whole module down.
    DeviceArray = all_actions.DeviceArray
except (ImportError, AttributeError):
    DeviceArray = None

__all__ = [
    # Single frame methods
    "frame_observation",
    "frame_voxelize",
    # Trajectory/frame-slice methods
    "density_flow",
    "marching_observer",
    # Device context lifecycle
    "init_context",
    "finalize_context",
    "context_valid",
    # DLPack GPU tensor methods
    "DeviceArray",
    "frame_voxelize_dlpack",
    "frame_observation_dlpack",
    "marching_observer_dlpack",
    "density_flow_dlpack",
]


def frame_voxelize(coords, weights, grid_dims, spacing, cutoff, sigma):
    """
    Voxelize a set of coordinates and weights (Single frame version of the density flow method)

    Parameters
    ----------
    coords : np.ndarray
      The coordinates of the points to be voxellized
    weights : np.ndarray
      The weights of the points to be voxellized
    grid_dims : tuple
      The dimensions of the grid
    spacing : float
      The spacing of the grid
    cutoff : float
      The cutoff distance
    sigma : float
      The sigma value for the Gaussian kernel

    Returns
    -------
    np.ndarray
      The voxellized grid sized grid_dims

    Examples
    --------
    >>> import numpy as np
    >>> from nearl import commands
    >>> coords = np.random.normal(size=(100, 3), loc=5, scale=2)
    >>> weights = np.full(100, 1)
    >>> grid_dims = np.array([32, 32, 32])
    >>> commands.frame_voxelize(coords, weights, grid_dims, 0.5, 5, 2)
    """
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    grid_dims = np.array(grid_dims, dtype=int)
    spacing = float(spacing)
    cutoff = float(cutoff)
    sigma = float(sigma)
    # NOTE: no auto translation in the C++ part
    ret_arr = all_actions.frame_voxelize(
        coords, weights, grid_dims, spacing, cutoff, sigma, 0
    )
    return ret_arr.reshape(grid_dims)


def frame_observation(coords, weights, grid_dims, spacing, cutoff, type_obs):
    """
    Perform marching observer on a single frame.

    Parameters
    ----------
    coords : np.ndarray
      The coordinates of the points to be voxellized
    weights : np.ndarray
      The weights of the points to be voxellized
    grid_dims : tuple
      The dimensions of the grid
    spacing : float
      The spacing of the grid
    cutoff : float
      The cutoff distance
    type_obs : int
      The type of observer

    Returns
    -------
    np.ndarray
      The voxellized grid sized grid_dims

    """
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    grid_dims = np.array(grid_dims, dtype=int)
    spacing = float(spacing)
    cutoff = float(cutoff)
    type_obs = int(type_obs)
    ret_arr = all_actions.frame_observation(
        coords, weights, grid_dims, spacing, cutoff, type_obs
    )
    return ret_arr.reshape(grid_dims)


def marching_observer(coords, weights, grid_dims, spacing, cutoff, type_obs, type_agg):
    """
    Marching observers algorithm to create a grid from a slice of frames. The number of atoms in each frame should be the same.

    Parameters
    ----------
    coords : np.ndarray
      The coordinates of the points to calculate the marching observer
    weights : np.ndarray
      The weights of the corresponding points
    grid_dims : tuple
      The dimensions of the grid
    spacing : float
      The spacing of the grid
    cutoff : float
      The cutoff distance
    type_obs : int
      The type of observer
    type_agg : int
      The type of aggregation function

    Returns
    -------
    ret_arr : np.ndarray
      The voxellized grid sized grid_dims

    """
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    grid_dims = np.asarray(grid_dims, dtype=int)
    ret_arr = all_actions.marching_observer(
        coords, weights, grid_dims, spacing, cutoff, type_obs, type_agg
    )
    return ret_arr.reshape(grid_dims)


def density_flow(traj, weights, grid_dims, spacing, cutoff, sigma, type_agg):
    """
    Voxelize a trajectory using the density flow method

    Parameters
    ----------
    traj : np.ndarray
      The trajectory to be voxellized
    weights : np.ndarray
      The weights of the trajectory
    grid_dims : tuple
      The dimensions of the grid
    spacing : float
      The spacing of the grid
    cutoff : float
      The cutoff distance
    sigma : float
      The sigma value for the Gaussian kernel
    type_agg : int
      The type of aggregation function

    Returns
    -------
    retgrid : np.ndarray
      The voxellized grid sized grid_dims

    Examples
    --------
    >>> import numpy as np
    >>> from nearl import commands
    """
    # Check the data type of the inputs; All arrays should be of type np.float32
    if traj.dtype != np.float32:
        traj = traj.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    grid_dims = np.array(grid_dims, dtype=int)
    spacing = float(spacing)
    cutoff = float(cutoff)
    sigma = float(sigma)
    type_agg = int(type_agg)

    ret_arr = all_actions.density_flow(
        traj, weights, grid_dims, spacing, cutoff, sigma, type_agg
    )
    if np.isnan(ret_arr).any():
        log.warning(f"Found nan in the return: {np.count_nonzero(np.isnan(ret_arr))}")
    return ret_arr.reshape(grid_dims)


def viewpoint_histogram_xyzr(
    xyzr_arr, viewpoint, bin_nr, write_ply=False, return_mesh=False
):
    """
    Generate the viewpoint histogram from a set of coordinates and radii (XYZR).

    Wiewpoint is the position of the observer.
    """
    import open3d as o3d
    import siesta

    thearray = np.asarray(xyzr_arr, dtype=np.float32)
    vertices, faces = siesta.xyzr_to_surf(thearray, grid_size=0.2)
    c_vertices = np.mean(vertices, axis=0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    if write_ply:
        filename = os.path.join("/tmp/", f"segment_{utils.get_timestamp()}.ply")
        log(f"Writing the surface to {filename}")
        o3d.io.write_triangle_mesh(filename, mesh, write_ascii=True)

    v_view = viewpoint - c_vertices
    v_view = v_view / np.linalg.norm(v_view)
    # Get the normal of each vertex
    normals = np.array(mesh.vertex_normals)
    # Get the cosine angle and split to bins
    cos_angle = np.dot(normals, v_view)
    bins = np.linspace(-1, 1, bin_nr + 1)
    hist, _ = np.histogram(cos_angle, bins, density=True)
    if return_mesh:
        return hist / np.sum(hist), mesh
    else:
        return hist / np.sum(hist)


def init_context():
    """
    Create the persistent GPU context used by all CUDA commands.

    CUDA commands also initialize this context automatically on first use.
    The extension reuses a single CUDA stream, cached device buffers, and
    pinned input buffers, eliminating repeated allocations at steady capacity.
    Call :func:`finalize_context` at shutdown to release GPU memory.

    Notes
    -----
    One context per process, bound to whichever device is current. Selecting a
    device, or holding a context per device, is future work and not part of this
    commit: for several GPUs, give each worker its own process and set
    ``CUDA_VISIBLE_DEVICES``.
    """
    all_actions.init_context()


def finalize_context():
    """Release cached buffers after collecting any pending result; output arrays remain valid."""
    all_actions.finalize_context()


def context_valid():
    """Return ``True`` if the persistent GPU context is active."""
    return all_actions.context_valid()


def _grid_dims(grid_dims):
    dims = np.asarray(grid_dims, dtype=np.int32)
    if dims.shape != (3,):
        raise ValueError(f"grid_dims must hold 3 entries, got {tuple(dims.shape)}")
    return dims


def frame_voxelize_dlpack(coords, weights, grid_dims, spacing, cutoff, sigma, out=None):
    """
    Voxelize a single frame straight into CUDA memory, skipping the
    Device-to-Host copy that :func:`frame_voxelize` performs.

    Parameters
    ----------
    out : object, optional
      Any DLPack-capable CUDA buffer (``torch.Tensor``, ``cupy.ndarray``, ...)
      to write into. It must be float32, C-contiguous, on this process's CUDA
      device, and hold ``prod(grid_dims)`` elements. When omitted, a new grid is
      allocated here.

    Returns
    -------
    object
      ``out`` when one was given, otherwise a :class:`DeviceArray` exporting the
      grid through DLPack.

    Examples
    --------
    >>> import torch
    >>> from nearl import commands
    >>> grid = commands.frame_voxelize_dlpack(coords, weights, (32, 32, 32), 0.5, 5, 2)
    >>> tensor = torch.from_dlpack(grid)          # zero-copy
    """
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    result = all_actions.frame_voxelize_dlpack(
        coords,
        weights,
        _grid_dims(grid_dims),
        float(spacing),
        float(cutoff),
        float(sigma),
        out=out,
    )
    return out if out is not None else result


def frame_observation_dlpack(
    coords, weights, grid_dims, spacing, cutoff, type_obs, out=None
):
    """
    Compute a single-frame marching-observer observable straight into CUDA
    memory. See :func:`frame_voxelize_dlpack` for ``out`` and the return value.
    """
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    result = all_actions.frame_observation_dlpack(
        coords,
        weights,
        _grid_dims(grid_dims),
        float(spacing),
        float(cutoff),
        int(type_obs),
        out=out,
    )
    return out if out is not None else result


def marching_observer_dlpack(
    coords, weights, grid_dims, spacing, cutoff, type_obs, type_agg, out=None
):
    """
    Run marching observers on a frame slice straight into CUDA memory. See
    :func:`frame_voxelize_dlpack` for ``out`` and the return value.
    """
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    result = all_actions.marching_observer_dlpack(
        coords,
        weights,
        _grid_dims(grid_dims),
        float(spacing),
        float(cutoff),
        int(type_obs),
        int(type_agg),
        out=out,
    )
    return out if out is not None else result


def density_flow_dlpack(
    traj, weights, grid_dims, spacing, cutoff, sigma, type_agg, out=None
):
    """
    Voxelize a trajectory straight into CUDA memory. See
    :func:`frame_voxelize_dlpack` for ``out`` and the return value.

    Examples
    --------
    Write a batch without allocating anything per sample:

    >>> batch = torch.empty((n, 32, 32, 32), dtype=torch.float32, device="cuda")
    >>> for i, traj in enumerate(trajectories):
    ...     commands.density_flow_dlpack(
    ...         traj, weights, (32, 32, 32), 0.5, 5, 2, 1, out=batch[i]
    ...     )
    """
    if traj.dtype != np.float32:
        traj = traj.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    result = all_actions.density_flow_dlpack(
        traj,
        weights,
        _grid_dims(grid_dims),
        float(spacing),
        float(cutoff),
        float(sigma),
        int(type_agg),
        out=out,
    )
    return out if out is not None else result


def discretize_coord(coords, weights, grid_dims, spacing):
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    grid_orig = np.zeros(grid_dims, dtype=np.float32)
    # mid = np.array(grid_dims) / 2
    # coords -= mid   # align the center of coord to the center of the grid
    coord_transform = np.floor(coords / spacing).astype(np.int32)
    for i in range(len(coords)):
        if np.any(coord_transform[i] < 0) or np.any(coord_transform[i] >= grid_dims):
            continue
        else:
            grid_orig[
                coord_transform[i][0], coord_transform[i][1], coord_transform[i][2]
            ] += weights[i]
    return grid_orig
