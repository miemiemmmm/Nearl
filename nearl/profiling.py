"""
NVTX annotations for the featurization pipeline.

The ranges emitted here show up in ``nsys`` next to the CUDA API calls and
kernels, which makes it possible to tell apart the host-side stages (trajectory
loading, ``cache``, ``query``, HDF5 ``dump``) from the device work issued by
:mod:`nearl.commands`.

Nothing here is required at runtime: if the ``nvtx`` package is missing, or if
``NVTX_DISABLE`` is set in the environment, every helper degrades to a no-op
that ``nvtx`` itself implements without any per-call overhead (the decorator
returns the undecorated function).

Examples
--------
.. code-block:: bash

  nsys profile -t cuda,nvtx -o nearl_run python your_script.py
  nsys stats --report nvtx_sum nearl_run.nsys-rep

.. code-block:: python

  from nearl.profiling import annotate, nvtx_range

  @annotate("my_stage", category="query")
  def my_stage(...):
      ...

  with nvtx_range(f"trajectory:{identity}", category="io"):
      ...

"""

import contextlib

__all__ = [
    "DOMAIN",
    "NVTX_AVAILABLE",
    "annotate",
    "mark",
    "nvtx_range",
    "pop_range",
    "push_range",
]

#: All Nearl ranges are emitted into this NVTX domain so that they can be
#: filtered away from the ranges of other libraries in the report.
DOMAIN = "nearl"

#: Colour per pipeline stage, so a timeline is readable without reading labels.
CATEGORY_COLORS = {
    "io": "orange",  # trajectory loading and HDF5 writing
    "cache": "red",  # per-trajectory atom property caching (RDKit/OpenBabel)
    "focus": "yellow",  # focal point parsing
    "query": "green",  # coordinate cropping/padding on the host
    "gpu": "blue",  # dispatch of and waiting for the CUDA actions
    "dump": "purple",  # HDF5 append
}

try:
    import nvtx as _nvtx

    NVTX_AVAILABLE = _nvtx.enabled()
except ImportError:  # pragma: no cover - depends on the environment
    _nvtx = None
    NVTX_AVAILABLE = False


if NVTX_AVAILABLE:

    def _color(color, category):
        if color is not None:
            return color
        return CATEGORY_COLORS.get(category, "blue")

    def annotate(message=None, color=None, category=None, domain=DOMAIN):
        """
        Annotate a range, as a decorator or as a context manager.

        Parameters
        ----------
        message : str, optional
          Label of the range. Defaults to the function name when used as a
          decorator. Keep it a constant string: ``nvtx`` interns every distinct
          message, so building it per call both costs time and grows the string
          table.
        color : str or int, optional
          Overrides the colour derived from ``category``.
        category : str, optional
          One of the keys of :data:`CATEGORY_COLORS`; used to colour and group
          the range.
        domain : str
          NVTX domain, :data:`DOMAIN` by default.
        """
        return _nvtx.annotate(
            message=message,
            color=_color(color, category),
            domain=domain,
            category=category,
        )

    def push_range(message=None, color=None, category=None, domain=DOMAIN):
        """Open a range that a matching :func:`pop_range` closes."""
        _nvtx.push_range(
            message=message,
            color=_color(color, category),
            domain=domain,
            category=category,
        )

    def pop_range(domain=DOMAIN):
        """Close the innermost range opened by :func:`push_range`."""
        _nvtx.pop_range(domain=domain)

    def mark(message=None, color=None, category=None, domain=DOMAIN):
        """Emit an instantaneous marker instead of a range."""
        _nvtx.mark(
            message=message,
            color=_color(color, category),
            domain=domain,
            category=category,
        )

    @contextlib.contextmanager
    def nvtx_range(message=None, color=None, category=None, domain=DOMAIN):
        """
        Context manager for ranges whose label is only known at runtime.

        Prefer :func:`annotate` for static labels; this one exists for messages
        built from data (a trajectory identity, a feature name).
        """
        push_range(message, color=color, category=category, domain=domain)
        try:
            yield
        finally:
            pop_range(domain=domain)

else:  # pragma: no cover - trivial no-op fallbacks

    class annotate(contextlib.nullcontext):  # Mirrors the nvtx class name
        """No-op stand-in used when ``nvtx`` is unavailable or disabled."""

        def __init__(self, *args, **kwargs):
            super().__init__()

        def __call__(self, func):
            return func

    # Plain functions rather than decorated ones: the overhead of the disabled
    # path should stay at one empty call. See NVIDIA/NVTX#24.
    def push_range(message=None, color=None, category=None, domain=DOMAIN):
        pass

    def pop_range(domain=DOMAIN):
        pass

    def mark(message=None, color=None, category=None, domain=DOMAIN):
        pass

    def nvtx_range(message=None, color=None, category=None, domain=DOMAIN):
        return contextlib.nullcontext()
