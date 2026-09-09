Commands
--------

CUDA results and lifetime
~~~~~~~~~~~~~~~~~~~~~~~~~

Voxelization, frame observation, density flow, marching observers, aggregation,
and summation initialize the shared context on first use and stage inputs in
reusable pinned memory. Public calls still return completed arrays or scalars.
Results are copied into ordinary NumPy arrays during collection; the context reuses one pinned
output buffer for asynchronous device-to-host transfers.

The extension also provides internal ``_dispatch_<action>`` entry points for all
six actions. These return a handle whose ``result()`` waits for completion. Only
one handle may be in flight on the context: collect it before another dispatch
or ``finalize_context()``. Dropping an unfinished handle waits before releasing
its memory. Summation copies its partial sums asynchronously and combines them
on the CPU during collection.

.. automodule:: nearl.commands
   :members:
   :undoc-members:
   :show-inheritance:
