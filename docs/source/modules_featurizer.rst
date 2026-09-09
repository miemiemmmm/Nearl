
.. _module_featurizer: 

Featurizer
----------

CUDA input preparation
~~~~~~~~~~~~~~~~~~~~~~

Both ``run()`` and ``loop_by_residue()`` prepare the next task's input while the
previous CUDA task runs. They collect the previous result before dispatching the
next task, so the shared buffers cannot be overwritten while in use. This applies
to ordinary voxel features, density flow, and marching observers. Features with
custom ``run()`` implementations keep their existing synchronous behavior.

The shared ``DeviceContext`` retains device buffers, pinned input staging buffers,
and one pinned output buffer on its CUDA stream. Collection copies the completed
output into ordinary NumPy memory, so result arrays do not retain CUDA-pinned memory.

.. automodule:: nearl.featurizer
   :members:
   :undoc-members:
   :show-inheritance:
