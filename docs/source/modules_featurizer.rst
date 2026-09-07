
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

The shared ``DeviceContext`` retains device buffers and pinned input staging
buffers on its CUDA stream. Each output array owns a separate pinned allocation;
the GPU copies directly into that array before collection waits for completion.
Keeping an array or one of its views alive also keeps its pinned memory alive.

.. automodule:: nearl.featurizer
   :members:
   :undoc-members:
   :show-inheritance:
