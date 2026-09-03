Verify installation
-------------------
Run the following command to check the installation of major components from Nearl:

.. code-block:: bash

  python -m nearl.valid_installation

The **static** checks need no GPU, so a build can be verified on a GPU-less login
node before an allocation is spent on it. They confirm that the core modules and
the PyTraj backend import, that the compiled ``nearl.all_actions`` extension
exports every expected symbol, and which GPU architectures are baked into it.

The **runtime** checks need a visible CUDA device and are reported as ``SKIPPED``
rather than ``OK`` when there is none. They report the device, compare its compute
capability against the architectures embedded in the extension, and run a small
voxelization to confirm the kernels actually execute.

.. code-block:: bash

  $ python -m nearl.valid_installation
  Nearl version 0.0.3.dev0

  Static checks (no GPU required)
    1 core modules........................ OK
    2 pytraj backend...................... OK (3.0.0.dev0)
    3 CUDA extension symbols.............. OK (6 symbols)
    4 embedded GPU architectures.......... OK (SASS sm_86; PTX sm_86)

  Runtime checks (CUDA device required)
    5 CUDA device......................... OK (1 device(s), sm_86)
    6 architecture compatibility.......... OK (device sm_86 has native SASS)
    7 GPU voxelization.................... OK (grid sum 92.28)

  Installation validation successful: 7 checks passed.

Check 6 catches an extension compiled for the wrong GPU, which otherwise surfaces
only at run time:

.. code-block:: text

  6 architecture compatibility.......... FAILED (ValueError: the extension is built
    for sm_90 and cannot run on this sm_86 device; rebuild with
    CUDA_COMPUTE_CAPABILITY=sm_86)

.. note::

  The command exits non-zero only when a check *fails*; skipped checks do not.
  Pass ``--require-gpu`` to turn a missing CUDA device into a failure, which is
  what a CI job running on a GPU node wants.
