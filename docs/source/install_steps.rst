Installation Guide
------------------

Install dependencies
^^^^^^^^^^^^^^^^^^^^
Nearl requires `AmberTools <https://ambermd.org/AmberTools.php>`_ (specifically `PyTraj <https://github.com/Amber-MD/pytraj>`_, the python wrapper of `CPPTRAJ <https://github.com/Amber-MD/cpptraj>`_), `OpenBabel <https://github.com/openbabel/openbabel>`_, and `RDKit <https://github.com/rdkit/rdkit>`_, for its functionality (we tested with ``Python 3.9``). 
`PyTorch <https://pytorch.org/>`_ is the deep learning framework used for pre-built ML models. Note that you may need to manually modify the ``pytorch-cuda`` version to match your ``CUDA`` version. 
The following command creates a new Python environment named ``nearl_env`` with the required dependencies, assuming ``micromamba`` is the package manager in use.
You can replace ``micromamba`` with ``conda`` or ``mamba`` if necessary. 

.. code-block:: bash

  git clone https://github.com/miemiemmmm/Nearl
  cd Nearl
  micromamba env create -f requirements.yml 


Install Nearl
^^^^^^^^^^^^^

The development and tests are performed on Linux(Ubuntu), and the software is not guaranteed to work on other operating systems. 
You can install Nearl using one of the following methods:

**Installation from GitHub**: Install directly from the repository. 

.. code-block:: bash

  pip install git+https://github.com/miemiemmmm/Nearl

**Installation from Source**: Clone the repository and install Nearl from the source. 

.. code-block:: bash

  git clone https://github.com/miemiemmmm/Nearl
  cd Nearl
  pip install . 


.. note:: 

  To correctly compile GPU code, ensure that the ``CUDA_COMPUTE_CAPABILITY`` is set appropriately for your GPU. 
  For older devices, adjust the ``CUDA_COMPUTE_CAPABILITY`` to match the `CUDA architecture <https://developer.nvidia.com/cuda-gpus>`_. The current default value is ``sm_80``.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^
Optional dependencies are available and can be obtained from their respective repositories: 

- ``ChargeFW2``: https://github.com/sb-ncbr/ChargeFW2
- ``SiESTA-Surf``: https://github.com/miemiemmmm/SiESTA
- ``Open3D``: https://github.com/isl-org/Open3D

.. The software is not yet uploaded to PyPI
.. pip install git+https://github.com/miemiemmmm/SiESTA.git
.. micromamba create -c conda-forge -n nearl_env python=3.9.17 AmberTools=23 openbabel=3.1.1
.. micromamba activate nearl_env