Prerequisites
-------------
Before installing Nearl, ensure that the following software is installed on your operating system:

- A C++ compiler: `g++ <https://gcc.gnu.org/>`_
- A CUDA compiler: `nvcc <https://developer.nvidia.com/cuda-downloads>`_
- Optional: a Python environment manager: `conda <https://docs.conda.io/en/latest/miniconda.html>`_, `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ or others
- Optional: if partial charge is not readily available from the topology, `ChargeFW2 <https://github.com/sb-ncbr/ChargeFW2>`_ is required for the additional charge calculation
- Optional: `SiESTA-Surf <https://github.com/miemiemmmm/SiESTA>`_ is required for visualization and surface calculation

.. tip:: 

  For Linux users, `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ is the recommended for Python package management due to its rapid dependency resolution and standalone installation of itself and packages. 
  (Conda's dependency resolution is tooooooooo slow for AmberTools 🥵)

  The code snippet below installs ``micromamba`` in the ``/tmp/micromamba`` directory. Modify the directory path as needed to suit your preferences. To completely uninstall ``micromamba`` and its associated environments, simply delete the specified directory.

  .. code-block:: bash 

    curl -s https://gist.githubusercontent.com/miemiemmmm/40d2e2b49e82d682ef5a7b2aa94a243f/raw/b9a3e3c916cbee42b2cfedcda69d2db916e637c0/install_micromamba.sh | bash -s -- /tmp/micromamba


