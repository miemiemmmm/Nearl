Installation
============

.. include:: install_prereq.rst

.. include:: install_steps.rst

.. include:: install_valid.rst


.. To make the steps visible to the left sidebar
.. toctree:: 
  :maxdepth: 1

  install_prereq
  install_steps
  install_valid

.. Prerequisites
.. -------------
.. Before installing Nearl, ensure that the following software is installed on your operating system:

.. - A C++ compiler: `g++`_
.. - A CUDA compiler: `nvcc`_
.. - Optional: a Python environment manager: `conda`_, `micromamba`_ or others
.. - Optional: if partial charge is not readily available from the topology, `ChargeFW2`_ is required for the additional charge calculation
.. - Optional: `SiESTA-Surf`_ is required for visualization and surface calculation

.. .. _conda: https://docs.conda.io/en/latest/miniconda.html

.. .. _micromamba: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html

.. .. _g++: https://gcc.gnu.org/

.. .. _nvcc: https://developer.nvidia.com/cuda-downloads

.. .. _ChargeFW2: https://github.com/sb-ncbr/ChargeFW2

.. .. _CUDA architecture: https://developer.nvidia.com/cuda-gpus

.. .. _SiESTA-Surf: https://github.com/miemiemmmm/SiESTA

.. .. tip:: 

..   For Linux users, `micromamba`_ is the recommended for Python package management due to its rapid dependency resolution and standalone installation of itself and packages. 
..   (Conda's dependency resolution is tooooooooo slow for AmberTools 🥵)

..   The code snippet below installs `micromamba` in the ``/tmp/micromamba`` directory. Modify the directory path as needed to suit your preferences. To completely uninstall `micromamba` and its associated environments, simply delete the specified directory.


..   .. The following code will install a micromamba in the ``/tmp/micromamba`` directory. Change the directory to your preferred location, if necessary. To fully remove the micromamba and its environments, simply delete the designated directory.

..   .. code-block:: bash 

..     curl -s https://gist.githubusercontent.com/miemiemmmm/40d2e2b49e82d682ef5a7b2aa94a243f/raw/b9a3e3c916cbee42b2cfedcda69d2db916e637c0/install_micromamba.sh | bash -s -- /tmp/micromamba

.. Installation
.. ------------

.. The development and tests are performed on Linux(Ubuntu), and the software is not tested on other operating systems. 
.. Since the software is still under development and not yet uploaded to PyPI, the installation can be done via direct installation from GitHub repository. 

.. .. code-block:: bash

..   micromamba create -n nearl_env python=3.9.17 AmberTools=23 openbabel=3.1.1
..   micromamba activate nearl_env
..   pip install git+https://github.com/miemiemmmm/Nearl
..   pip install git+https://github.com/miemiemmmm/SiESTA.git

.. .. note:: 

..   To correctly compile the GPU code, the older device have to adjust the ``CUDA_COMPUTE_CAPABILITY`` accordingly, to match the `CUDA architecture`_. The current default value is ``sm_80``.

.. Install from source: 

.. .. code-block:: bash

..   git clone https://github.com/miemiemmmm/Nearl
..   cd Nearl
..   pip install .


