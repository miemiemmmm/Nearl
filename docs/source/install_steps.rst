Installation
------------

The development and tests are performed on Linux(Ubuntu), and the software is not tested on other operating systems. 
Since the software is still under development and not yet uploaded to PyPI, the installation can be done via direct installation from GitHub repository. 

.. code-block:: bash

  micromamba create -c conda-forge -n nearl_env python=3.9.17 AmberTools=23 openbabel=3.1.1
  micromamba activate nearl_env
  pip install git+https://github.com/miemiemmmm/Nearl
  pip install git+https://github.com/miemiemmmm/SiESTA.git

.. note:: 

  To correctly compile the GPU code, the older device have to adjust the ``CUDA_COMPUTE_CAPABILITY`` accordingly, to match the `CUDA architecture <https://developer.nvidia.com/cuda-gpus>`_. The current default value is ``sm_80``.

Install from source: 

.. code-block:: bash

  git clone https://github.com/miemiemmmm/Nearl
  cd Nearl
  pip install .


