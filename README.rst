Nearl
=====

|pythonver| |license| |rtdlink|

⚠️ Under construction ⚠️

`Nearl <https://github.com/miemiemmmm/Nearl>`_ is an open-source machine learning framework for mining protein dynamics from molecular dynamics (MD) trajectories. 
In current release, featurization centers on 3D voxel-based representation for 3D-CNN-based frameworks. 


Key Features
------------
- Automated pipeline for the featurization of MD trajectories 
- Flexible definition of features and trajectory container 
- User-friendly API for the customization of featurization workflow
- Multiple true 3D features including 2 dynamic features and 14 static features 
- Pre-built 3D-CNN models for training and development 

.. ###############################################
.. Upon changing the installation guide, sync here

Installation
------------

It is recommended to install the package via the direct installation from GitHub repository. 
The development and tests are performed on Linux(Ubuntu), and the software is not guaranteed to work on other platforms. 
.. Since the software is still under development and not yet uploaded to PyPI, 


.. code-block:: bash

  micromamba create -n nearl_env python=3.9.17 AmberTools=23 openbabel=3.1.1
  micromamba activate nearl_env
  pip install rdkit, pandas, torch, torchvision
  pip install git+https://github.com/miemiemmmm/Nearl.git   # Install the main package
  pip install git+https://github.com/miemiemmmm/SiESTA.git

.. note:: 

  To correctly compile the GPU code, the older device have to adjust the ``CUDA_COMPUTE_CAPABILITY`` accordingly, to match the `CUDA architecture <https://developer.nvidia.com/cuda-gpus>`_. The current default value is ``sm_80``.

To install the package from source with test data, run the following commands:  

.. code-block:: bash

  git clone https://github.com/miemiemmmm/Nearl
  cd Nearl
  pip install . 

Verify installation
-------------------
Runing the following command to check the installation of major components from Nearl:

.. code-block:: bash

  python -m nearl.valid_installation

.. ###################################################################


Quick start
-----------

.. code-block:: python

  import nearl
  import nearl.data
  loader = nearl.TrajectoryLoader([nearl.data.MINI_TRAJ])
  feat = nearl.Featurizer({"dimensions": 32, "lengths":16, "time_window":10})
  feat.register_feature(nearl.features.Mass(outkey='mass', outfile="/tmp/test.h5", sigma=1.5, cutoff=5.0))
  feat.register_focus([":ARG"], "mask")
  feat.register_trajloader(loader)
  feat.main_loop()


Documentation 
-------------

You can find detailed documentation at either of the following locations:

`ReadTheDocs <https://nearl.readthedocs.io/en/latest/>`_

`Documentation <https://miemiewebsites.b-cdn.net/nearl_doc/html/index.html>`_

License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.



.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://opensource.org/licenses/MIT
  :alt: License

.. |pythonver| image:: https://img.shields.io/badge/python-3.9-blue.svg
  :target: https://www.python.org/downloads/release/python-3917/
  :alt: Python 3.9

.. |rtdlink| image:: https://readthedocs.org/projects/nearl/badge/?version=latest
  :target: https://nearl.readthedocs.io/en/latest/
  :alt: Documentation Status
  