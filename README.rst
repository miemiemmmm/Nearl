Nearl
=====

|pythonver| |license| |rtdlink|


`Nearl <https://github.com/miemiemmmm/Nearl>`_ is an open-source machine learning framework designed to mine protein dynamics information from molecular dynamics (MD) trajectories. 
The current release focuses on a 3D voxel-based representation for use in 3D convolutional neural networks (3D-CNNs). 


Key Features
------------
- **Automated pipeline** for featurizing MD trajectories 
- Flexible definition of features and trajectory supplier 
- User-friendly API for customizing featurization workflow
- Support for true 3D dynamic and static features 
- Pre-built 3D-CNN models for machine learning training and development 


Documentation
-------------
The installation guide and tutorials are available on `ReadTheDocs <https://nearl.readthedocs.io/en/latest/installation.html>`_. 


Quick Start
-----------
Below is a simple example demonstrating how to featurize an example trajectory set with Nearl. 
The resulting feature, with dimensions of ``32×32×32`` and a grid resolution of ``0.5``, represents the mass distribution of substructures near all ``ARG`` residues.  
A small example dataset (approximately ``26MB``) will be downloaded to the directory ``/tmp/test``. 
For more details, please refer to the `documentation <https://nearl.readthedocs.io/en/latest/>`_.

.. code-block:: python

  import nearl
  import nearl.featurizer, nearl.features, nearl.io 

  loader = nearl.io.TrajectoryLoader(nearl.get_example_data("/tmp/test")["MINI_TRAJSET"])
  feat = nearl.featurizer.Featurizer({"dimensions": 32, "lengths":16, "time_window":10})
  feat.register_feature(nearl.features.Mass(outkey='mass', outfile="/tmp/test.h5", sigma=1.5, cutoff=5.0))
  feat.register_focus([":ARG"], "mask")
  feat.register_trajloader(loader)
  feat.main_loop()


License
-------

This project is licensed under the `MIT LICENSE <LICENSE>`_.



.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://opensource.org/licenses/MIT
  :alt: License

.. |pythonver| image:: https://img.shields.io/badge/python-3.9-blue.svg
  :target: https://www.python.org/downloads/release/python-3917/
  :alt: Python 3.9

.. |rtdlink| image:: https://readthedocs.org/projects/nearl/badge/?version=latest
  :target: https://nearl.readthedocs.io/en/latest/
  :alt: Documentation Status
  