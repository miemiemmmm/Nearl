NEARL
=====
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/badge/python-3.9-blue.svg

`NEARL <https://github.com/miemiemmmm/Nearl>`_ is a 3D structural data generation framework to featurize bio-molecules specifically focus on their 3D coordinate and protein dynamics 
to make users benefit from the recent development in machine learning algorithms.

Features
--------
- Obtain and embed molecule blocks from 3D molecular structures
- Load arbitrary number of 3D structures into a trajectory container
- Multiple pre-defined 2D or 3D features for the featurization
- Pipeline for featurizing the trajectory container



.. Direct include from the docs/source folder

.. include:: docs/source/install_steps.rst

.. include:: docs/source/install_valid.rst




Quick start
-----------

.. code-block:: python

  import nearl
  parms = nearl.data.MINI_PARMS
  loader = nearl.io.TrajectoryLoader([nearl.data.MINI_TRAJ])
  feat = nearl.features.Featurizer3D(parms)
  feat.register_feature(nearl.features.Mass())


Documentation 
-------------

The detailed documentation is available at `ReadTheDocs <https://nearl.readthedocs.io/en/latest/>`_ or the `documentation <https://miemiewebsites.b-cdn.net/nearl_doc/html/index.html>`_ for more details.




License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

