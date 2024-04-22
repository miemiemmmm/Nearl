
Prerequisites
-------------
Before you start, ensure that you have the following installed:

<p>A CUDA compiler: <a href="https://developer.nvidia.com/cuda-downloads">nvcc</a></p>

<p>Optional <a hef="https://github.com/sb-ncbr/ChargeFW2">ChargeFW2</a> if additional charge calculation is needed 
<p>


- CUDA




Installation
------------


Initialize a featurizer
-----------------------
>>> import nearl
>>> FEATURIZER_PARMS = {
  "dimensions": 32, 
  "lengths": 16, 
  "time_window": 10, 
  # For default setting inference of registered features
  "sigma": 2.0, 
  "cutoff": 2.0, 
  "outfile": outputfile, 
  # Other options
  "progressbar": False, 
}
>>> featurizer = nearl.features.Featurizer(FEATURIZER_PARMS)


Load trajectory container
-------------------------
>>> loader = nearl.io.TrajectoryLoader(trajlists)


Register featurizers
--------------------
Register a single feature
>>> feat.register_feature(nearl.features.Mass())
or via a dictionary or a list of features
>>> from collections import OrderedDict
>>> features = OrderedDict()
>>> features["mass"] = nearl.features.Mass()
>>> features["charge"] = nearl.features.Charge()
>>> feat.register_features(features)



Start featurization
-------------------


>>> feat.register_focus([":15&@CA"], "mask")
>>> feat.main_loop()


