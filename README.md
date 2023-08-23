# NEARL
[NEARL](https://github.com/miemiemmmm/BetaPose)(Nanoscale Environment Assessment and Resonance Landscapes) is a 3D structural 
data generation framework to featurize bio-molecules specifically focus on their 3D coordinate and protein dynamics 
to make users benefit from the recent development in machine learning algorithms. <br>

- Obtain and embed molecule blocks from 3D molecular structures
- Load arbitrary number of 3D structures into a trajectory container
- Multiple pre-defined 2D or 3D features for the featurization
- Pipeline for featurizing the trajectory container


## NEARL
* [Installation](#Installation)
* [Get started](#Get-started)
* [Trajectory loader](#Trajectory-loader)
* [Featurizer](#Featurizer)
* [Feature deposition](#Feature-deposition)
* [Model training](#Model-training)
* [License](#License)


# Installation

--------
NEARL supports only the Linux platform for the time being. It is recommended to install via PyPI: <br>
```pip install nearl``` <br>
By defaults, it uses [OpenMP](https://www.openmp.org/) when doing feature density interpolation, there are some 
key components accelerated by [OpenACC](https://www.openacc.org/). To install the GPU version, [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)
is required. <br>
Use the following command to install the GPU version: <br>
```pip install nearl-gpu``` <br>

To test the installation: <br>
```python -c "import nearl; nearl.test.vectorize(); nearl.test.featurize_pdbbind()"``` <br>


# Get started

--------
```
import nearl as nl
loader = nl.trajloader("test.nc", "test.pdb")
featurizer = nl.featurizer()
featurizer.register_feature(nl.features.Mass())
......
......
```


# Trajectory loader

--------
### Load structures into trajectory container
NEARL regards every 3D structure as trajectories rather than separate molecules. [pytraj](https://amber-md.github.io/pytraj/latest/index.html) is the backend for trajectory processing. <br>

Trajectory loader currently supports the following formats: **NetCDF**, **PDB**, **XTC**, **TRR**, **DCD**, **h5/hdf**. <br>
The trajectory loader normally reads trajectory/topology pairs. 
```
from nearl import trajloader
traj_list = [traj1, traj2, traj3, ..., trajn]
top_list = [top1, top2, top3, ..., topn]
traj_loader = trajloader.TrajectoryLoader(traj_list, top_list)
```
### Static structures
Single snapshot from MD or static structure (like PDB) are dealt as a trajectory with only one frame. If this is the case, 
you could only needs to load the structure as  

```
from nearl import trajloader
traj_list = [pdb1, pdb2, pdb3, ..., pdbn]
traj_loader = trajloader.TrajectoryLoader(traj_list, range(len(traj_list)))
```

# Featurizer

--------
Featurizer is the primary hook between features and trajectories. 
### Load trajectories to a container and register to a featurizer
```
featurizer = nl.featurizer()
......
......
```

### Start featurization
```
featurizer = nl.featurizer()
......
......
```
### Write your own feature
When defining a new feature, you need to inherit the base class Features and implement the feature function.
```
from nearl.features import Features
class YourFeature(Features): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # your own initialization
    def feature(self, *args, **kwargs):
        # your own feature
        return feature_vector
```
### Register a feature to featurizer
```
from nearl.featurizer import Featurizer
featurizer = Featurizer()
featurizer.register_feature(YourFeature)
```
View the example project featurizing a small subset of the [PDBbind](http://www.pdbbind.org.cn/) dataset
[in this script](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/prepare_pdbbind.py)

# Feature deposition

--------
### NEARL supports the following features 
- TEMPLATE_STRING
```
......
......
```

### Draw the hdf structure
- TEMPLATE_STRING
```
from nearl import hdf 
with hdf.hdf_operator(output_hdffile, readonly=True) as h5file:
    h5file.draw_structure();
```


# Model training

--------
There are several pre-defined models in the [nearl.models](https://github.com/miemiemmmm/BetaPose/tree/main/BetaPose/models) using 
[PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io/en/latest/) framework.
You could easily re-use these models or write your own model. <br>
```
......
......
```

View the example project training on a small dataset in [PyTorch framework](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/train_simple_network.py) 
or [JAX framework](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/train_simple_network_Jax.py)

# License

--------
[MIT License](https://github.com/miemiemmmm/BetaPose/blob/master/LICENSE)