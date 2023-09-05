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

### Clone the repository
```
git clone https://github.com/miemiemmmm/NEARL.git
cd NEARL
```

### Manage your python environment
[Mamba](https://mamba.readthedocs.io/en/latest/) is a Python package manager implemented in C++ and aims to provide all 
the functionality of [Conda](https://docs.conda.io/en/latest/) but with higher speed. 
[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) is a lighter-weight version of Mamba, 
aiming to provide a minimal, fast, and standalone executable for environment and package management. They both can be used as a drop-in replacement for Conda and 
we recommend using micromamba to manage the python environement. <br>
If there is no micromamba installed on your system, the following script could help you to install micromamba <br>
```
# The following command downloads the micromamba to /home/myname/micromamba/bin and generates a loadmamba script
bash scripts/install_mamba.sh /home/myname/micromamba 

# Use the following command to configure the shell to use micromamba. 
# To load micromamba upon starting a new shell, add this line to .bashrc or .zshrc
source /home/myname/micromamba/bin/loadmamba
```

### Create a test environment  
Load the mamba environment and create a new environment named NEARL <br>
```
bash scripts/create_env_mamba.sh NEARL jax
micromamba activate NEARL
```


### Install NEARL
NEARL supports only the Linux platform for the time being. It is recommended to install via PyPI: <br>
```pip install nearl``` <br>
By defaults, it uses [OpenMP](https://www.openmp.org/) when doing feature density interpolation, there are some 
key components accelerated by [OpenACC](https://www.openacc.org/). To install the GPU version, [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)
is required. <br>
Use the following command to install the GPU version: <br>
```

pip install .
```

### Test the installation
Activate the new NEARL environment and run the following commands to test the installation: <br>
```
# To test the featurizer: 
python -c "from nearl import tests; tests.vectorize()"
# To test some simple models:
python -c "from nearl import tests; tests.jax_2dcnn()"  
```

# Get started

--------
```
import nearl as nl
_trajfile, _topfile = nl.data.MINI_TRAJ
_parms = nl.data.MINI_PARMS
loader = nl.io.TrajectoryLoader(_trajfile, _topfile)
feat = nl.features.Featurizer3D(_parms)
feat.register_feature(nl.features.Mass())
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
trajloader = TrajectoryLoader(trajs, tops, **kwarg)
for traj in trajloader:
  # Do something with traj
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
featurizer = nl.features.Featurizer3D()
......
......
```

### Start featurization
```
feat = nl.features.Featurizer3D()
feat.register_feature(nl.features.Mass())
feat.register_frame()
......
```

### Register a feature to featurizer
```
from nearl.featurizer import Featurizer
featurizer = Featurizer()
featurizer.register_feature(YourFeature)
feat.register_traj(trajectory)
feat.register_frames(range(100))
index_selected = trajectory.top.select(":LIG")
repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog")

```
View the example project featurizing a small subset of the [PDBbind](http://www.pdbbind.org.cn/) dataset
[in this script](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/prepare_pdbbind.py)


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

# Feature data deposition

--------
### NEARL supports the following features 
- TEMPLATE_STRING
```
......
......
```

### Draw the hdf structure
- Since temporal features are 
```angular2html
from nearl import hdf 
with hdf.hdf_operator(output_hdffile, readonly=True) as h5file:
    h5file.draw_structure();
```


# Model training

--------
There are several pre-defined models in the [nearl.models](https://github.com/miemiemmmm/BetaPose/tree/main/BetaPose/models) using 
[PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io/en/latest/) framework.
You could easily re-use these models or write your own model. <br>
```angular2html
......
......
```

View the example project training on a small dataset in [PyTorch framework](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/train_simple_network.py) 
or [JAX framework](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/train_simple_network_Jax.py)

# Visualize the trajectory
```angular2html
from nearl import utils, io, data
config = {
  ":LIG<:10&!:SOL,T3P": "ribbon", 
  ":LIG<:5&!:SOL,T3P,WAT": "line", 
  ":LIG": "ball+stick", 
}

traj = io.traj.Trajectory(*data.traj_pair_1)
traj.top.set_reference(traj[0])

dist, info = utils.dist_caps(traj, ":LIG&!@H=", ":LIG<:6&@CA,C,N,O,CB")
tv = utils.TrajectoryViewer(traj)
tv.config_display(config)
tv.add_caps(info["indices_group1"], info["indices_group2"])
tv.resize_stage(400,400)
tv.viewer
```

# Visualize voxelized feature and the molecule block
```angular2html
from nearl import utils, io, data
``` 

# License

--------
[MIT License](https://github.com/miemiemmmm/BetaPose/blob/master/LICENSE)