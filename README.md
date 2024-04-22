# NEARL
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

[NEARL](https://github.com/miemiemmmm/BetaPose)(Nanoscale Environment Assessment and Resonance Landscapes) is a 3D structural 
data generation framework to featurize bio-molecules specifically focus on their 3D coordinate and protein dynamics 
to make users benefit from the recent development in machine learning algorithms. <br>

- Obtain and embed molecule blocks from 3D molecular structures
- Load arbitrary number of 3D structures into a trajectory container
- Multiple pre-defined 2D or 3D features for the featurization
- Pipeline for featurizing the trajectory container


<!-- 
## NEARL
* [Installation](#Installation)
* [Get started](#Get-started)
* [Trajectory loader](#Trajectory-loader)
* [Featurizer](#Featurizer)
* [Feature deposition](#Feature-deposition)
* [Model training](#Model-training)
* [License](#License) 
-->


Installation
------------
NEARL supports only the Linux platform for the time being. It is recommended to install via PyPI: 
CUDA acceleration is used for grid-based operations. NVCC is required to compile the CUDA code. 

```bash 
pip install nearl
``` 

or install via the source code: 

```
git clone https://github.com/miemiemmmm/Nearl.git
cd Nearl
pip install .


```

Validate installation
---------------------

Activate the new NEARL environment and run the following commands to test the installation: <br>
```bash
# To test the featurizer: 
python -c "from nearl import tests; tests.vectorize()"
# To test some simple models:
# python -c "from nearl import tests; tests.jax_2dcnn()"  
```


<!-- 
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
-->




# Get started

```
import nearl
_trajfile, _topfile = nearl.data.MINI_TRAJ
_parms = nearl.data.MINI_PARMS
loader = nearl.io.TrajectoryLoader(_trajfile, _topfile)
feat = nearl.features.Featurizer3D(_parms)
feat.register_feature(nearl.features.Mass())
......
......
```

Documentation
-------------

The detailed documentation is available at [ReadTheDocs](https://nearl.readthedocs.io/en/latest/). 


<!-- 
Citation
--------
If you find this repository useful in your research, please consider citing the following <a href="">paper</a>. 
```bibtex
``` 
-->


<!-- License
------- -->



# Trajectory loader

### Load structures into trajectory container
NEARL regards every 3D structure as trajectories rather than separate molecules. [pytraj](https://amber-md.github.io/pytraj/latest/index.html) is the backend for trajectory processing. <br>

Trajectory loader currently supports the following formats: **NetCDF**, **PDB**, **XTC**, **TRR**, **DCD**, **h5/hdf**. <br>
The trajectory loader normally reads trajectory/topology pairs. 
```python
from nearl import trajloader
traj_list = [traj1, traj2, traj3, ..., trajn]
top_list = [top1, top2, top3, ..., topn]
traj_loader = trajloader.TrajectoryLoader(traj_list, top_list)
trajloader = TrajectoryLoader(trajs, tops, **kwarg)
for traj in trajloader:
  # Do something with traj
  pass
```

### Static PDB structures or Single snapshots
Single snapshot from MD or static structure (like PDB) are dealt as a trajectory with only one frame. If this is the case, 
you could only needs to load the structure as  

```python
from nearl.io import Trajectory, TrajectoryLoader
traj_list = [pdbfile1, pdbfile2, pdbfile3, ..., pdbfileN]
traj_loader = TrajectoryLoader([(i,) for i in traj_list], trajtype=Trajectory)
```



# Model training

There are several pre-defined models in the [nearl.models](https://github.com/miemiemmmm/BetaPose/tree/main/BetaPose/models) using 
[PyTorch](https://pytorch.org/) 
You could easily re-use these models or write your own model. <br>
```python
TO BE UPDATED
```

View the example project training on a small dataset in [PyTorch framework](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/train_simple_network.py) 


# Visualize voxelized feature and the molecule block
```python
from nearl import utils, io, data
``` 


