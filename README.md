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
```base
pip install .
```

### Test the installation
Activate the new NEARL environment and run the following commands to test the installation: <br>
```bash
# To test the featurizer: 
python -c "from nearl import tests; tests.vectorize()"
# To test some simple models:
python -c "from nearl import tests; tests.jax_2dcnn()"  
```

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


# Featurizer


### Load trajectories to a container and register to a featurizer
```python
featurizer = nearl.features.Featurizer()
```

### Start featurization
```python
FEATURIZER_PARMS = {
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
feat = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
loader = nearl.io.TrajectoryLoader(trajlists)
feat.register_feature(nearl.features.Mass())
feat.register_focus([":15&@CA"], "mask")
feat.main_loop()
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


### Register the focused miety

```python
# Register the area of interest
feat.register_focus([":MOL"],  "mask")

```

### Manual focus parser
If the built-in focus parser does not meet your needs, you could define your own focus parser. 
The following is an example to read a standalone json file recording the atom indices of the structure of interest. 

```python
def manual_focal_parser(traj): 
  # The reference json file to map the indices to track
  ligandmap = "/MieT5/BetaPose/data/misato_ligand_indices.json"
  # The sliding time window has to match the one put in the featurizer
  timewindow = 10 

  with open(ligandmap, "r") as f:
    LIGAND_INDICE_MAP = json.load(f)
    assert traj.identity.upper() in LIGAND_INDICE_MAP.keys(), f"Cannot find the ligand indices for {traj.identity}"
    ligand_indices = np.array(LIGAND_INDICE_MAP[traj.identity.upper()])

  # Initialize the proper shaped array to store the focal points
  # There has to be three dimensions: number of frame slices, number of focal points, and 3
  FOCALPOINTS = np.full((traj.n_frames // timewindow, 1, 3), 99999, dtype=np.float32)
  for i in range(traj.n_frames // timewindow):
    FOCALPOINTS[i] = np.mean(traj.xyz[i*timewindow][ligand_indices], axis=0)
  return FOCALPOINTS

# To use the manually defined focal point parser, you need to register that function to the featurizer
feat.register_focus(manual_focal_parser, "function")
```



### Define your own feature
When defining a new feature, you need to inherit the base class Features and implement the feature function.

```python
from nearl import commands, utils
from nearl.features import Features

class MyFirstFeature(Features): 
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # your own initialization
    
  # Implement your own feature calculation
  def cache(self, trajectory):
    # this example is to generate a random number for each atom
    self.cached_props = np.random.rand(trajectory.n_atoms)

  # Topology of the trajectory
  def query(self, topology, frames, focual_point):
    frame = frames[0]
    mask_inbox = super().query(topology, frame, focual_point)
    coords = frame.xyz[mask_inbox]
    weights = self.cached_props[mask_inbox]
    return coords, weights

  # your own feature calculation
  def run(self, coords, weights): 
    feature_vector = commands.voxelize_coords(coords, weights,  self.dims, self.spacing, self.cutoff, self.sigma )
    return feature_vector

  def dump(self, feature_vector): 
    # your own output
    utils.append_hdf_data(self.outfile, self.outkey, np.asarray([result], dtype=np.float32), dtype=np.float32, maxshape=(None, *self.dims), chunks=True, compression="gzip", compression_opts=4)
    
```



# Model training

There are several pre-defined models in the [nearl.models](https://github.com/miemiemmmm/BetaPose/tree/main/BetaPose/models) using 
[PyTorch](https://pytorch.org/) 
<!-- and 
[JAX](https://jax.readthedocs.io/en/latest/) framework. -->
You could easily re-use these models or write your own model. <br>
```python
TO BE UPDATED
```

View the example project training on a small dataset in [PyTorch framework](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/train_simple_network.py) 
<!-- or [JAX framework](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/train_simple_network_Jax.py) -->


# Visualize voxelized feature and the molecule block
```python
from nearl import utils, io, data
``` 

# License

[MIT License](https://github.com/miemiemmmm/BetaPose/blob/master/LICENSE)
