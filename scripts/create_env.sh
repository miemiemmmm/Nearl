#!/bin/bash -l 

#env_name=${1}
tstamp(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : "
}
logit(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : $@"
}

env_name="mljax"; 

if [ ${#env_name} -eq 0 ]; then 
  logit "Please define a environment name: bash ${0} <environment name>";
  exit -1
else
  logit "Installing the environment named: ${env_name}";
fi

# Check if the conda is installed
if ! command -v conda &> /dev/null; then
  logit "Conda could not be found, please install it first";
  exit -1;
fi

# Initialize the conda
if [ -f $(conda info --base)/etc/profile.d/conda.sh ]; then
  logit "Initializing conda";
  source $(conda info --base)/etc/profile.d/conda.sh;
else
  logit "Conda init script not found, please check if conda is installed correctly";
  exit -1;
fi

foundenv=$(conda info --envs | egrep "/${env_name}$" | awk '{print $1}')
if [ ${#foundenv} -eq 0 ]; then
  logit "Warning: Found a pre-existing environment with the same name ${foundenv}; Removing it; ";
  conda env remove --name ${env_name} || true;
fi

logit "Creating the environment, installation of AmberTools might take a while ...";
conda create --name ${env_name} -c conda-forge python=3.9.11 ambertools=22.0 -y
logit "Activating the environment ${env_name}";
conda activate ${env_name}

logit "Current active ${CONDA_DEFAULT_ENV} | ${env_name}  <<<<<<<<";
if [[ ${CONDA_DEFAULT_ENV} != ${env_name} ]]; then
  logit "Failed to activate the environment ${env_name}";
  exit -1;
else
  logit "Successfully activated the environment ${CONDA_DEFAULT_ENV}";
  logit "Installing other necessary packages";

  pip3 install scipy pandas scikit-learn h5py seaborn matplotlib numba
  python -m pip install "dask[complete]"
  pip3 install open3d rdkit

  if (( ${1} == "torch" )); then
    logit "Installing PyTorch ecosystem";
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  elif (( ${1} == "jax" )); then
    logit "Installing Jax ecosystem";
    pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip3 install optax flax
  else
    logit "Not designated machine learning model to install (jax|torch), omitting ..."
  fi
fi



# Packages useful for development
conda install -c conda-forge -c anaconda notebook nglview=3.0.3 trimesh -y
conda install -c conda-forge requests biopython -y
pip3 install line_profiler
pip3 install pybind11

# Other packages
# imageio=2.9.0 # hilbertcurve=2.0.5

