#!/bin/bash -l 

#env_name=${1}
tstamp(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : "
}
logit(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : $@"
}

env_name="mltest"; 

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
conda create --name ${env_name} -c conda-forge python=3.10 ambertools=22.0 -y
logit "Activating the environment ${env_name}";
conda activate ${env_name}

logit "Current active ${CONDA_DEFAULT_ENV} | ${env_name}  <<<<<<<<";
if [[ ${CONDA_DEFAULT_ENV} != ${env_name} ]]; then
  logit "Failed to activate the environment ${env_name}";
  exit -1;
else
  logit "Successfully activated the environment ${CONDA_DEFAULT_ENV}";
  logit "Installing other necessary packages";

  # conda install -c conda-forge -c anaconda scipy=1.10 numpy=1.24 pandas=2.0 scikit-learn=1.2 seaborn matplotlib h5py dask -y
  pip install scipy pandas scikit-learn h5py seaborn matplotlib
  python -m pip install "dask[complete]"
  pip3 install open3d rdkit sgt
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

fi


# Packages useful for development
conda install -c conda-forge -c anaconda notebook nglview=3.0.3 -y
conda install -c conda-forge requests biopython -y
# Jax ecosystem
pip3 install jax optax flex

# Other packages
# imageio=2.9.0 # hilbertcurve=2.0.5


