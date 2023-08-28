#!/bin/bash -l

tstamp(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : "
}
logit(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : $@"
}

env_name=${1:-"mlenv"};
mode=${2:-"cpu"};

if [ ${#env_name} -eq 0 ]; then
  logit "Please define a environment name: bash ${0} <environment name>";
  exit -1
else
  logit "Installing the environment named: ${env_name}";
fi

# Check the installation of micromamba via environment variable MAMBA_EXE
if [ ${#MAMBA_EXE} -gt 10 ]; then
  MAMBA_PATH=$(dirname ${MAMBA_EXE})
  export PATH="${MAMBA_PATH}:${PATH}"
  logit "MAMBA_EXE is defined, adding it to the PATH ${MAMBA_PATH}";
else
  logit "MAMBA_EXE is not defined, please check if micromamba is installed or not";
  exit -1;
fi

micromamba env list
foundenv=$(micromamba env list | grep envs | awk '{print $1,$2}' | egrep "/${env_name}$" | awk '{print $NF}')
if [ ${#foundenv} -gt 0 ]; then
  logit "Warning: Found a pre-existing environment with the same name ${foundenv}; Removing it; ";
  echo -e "Y\n" | micromamba env remove --name ${env_name};
fi

logit "Creating the environment, installation of AmberTools might take a while ...";
micromamba create --name ${env_name} -c conda-forge python=3.9.17 ambertools=22.0 -y
eval "$(micromamba shell hook --shell bash)"
micromamba activate ${env_name}

logit "Current active ${CONDA_DEFAULT_ENV} | ${env_name}  <<<<<<<<";
if [[ ${CONDA_DEFAULT_ENV} != ${env_name} ]]; then
  logit "Failed to activate the environment ${env_name}";
  exit -1;
else
  logit "Successfully activated the environment ${CONDA_DEFAULT_ENV}";
  logit "Installing other necessary packages";

  micromamba install -c conda-forge scipy pandas scikit-learn h5py seaborn matplotlib -y
  python -m pip install "dask[complete]"
  pip3 install open3d rdkit

  if [[ ${mode} == "torch" ]]; then
    logit "Installing PyTorch ecosystem";
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  elif [[ ${mode} == "jax" ]]; then
    logit "Installing Jax ecosystem";
    pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip3 install optax flax
  else
    logit "Not designated machine learning model to install (jax|torch), omitting ..."
  fi
fi

micromamba install -c conda-forge -c anaconda notebook==7.6 ipywidgets==7.6 nglview=3.0.3 -y  # traj visualizer
micromamba install -c conda-forge requests biopython trimesh -y
pip3 install line_profiler
pip3 install pybind11 build cmake
