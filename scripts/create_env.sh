#!/bin/bash -l 

#env_name=${1}
tstamp(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : "
}

env_name="mltest"; 

if [ ${#env_name} -eq 0 ]; then 
  echo "$(tstamp)Please define a environment name: bash ${0} <environment name>";
  exit -1
else
  echo "$(tstamp)Installing the environment named: ${env_name}";
fi

# Initialize the conda
echo "$(tstamp)Looking for the conda init script";
if [ -f $(conda info --base)/etc/profile.d/conda.sh ]; then
  echo "$(tstamp)Initializing conda";
  source $(conda info --base)/etc/profile.d/conda.sh;
fi

foundenv=$(conda info --envs | egrep "/${env_name}$" | awk '{print $1}')
echo "$(tstamp)Found the environment: ${foundenv}"
if [ ${#foundenv} -eq 0 ]; then
  echo "$(tstamp)Found the pre-existing environment with the same name ${foundenv}; Removing it; ";
  conda env remove --name ${env_name} || true;
fi

echo "$(tstamp)Creating the environment, installation of AmberTools might take a while ...";
conda create --name ${env_name} -c conda-forge python=3.10 ambertools=22.0 -y
echo "$(tstamp)Activating the environment ${env_name}";
conda activate ${env_name}

echo "$(tstamp)Current active ${CONDA_DEFAULT_ENV} | ${env_name}  <<<<<<<<";
if [[ ${CONDA_DEFAULT_ENV} != ${env_name} ]]; then
  echo "$(tstamp)Failed to activate the environment ${env_name}";
  exit -1;
else
  echo "$(tstamp)Successfully activated the environment ${CONDA_DEFAULT_ENV}";
  echo "$(tstamp)Installing other necessary packages";

  conda install -c conda-forge -c anaconda scipy=1.10 numpy=1.24 pandas=2.0 scikit-learn=1.2 seaborn matplotlib h5py -y
  pip3 install open3d rdkit sgt
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi


# Other packages used for development
conda install -c conda-forge -c anaconda notebook nglview=3.0.3 -y
conda install -c conda-forge requests biopython -y


# imageio=2.9.0 # hilbertcurve=2.0.5


