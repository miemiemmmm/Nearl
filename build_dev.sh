#!/bin/bash -l
# Build the package for Developer purpose
# Four steps:
# 1. Build the package and wheel
# 2. Install the wheel to the current environment
# 3. Install from source distribution as a test
# 4. Switch back to editable mode

env_name="mlenv";

tstamp(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : "
}
logit(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : $@"
}
# Find and initialize conda, then find and activate the desired conda environment
if [ ${#env_name} -eq 0 ]; then
  logit "Please define a environment name: bash ${0} <environment name>";
  exit -1;
fi
if ! command -v conda &> /dev/null; then
  logit "Conda could not be found, please install it first";
  exit -1;
fi
if [ -f $(conda info --base)/etc/profile.d/conda.sh ]; then
  logit "Initializing conda";
  source $(conda info --base)/etc/profile.d/conda.sh;
else
  logit "Conda init script not found, please check if conda is installed correctly";
  exit -1;
fi
foundenv=$(conda info --envs | egrep "/${env_name}$" | awk '{print $1}')
if [ ${#foundenv} -eq 0 ]; then
  logit "Warning: Cannot found the environment ${env_name} to install the package, please check if the environment is created correctly";
  exit -1;
fi
conda activate ${env_name};
if [[ ${CONDA_DEFAULT_ENV} != ${env_name} ]]; then
  logit "Failed to activate the environment ${env_name}";
  exit -1;
else
  logit "Running in the environment ${CONDA_DEFAULT_ENV}";
fi

current_path=$(dirname $(realpath ${0}));
cd ${current_path};
files_to_check="setup.py%pyproject.toml%MANIFEST.in"
echo ${files_to_check} | tr "%" "\n" | while read file; do
  if [ ! -f ${file} ]; then
    logit "Cannot find the file ${file}, please check if the file exists";
    exit -1;
  fi
done

# Build the python package and wheel, then install the wheel to the current environment
logit "Building the package in the ${PWD}";

python -m build -o ${PWD}/dist/;
if [ $? -ne 0 ]; then
  logit "Error: Python building failed!"
  exit 1;
else
  logit "Great! Build succeeded!";
  PACKAGE_NAME=$(egrep "^name" ${PWD}/pyproject.toml | awk -F= '{print $2}' | awk '{print $1}' | sed 's/"//g');
  PACKAGE_VERSION=$(egrep "^version" ${PWD}/pyproject.toml | awk -F= '{print $2}' | awk '{print $1}'  | sed 's/"//g');
  PYTHON_VERSION=cp$(python -c "import sys; print(''.join(map(str, sys.version_info[:2])))");
  EXPECTED_WHL_NAME="${PWD}/dist/${PACKAGE_NAME}-${PACKAGE_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl";
  EXPECTED_SDIST_NAME="${PWD}/dist/${PACKAGE_NAME}-${PACKAGE_VERSION}.tar.gz";
  logit "Expected wheel file name: ${EXPECTED_WHL_NAME}";
  logit "Expected source distribution file name: ${EXPECTED_SDIST_NAME}";

  if [ ! -f ${EXPECTED_WHL_NAME} ]; then
    logit "Error: Cannot find the expected wheel file ${EXPECTED_WHL_NAME}, please check if the build is successful";
    exit -1;
  else
    pip install ${EXPECTED_WHL_NAME} --force-reinstall;
    if [ $? -eq 0 ]; then
      logit "Great! Install succeeded!";
    fi
  fi
fi

echo "####################################################################################"
echo "####################################################################################"
echo "####################################################################################"

install_sdist=false;
if ${install_sdist}; then
  # Install from source distribution as a test
  logit "Installing from source distribution as a test"
  pip install -v ${EXPECTED_SDIST_NAME} --force-reinstall;
  if [ $? -eq 0 ]; then
    logit "Great! Source install succeeded!";
  fi
fi

echo "####################################################################################"
echo "####################################################################################"
echo "####################################################################################"
echo "Finally switch back to editable mode"
pip install -v -e . --force-reinstall;


