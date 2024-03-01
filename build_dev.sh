#!/bin/bash -l
# Build the package for Developer purpose
# Four steps:
# 1. Build the package and wheel
# 2. Install the wheel to the current environment
# 3. Install from source distribution as a test
# 4. Switch back to editable mode

logit(){
  echo ">>>> $(date '+%d.%m.%Y %H.%M.%S') : $@"
}

env_name=${1};
if [ ${#env_name} -eq 0 ]; then
  logit "Please define a environment name: bash ${0} <environment name>";
  echo "Usage: bash ${0} <environment name>";
  exit -1;
fi

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


