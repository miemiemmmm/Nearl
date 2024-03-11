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

# Build the python package and wheel, then install the wheel to the current environment
logit "Building the package in the ${PWD}";

python -m build


if [ $? -ne 0 ]; then
  logit "Error: Python building failed!"
  exit 1;
else
  logit "Great! Build succeeded!";
  pip install -v ./dist/nearl-0.0.1-py3-none-any.whl --force-reinstall;

  # PACKAGE_NAME=$(egrep "^name" ${PWD}/pyproject.toml | awk -F= '{print $2}' | awk '{print $1}' | sed 's/"//g');
  # PACKAGE_VERSION=$(egrep "^version" ${PWD}/pyproject.toml | awk -F= '{print $2}' | awk '{print $1}'  | sed 's/"//g');
  # PYTHON_VERSION=cp$(python -c "import sys; print(''.join(map(str, sys.version_info[:2])))");
  # EXPECTED_WHL_NAME="${PWD}/dist/${PACKAGE_NAME}-${PACKAGE_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl";
  # EXPECTED_SDIST_NAME="${PWD}/dist/${PACKAGE_NAME}-${PACKAGE_VERSION}.tar.gz";

fi

