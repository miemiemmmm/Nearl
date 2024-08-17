#!/bin/bash -l
# Build the package for Developer purpose
# Four steps:
# 1. Build the package and wheel
# 2. Install the wheel to the current environment
# 3. Install from source distribution as a test
# 4. Switch back to editable mode


# Build the python package and wheel, then install the wheel to the current environment
logit "Building the package in the ${PWD}";

python -m build

if [ $? -ne 0 ]; then
  logit "Error: Python building failed!"
  exit 1;
else
  pip install -v ./dist/nearl-0.0.1-py3-none-any.whl --force-reinstall;
fi

