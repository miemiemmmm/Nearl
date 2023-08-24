#!/bin/bash
# Install the micromamba to be able to fast deploy the python environment
# The default base directory is the current working directory
# Usage: bash install_mamba.sh <base directory>
set -eu


basedir=${1:-./micromamba}
basedir=$(realpath $basedir)
mkdir -p ${basedir}/bin
echo "Installing micromamba to ${basedir}"
export BIN_FOLDER="${basedir}/bin"
export PREFIXLOCATION="${basedir}"
export INIT_YES="N"
export CONDA_FORGE_YES="Y"


ARCH="$(uname -m)"
OS="$(uname)"

if [[ "$OS" == "Linux" ]]; then
  PLATFORM="linux"
  if [[ "$ARCH" == "aarch64" ]]; then
    ARCH="aarch64"
  elif [[ $ARCH == "ppc64le" ]]; then
    ARCH="ppc64le"
  else
    ARCH="64"
  fi
elif [[ "$OS" == "Darwin" ]]; then
  PLATFORM="osx"
  if [[ "$ARCH" == "arm64" ]]; then
    ARCH="arm64"
  else
    ARCH="64"
  fi
elif [[ "$OS" =~ "NT" ]]; then
  PLATFORM="win"
  ARCH="64"
else
  echo "Failed to detect your OS" >&2
  exit 1
fi


RELEASE_URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-${PLATFORM}-${ARCH}"
echo "Basic info: $PLATFORM $ARCH $OS"
echo "Config settings: $BIN_FOLDER $PREFIXLOCATION $INIT_YES $CONDA_FORGE_YES"
echo "Release url: "$RELEASE_URL


curl "${RELEASE_URL}" -o "${BIN_FOLDER}/micromamba" -fsSL --compressed ${CURL_OPTS:-}
chmod +x "${BIN_FOLDER}/micromamba"

# Initializing shell
if [[ "$INIT_YES" == "" || "$INIT_YES" == "y" || "$INIT_YES" == "Y" || "$INIT_YES" == "yes" ]]; then
  case "$("${BIN_FOLDER}/micromamba" --version)" in
    1.*|0.*)
      "${BIN_FOLDER}/micromamba" shell init -p "${PREFIXLOCATION}"
      ;;
    *)
      "${BIN_FOLDER}/micromamba" shell init --root-prefix "${PREFIXLOCATION}"
      ;;
  esac

  echo "Please restart your shell to activate micromamba or run the following:\n"
  echo "  source ~/.bashrc (or ~/.zshrc, ...)"
fi


# Initializing conda-forge
if [[ "$CONDA_FORGE_YES" == "" || "$CONDA_FORGE_YES" == "y" || "$CONDA_FORGE_YES" == "Y" || "$CONDA_FORGE_YES" == "yes" ]]; then
  "${BIN_FOLDER}/micromamba" config append channels conda-forge
  "${BIN_FOLDER}/micromamba" config append channels nodefaults
  "${BIN_FOLDER}/micromamba" config set channel_priority strict
fi


# Several steps to initialize and hook the micromamba
export PATH="$(realpath ${BIN_FOLDER}):${PATH}"
export MAMBA_EXE=${BIN_FOLDER}/micromamba;
export MAMBA_ROOT_PREFIX=${PREFIXLOCATION};
eval "$(micromamba shell hook --shell bash --root-prefix ${PREFIXLOCATION})"


# Finally check if micromamba is able to create a new environment and activate it
micromamba create -y -n test_installation python=3.9.17 -c conda-forge
micromamba activate test_installation
if micromamba env list | grep test_installation | grep "*" &> /dev/null; then
  echo "Great! Micromamba installation successful and test environment activated"
else
  echo "No!!!! Micromamba cannot activate the test_installation environment"
  exit 1
fi
