#!/bin/bash
# Install the micromamba to be able to fast deploy the python environment
# The default base directory is the current working directory
# Usage: bash install_mamba.sh <base directory>
if [ ${#1} -eq 0 ]; then
  echo "Usage: bash install_mamba.sh <base directory>" 1>&2
  exit 0
fi
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
export PATH="$(realpath ${BIN_FOLDER}):${PATH}";
export MAMBA_EXE=${BIN_FOLDER}/micromamba;
export MAMBA_ROOT_PREFIX=${PREFIXLOCATION};
eval "$(micromamba shell hook --shell bash --root-prefix ${PREFIXLOCATION})";


# Finally check if micromamba is able to create a new environment and activate it
echo "Installing a test environment"
micromamba create --name test_installation python=3.9.17 -c conda-forge -q -y
micromamba activate test_installation
if micromamba env list | grep test_installation | grep "*" &> /dev/null; then
  echo "Great! Micromamba installation successful and test environment activated";
  echo -e "Y\n" | micromamba env remove --name test_installation -q -y;
else
  echo "No!!!! Micromamba cannot activate the test_installation environment";
  exit 1
fi

loadmamba_b64="YmFzZWRpcj0kezE6LVRFTVBMQVRFRElSfQpiYXNlZGlyPSQocmVhbHBhdGggJGJhc2VkaXIpCkJJTl9GT0xERVI9IiR7YmFzZWRpcn0vYmluIgpQUkVGSVhMT0NBVElPTj0iJHtiYXNlZGlyfSIKZXhwb3J0IFBBVEg9IiQocmVhbHBhdGggJHtCSU5fRk9MREVSfSk6JHtQQVRIfSIKZXhwb3J0IE1BTUJBX0VYRT0ke0JJTl9GT0xERVJ9L21pY3JvbWFtYmE7CmV4cG9ydCBNQU1CQV9ST09UX1BSRUZJWD0ke1BSRUZJWExPQ0FUSU9OfTsKZXZhbCAiJChtaWNyb21hbWJhIHNoZWxsIGhvb2sgLS1zaGVsbCBiYXNoIC0tcm9vdC1wcmVmaXggJHtQUkVGSVhMT0NBVElPTn0pIgo="
python3 -c "import base64; tmp_str=base64.b64decode(\"${loadmamba_b64}\".encode('utf-8')).decode('utf-8'); print(tmp_str)" | sed "s|TEMPLATEDIR|${basedir}|g" > ${basedir}/bin/loadmamba
chmod +x ${basedir}/bin/loadmamba

echo "If you wish to load micromamba upon opening a new shell, please add the following lines to your ~/.bashrc or ~/.zshrc"
echo -e '    source '${basedir}'/bin/loadmamba'

