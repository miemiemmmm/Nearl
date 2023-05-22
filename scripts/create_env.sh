#!/bin/bash -l 

# ${1}
conda create -n mlprod python=3.8 -y
conda activate mlprod
conda install -c conda-forge ambertools -y
conda install -c conda-forge -c anaconda ipywidgets=7.6 notebook nglview=3.0.3 -y
conda install -c conda-forge -c anaconda h5py scipy numpy scikit-learn pandas requests seaborn biopython -y

pip install open3d rdkit matplotlib sgt




