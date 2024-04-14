---
title: "Nearl: Extracting dynamic features from molecular dynamics trajectories"
tags: 
  - Python
  - CUDA
  - molecular dynamics
  - computational biology 
authors:
  - name: "Yang Zhang"
    orcid: 0000-0003-0801-8064
    affiliation: 1
  - name: "Andreas Vitalis"
    orcid: 0000-0002-5422-5278
    affiliation: 1
    corresponding: true
  - name: "Amedeo Caflisch"
    orcid: 0000-0002-2317-6792
    affiliation: 1
    corresponding: true
affiliations:
 - name: Department of Biochemistry, University of Zurich, Zurich, Switzerland
   index: 1
date: 1 May 2024
bibliography: paper.bib
---

# Summary 
<!-- High level functionality and purpose of the software and the target of the software -->

We present Nearl, a Python package for extracting dynamic features from molecular dynamics (MD) trajectories. 
Nearl is designed to be fast and efficient, leveraging CUDA for GPU acceleration to transform a substructure of a protein into a 3D grid of local dynamic features and keep track of the global features. <!-- test -->
These features are suitable for the training of 3D convolutional neural networks (3D-CNNs) for the prediction of protein-ligand binding affinities. <!--  -->
The two algorithms implemented in Nearl, marching observers and property-density flow shown in \autoref{fig:dynalgorithms}, provide a flexible way to extract dynamic features from MD trajectories.





![Schematic representation of the two dynamic features. 
\label{fig:dynalgorithms}](
  https://miemiemmmm.b-cdn.net/experience_introduction/Nearl_Dyna_features_Scheme.png
){width=85%}

# State of the field

The 3D convolutional neural network (3D-CNN) is a powerful framework for extracting features from 3D grids.
Current mainstream featurization methods are mainly atom type-based. 


DeepRank [@crocioni2024deeprank2] offers a systematic way to deal with protein-protein interaction for voxelization.
The libmogrid [@sunseri2020libmolgrid] is also a library for voxelization specialized for protein-ligand interaction. 




# Statement of need
<!-- Demand of the tool -->

MD data is requires intensive computation and storage. 
Nearl could facilitate the data-mining of MD trajectory and improve the 


<!-- Comparison with current commonly used tools -->

There are several general pakcages that could process MD trajectories such as pytraj/CPPTRAJ[@Hai2017pytraj; @roe2013ptraj], MDTraj [@McGibbon2015MDTraj], MDAnalysis[@gowers2019mdanalysis]. 

<!-- Limitation of the current software -->

Current available voxelization frameworks mainly focus on static structures and do not provide insights into the dynamic behavior of proteins. 
Nearl aims to fill this gap by providing a fast and efficient way to extract dynamic features from molecular dynamics trajectories.

# Key features
Nearl convert the trajectory into 3D grid representation with global and local features. There are two types of dynamic features extraction algorithms implemented in Nearl:
- Marching observers: grid-major order by calculating local ovservables along a trajectory slice
- Property-density flow: atom-major order by voxelizing into 3D grids across a slice of frames. 

Nearl takes commonly used MD trajectory formats including PDB, NetCDF. 
It also provides a flexible way to customize the input format by allowing users to define their own trajectory supplier.


<!-- Design principles -->

Inspred by the polymorphic design of the PyTorch framework, Nearl abstracts five main components for the full control of the customizability of features. The detailed components are 
In the most time-consuming grid-based operations, CUDA is used to accelerate the computation. 

User have the freedom to control the process of trajectory processing. 


<!-- Implemented various pre-defined properties and features -->

Regarding the diverse atomic properties
 all input structures as a trajectory and provides a 

data supplier that occupies a process pool to concurrently access the data from the HDF file.  
The 
Multiple files could be access



Nearl implemented several 3D-CNN models models in [PyTorch](https://pytorch.org/) and their pretrained weights for the scoring of ligands via a well documented commandline interface.  



# Acknowledgements
This work is supported by the Swiss National Science Foundation (SNSF) grant xxxxxxxx.


# References

