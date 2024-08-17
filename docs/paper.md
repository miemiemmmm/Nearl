---
title: "Nearl: Extracting dynamic features from molecular dynamics trajectories"
tags: 
  - Python
  - CUDA
  - molecular dynamics
  - computational biology 
  - deep learning
  - 3D-CNN
  - machine learning
  - drug discovery
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
Despite the rapid growth of machine learning in biomolecular applications, information about protein dynamics is underutilized.
Here, we introduce Nearl, an automated Python pipeline designed to extract dynamic features from large ensembles of molecular dynamics (MD) trajectories. 
Nearl aims to identify intrinsic patterns of molecular motion and to provide informative features for predictive modelling tasks. 
We implement two classes of dynamic features, termed marching observers and property-density flow, to capture local atomic motions while maintaining a view of the global configuration.
Complemented by standard voxelization techniques, Nearl transforms substructures of proteins into 3D grids, suitable for contemporary 3D convolutional neural networks (3D-CNNs).
The pipeline leverages modern GPU acceleration, adheres to the FAIR principles for research software, and prioritizes flexibility and user-friendliness, allowing customization of input formats and feature extraction.

<!-- 124 words 982 characters -->

![Workflow of Nearl for transforming MD trajectories into 3D dynamic features for 3D-CNN applications. 
Featurizer could register a set of features and trajectories for their target computation.
Registered components are processed sequentially for the computation 3D dynamic features. 
The computed features are saved in HDF5 format for efficient use by downstream 3D-CNN models.
\label{fig:dynalgorithms}](media/nearl_workflow.png){width=95%}

# State of the field
<!-- ### Intro ML and Example with and 3D CNN -->
Recent advancements in machine learning have significantly reshaped the landscape of protein structure prediction [@jumper2021highly; @baek2021accurate; @zheng2021folding], protein design [@Dauparas2022], and generative drug discovery [@GmezBombarelli2018; @de2018molgan] and expand the capabilities of traditional computational biology methods by incorporating techniques ranging from shallow learning to complex neural networks and even reinforcement learning.
<!-- Intro to 3D-CNN -->
Three-dimensional voxels represent the spatial distribution of atoms within a molecule, essentially functioning as 3D images[@Maturana2015]. 3D convolutional neural networks apply the convolution operation to 3D space and offer a popular framework for tasks such as predicting protein-ligand binding affinity [@mcnutt2021gnina; @stepniewska2018development; @hassan2020rosenet; @jimenez2018k; @townshend2020atom3d], protein-protein interface prediction [@crocioni2024deeprank2; @townshend2018end], and binding pocket prediction [@Simonovsky2020].

<!-- ### Molecular dynamics / Motivation -->
Molecular dynamics (MD) simulations are an established and widely used tool to predict the evolution of a molecular system based on physical laws [@McCammon1977; @Karplus2002].
They allow atomic-level exploration of the conformational space of biomolecules, providing insights for various applications such as the design of ligands and biomolecules, guiding the interpretation of experimental results, and investigating the dynamic behavior of proteins [@Hollingsworth2018].
<!-- ### Current strategy in information condensation -->
Several strategies have been developed to bridge the gap between the information contained in MD trajectories and the requirements posed toward inputs of machine learning frameworks.
The high dimensionality of the phase space sampled in MD motivates employing dimensionality reduction and information condensation techniques. This is often achieved by calculating a set of simple collective variables (CVs) that effectively capture the essential dynamics of the system [@gorostiola20233ddpds; @riniker2017molecular; @cocina2020sapphire; @Morishita2021]. 
In recent years, alternative methods for 3D feature embedding have been proposed, which move beyond traditional CVs, such as geometric bit vectors [@Kumar2024], atom-centered symmetry functions [@Gallegos2024], Coulomb matrices [@Farahvash2020], and 2D grayscale images of radial distribution functions [@Ayush2023]. 



# Statement of need

<!-- Statement of the technical problem  -->
Mining motion patterns from MD trajectories is demanding due to their size (the biopolymers alone routinely contain ~10000 atoms, and individual trajectories contain 1000s of samples) and the heterogeneity across different components (polymers vs solvent vs small molecules).
Extracting atomic motions, conformational changes, and molecular interactions as 3D features requires significant computational power, extensive file input/output operations, and substantial file storage. 
<!-- #### Comparison with current commonly used tools and their limitations -->
<!-- Common MD analysis tools -->
Current mainstream MD analysis tools such as PYTRAJ/CPPTRAJ [@Hai2017pytraj; @roe2013ptraj], MDTraj [@McGibbon2015MDTraj], MDAnalysis [@gowers2019mdanalysis], process MD trajectories but primarily focus on simple features like RMSD, RMSF, and hydrogen bonds, which usually require manual definition. 
<!-- Common voxelization tools -->
The majority of 3D-CNN frameworks require specialized voxelization schemes [@stepniewska2018development; @hassan2020rosenet] that are not universally transferable to other applications.
While tools like DeepRank [@crocioni2024deeprank2] and libmogrid [@sunseri2020libmolgrid] 
systematically address the voxelization of protein-protein interactions and atomic properties, they mainly focus on static structures and do not provide insights into the dynamic behavior of proteins.

<!-- #### Demand for our tool -->
Due to the data-hungry nature of deep learning frameworks [@adadi2021survey], MD trajectories present a valuable resource both for data augmentation of the experimentally visible, conformational space and for incorporating information about dynamics. 
The recent emergence of large-scale MD trajectory datasets [@siebenmorgen2023misato; @rodriguez2020gpcrmd; @meyer2010model; @korlepara2022plas; @korlepara2024plas] and standardized trajectory formats [@McGibbon2015MDTraj] has significantly improved the accessibility of MD results for researchers. 
Consequently, there is a high demand for a framework that facilitates batch processing of MD trajectories and dynamic feature extraction.


<!-- ######### -->
<!-- Main body -->
# Key features

<!-- Dynamic features extraction -->
Nearl converts MD trajectories of user-defined subsystems into 3D grid representations encoded by the following dynamic features.
Marching observers monitor individual grid points and calculate a local observable along the time axis. With each observer being able to perceive their local environment, the collection of these observers is able to grasp local atomic motions while maintaining perception of the global configuration. 
Property-density flow instead voxelizes the atoms into 3D grids for individual frames and aggregates the properties along the time axis per grid point. 
The most expensive grid-based computations, such as Gaussian interpolation, computation of observables, and time-series aggregation, are accelerated by Compute Unified Device Architecture (CUDA) on NVIDIA GPU.
Apart from these two dynamic features, the voxelization of static structures is naturally implemented in Nearl with various pre-implemented atomic properties.
Further details can be found in the [documentation](http://nearl.readthedocs.io/).

<!-- Design principles -->
Inspired by the polymorphic design principles of [PyTorch](https://pytorch.org/)[@paszke2019pytorch], Nearl adopts a modular and flexible approach for 3D feature generation. This approach abstracts the process into five steps covering the life cycle of each feature computation: 1. defining global perception settings (subsystem size, etc); 2. caching of properties (one-time); 3. extracting the subsystem from trajectory frames; 4. computing features; 5. dumping of results. 
This modular design lets users customize their workflows fully: they can register any number of desired features with corresponding labels, thus facilitating the preparation of training data tailored to specific 3D-CNN models.


<!-- Input supports --> 
Nearl accommodates common trajectory file formats, including PDB, NetCDF, and DCD, to ensure compatibility with a wide range of workflows. Notably, Nearl handles static PDB files by treating them implicitly as single-frame trajectories. Furthermore, the pipeline adheres to a polymorphic/modular design for the trajectory module: users can easily insert their own custom trajectory suppliers.
<!-- Output supports -->
Processed results are by default stored in Hierarchical Data Format (HDF) files, leveraging their high performance in data access. Users can readily switch to alternative storage formats by modifying the dump method within the features module. Notably, the HDF dataset supplier employs process pools during model training, enabling concurrent access to data across multiple HDF files for enhanced efficiency.
<!-- Model supports -->
For seamless integration with deep learning frameworks, Nearl implements several pre-built 3D-CNNs within the PyTorch framework. These models are adapted from established architectures [@mcnutt2021gnina; @stepniewska2018development; @crocioni2024deeprank2; @townshend2020atom3d]. 

<!-- Nearl provides a fast and efficient tool to extract dynamic features from MD trajectories to  -->
This project aims to bridge the current gap between deep learning applications and MD simulations and to enhance the reusability of MD trajectories, thereby unlocking their full potential for informing and accelerating biological research. 
Nearl's design prioritizes user customization and extensibility, which allows researchers to seamlessly integrate their own features and models.
We anticipate that Nearl will be a valuable tool for computational biologists, bioinformaticians, and machine learning engineers. These researchers can deploy Nearl to incorporate information about biomolecular dynamics into diverse areas of investigation, including drug discovery, molecular design, and structural biology.

<!-- 
Shall I again mention the FAIR principles like how I adhere to them? 
Abbreviation of RMSD, RMSF. 
Callback the dynamics feature extraction in the conclusion sentence. 
-->

<!-- 1047 words: a rough word count -->

# Acknowledgements
This work was supported in part by the Swiss National Science Foundation grant #189363.


# References

