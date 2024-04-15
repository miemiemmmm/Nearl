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
Despite the rapid growth of machine learning in biomolecular applications, the protein dynamics information is often underutilized.
Here, we introduce Nearl, an automated pipeline designed to extract dynamic features from large ensembles of molecular dynamics (MD) trajectories. 
Nearl aims to identify intrinsic motion patterns and provide informative dynamic features for diverse biomolecular applications. 
It implemented two dynamic features: marching observers (MOBS) and property-density flow (PDF), capturing local atomic motions while maintaining the global configuration
Complemented by standard static feature voxelization, Nearl transforms substructures of proteins into 3D grids, suitable for contemporary 3D convolutional neural network (3D-CNN) frameworks.
The pipeline leverages modern GPU acceleration, adheres to the FAIR principles for research software, and prioritizes flexibility and user-friendliness, allowing customization of input formats and feature extraction.


<!-- 134 words 1,023 characters -->

![Schematic representation of the two dynamic features. 
\label{fig:dynalgorithms}](nearl_workflow.png){width=95%}

# State of the field
<!-- ### Intro ML and Example with and 3D CNN -->
The recent advancements in machine learning, particularly deep learning, have significantly reshaped the landscape of protein structure prediction [@jumper2021highly; @baek2021accurate; @zheng2021folding], protein design [@Dauparas2022], and generative drug discovery [@GmezBombarelli2018; @de2018molgan]. 
This advancement broadens the capabilities of traditional computational biology methods by incorporating techniques ranging from shallow learning to complex neural networks and even reinforcement learning.
<!-- Intro to 3D-CNN -->
Three-dimensional voxels represent the spatial distribution of atoms within a molecule, essentially functioning as 3D images[@Maturana2015]. The 3D convolutional neural network (3D-CNN) applies the convolution operation to the 3D space, and becomes a prominent framework for tasks such as predicting protein-ligand binding affinity [@mcnutt2021gnina; @stepniewska2018development; @hassan2020rosenet; @jimenez2018k; @townshend2020atom3d], protein-protein interface prediction [@crocioni2024deeprank2; @townshend2018end] and binding pocket prediction [@Simonovsky2020].

<!-- ### Molecular dynamics / Motivation -->
Molecular dynamics (MD) simulation, leveraging its ability to predict the evolution of a molecular system based on physical laws [@McCammon1977; @Karplus2002], has emerged as a powerful tool in biomolecular research.
It allows atomic-level exploration of the conformational space of biomolecules, providing insights for various applications such as the design of ligands and biomolecules, guiding interpretation of experimental results, and investigating the dynamic behavior of proteins [@Hollingsworth2018].
<!-- ### Current strategy in information condensation -->
Several strategies have been developed to bridge the gap between the rich information contained in MD trajectories and the requirements of machine learning frameworks.
Due to the high dimensionality of the phase space explored during simulations, a common approach involves dimensionality reduction and information condensation. This is often achieved by calculating a set of simple collective variables (CVs) that effectively capture the essential dynamics of the system [@gorostiola20233ddpds, @riniker2017molecular, @cocina2020sapphire, @Morishita2021]. 
In recent years, researchers have explored alternative methods for 3D feature embedding that move beyond traditional CVs, such as geometric bit vector [@Kumar2024], atom centered symmetry functions [@Gallegos2024], Coulomb matrix [@Farahvash2020], and 2D grayscale image of radial distribution function curve [@Ayush2023]. 



# Statement of need

<!-- Statement of the technical problem  -->
Mining motion patterns from MD trajectories is complex due to their high dimensionality (up to a million atoms for explicit solvent simulations), heterogeneity among different components and hierarchical structures of the simulation system.
Capturing 3D features like atomic motion, conformational changes, and molecular interactions requires significant computational power, extensive file input/output operations, and substantial file storage. 
<!-- #### Comparison with current commonly used tools and their limitations -->
<!-- Common MD analysis tools -->
Current mainstream MD analysis tools such as pytraj/CPPTRAJ [@Hai2017pytraj; @roe2013ptraj], MDTraj [@McGibbon2015MDTraj], MDAnalysis [@gowers2019mdanalysis], efficiently process MD trajectories but primarily focus on simple features like RMSD, RMSF, and hydrogen bonds, which usually require manual definition. 
<!-- Common voxelization tools -->
The majority of 3D-CNN frameworks require specialized voxelization schemes [@stepniewska2018development; hassan2020rosenet] that are not universally transferable to other applications.
While tools like DeepRank [@crocioni2024deeprank2] and libmogrid [@sunseri2020libmolgrid] 
systematically address voxelization of protein-protein interactions (PPI) and atomic properties, they mainly focus on static structures and do not provide insights into the dynamic behaviors of proteins.

<!-- #### Demand for our tool -->
Due to the data-hungry nature of deep learning frameworks [@adadi2021survey], MD trajectories present a valuable resource for data augmentation and an attractive strategy for incorporating protein dynamics information by filling the sparse conformational space. 
The recent emergence of large-scale MD trajectory datasets [@siebenmorgen2023misato; @rodriguez2020gpcrmd; @meyer2010model; @korlepara2022plas; @korlepara2024plas] and standardized trajectory formats [@McGibbon2015MDTraj] has significantly improved the accessibility of MD results for researchers. 
Consequently, there is a high demand for a framework that facilitates batch processing of MD trajectories and dynamic feature extraction.


<!-- ######### -->
<!-- Main body -->
# Key features

<!-- Dynamic features extraction -->
Nearl converts MD trajectories into 3D grid representations encoded by the following dynamic features.
Marching OBServers (MOBS) views each grid point as an observer and calculate the local observable along the time axis. With each observer being able to perceive its local environment, the marching observers are able to capture the local atomic motion patterns but maintain the global configuration. 
Property-Density Flow (PDF) which voxelizes the atoms into 3D grids for individual frames and aggregates the properties along the time axis for the save grid point. 
The most expensive grid-based computations, such as Gaussian interpolation, observable computation and time-series aggregation, are accelerated by CUDA.
Apart from the two dynamic features, voxelization of static structure (part of PDF method) is also implemented in Nearl with various pre-implemented atomic properties. Further details can be viewed in the [documentation](http://nearl.readthedocs.io/).


<!-- encoding the geometric, chemical/physical -->
<!-- 
grid-major order by calculating local ovservables along a trajectory slice
Property-density flow: atom-major order by voxelizing into 3D grids across a slice of frames.  
-->

<!-- Design principles -->
Inspired by the polymorphic design of the [PyTorch](https://pytorch.org/)[@paszke2019pytorch], Nearl abstracts five steps during 3D feature generation including 1. featurizer setting perception; 2. property caching (one-time); 3. point query; 4. feature computation; and 5. result dumping. 
The pipeline is designed to be modular and flexible, allowing for full control of the customizability of features. 
With this design, the user can register an arbitrary number of desired features and the corresponding label-like features, and easily prepare the training data and their corresponding labels for the 3D-CNN models. 


<!-- Input supports --> 
<!-- Nearl treats all input structures as trajectories, seamlessly accommodating both dynamic (MD simulations) and static (PDB files) data. 
Since all of the input structures are viewed as a trajectory, static PDB files are implicitly treated as a trajectory with a single frame. 
Nearl supports commonly used trajectory formats including PDB, NetCDF and DCD. 
The trajectory module also follows the polymorphic design enabling users to define their dedicated trajectory supplier. -->
Nearl accommodates a variety of trajectory formats commonly used in MD simulations, including PDB, NetCDF, and DCD. This flexibility ensures compatibility with a wide range of workflows. Notably, Nearl can handle static PDB files by treating them implicitly as single-frame trajectories. Furthermore, the pipeline adheres to a polymorphic design for the trajectory module. This empowers users to define their own custom trajectory suppliers, catering to specific needs.
<!-- Output supports -->
Processed results are by default stored in HDF5 files, leveraging this format's high-performance data access capabilities. Users can readily switch to alternative storage formats by modifying the dump method within the features module. Notably, the HDF dataset supplier employs process pools during model training, enabling concurrent access to data across multiple HDF files for enhanced efficiency.
<!-- Model supports -->
For seamless integration with deep learning frameworks, Nearl implements several pre-built 3D convolutional neural network (3D-CNN) models within the PyTorch framework. These models are adapted from established architectures [@mcnutt2021gnina; @stepniewska2018development; @crocioni2024deeprank2; @townshend2020atom3d]. 

<!--
and their pretrained weights for the scoring of ligands via a well documented commandline interface.   
-->

<!-- Nearl provides a fast and efficient tool to extract dynamic features from MD trajectories to  -->
This project aims to bridge the current gap between deep learning applications and MD simulations and enhance the reusability of MD trajectories, thereby unlocking their full potential for informing and accelerating biological research. 
Nearl's design prioritizes user customization and extensibility which allows researchers to seamlessly integrate their own features and models, fostering broader applicability.
We anticipate Nearl will be a valuable tool for computational biologists, bioinformaticians, and machine learning engineers. These researchers can leverage Nearl to incorporate crucial protein dynamics information into diverse areas of investigation, including drug discovery, molecular design, and structural biology studies.

<!-- 1047 words: a rough word count -->

# Acknowledgements
This work is supported by the Swiss National Science Foundation (SNSF) grant xxxxxxxx. 


# References

