import time
import pickle 
import json
import numpy as np 
import pytraj as pt 
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy

from BetaPose import utils, cluster

# Forcefield and xml related functions
import re
from xml.dom import minidom
import xml.etree.ElementTree as ET

"""
Rethink: If it is necessary to separate the components into different channels and how 
For the prototype, use the plain version. 
"""


class featurizer_3d:
  def __init__(self, pdbfile, trajfile, parmdic):
    
    print("Initializing the featurizer object......")
    self.pdbfile = pdbfile; 
    self.trajfile = trajfile; 
    if isinstance(parmdic, dict):
      # Loading a dictionary 
      self.parmdic = parmdic; 
      self._init_settings(); 
    elif isinstance(parmdic, str):
      # Loading from a json file
      self.load_settings(parmdic); 
    
    # TODO: new func
    ####################################################################################################
    ######################## Cluster the trajectory based on pairwise distance #########################
    ####################################################################################################
    ligmask = self.normmask(self.parmdic["MASK_LIG"]); 
    traj2 = pt.load(self.trajfile, top=self.pdbfile, stride=1); 
    traj2.top.set_reference(traj2[0]); 
    self.frame_total = len(traj2.xyz);
    pdist, y = utils.PairwiseDistance(traj2, f"{ligmask}&!@H=", f"{ligmask}<@6&!{ligmask}&@C,CA,CB,N,O",use_mean=True);
    clusters = cluster.ClusterAgglomerative(pdist, 10)
    self.frames = cluster_rand = cluster.RandomPerCluster(clusters, number=1)
    
    print(f"Total frame Number: {self.frame_total}"); 
    
    print("The following frames are selected", cluster_rand)
    
    
    ####################################################################################################
    ####################################### Load the trajectory ########################################
    ####################################################################################################
    self.traj = pt.load(self.trajfile, top=self.pdbfile, frame_indices=cluster_rand); 
    self.traj.top.set_reference(self.traj[0])
    self.frameNr = len(self.traj.xyz); 
    self.reslist = [i.name for i in self.traj.top.residues];     
    print(f"Frame Nr: {self.frameNr}")
    self.updatesearchlist()
    self.coordinates = {}
    # Initialize the groups 
    for group in self.atom_groups:
      print(group)
#       self.init_group(i, atom_groups[i])
#############
#############
#     if 'ligand' in atom_groups.keys():
#       com0 = pt.center_of_mass(self.trajpdb, atom_groups["ligand"], frame_indices=[0]).squeeze(); 
#       print(f"Using ligand to align 3D curve {np.round(com0,2)}")
#       self.alignBy = "ligand"
#     else: 
#       com0 = pt.center_of_mass(self.trajpdb, atom_groups[atom_groups.keys()[0]], frame_indices=[0]).squeeze();
#       print(f"Using {atom_groups.keys()[0]} to align 3D curve {com0}")
#       self.alignBy = atom_groups.keys()[0]
#     self.alignCenter(com0)
#     self.curveCenter = np.mean(self.points3D, axis = 0).reshape(1,3); 

    
  def normmask(self, mask):
    tmptraj = pt.load(self.pdbfile, top=self.pdbfile); 
    idxs = tmptraj.select(mask)
    retmask = "@"
    for i in idxs:
      retmask += f"{i+1},"
    retmask = retmask.strip(",")
    print(retmask, tmptraj.select(retmask), tmptraj.select(mask))
    return retmask
    
    
  def _init_settings(self):
    """
    Initialize important parameters/settings
    """
    print(self.parmdic)
    pointnr = 16
    self.parms = {
      "lattice_nr"  : pointnr,  # unit points 
      "lattice_len" : 1,        # "length3D"  : defpointnr 
      "shift" : np.array([0,0,0]),
      "atom_groups" : [],
      "atmpro" : np.array([]), 
      "atmlig" : np.array([]), 
      "atmsol" : np.array([]), 
      "stride" : 1,
      "search_cutoff" : 18
    }
    self.updatecell(); 
    
    tmptop = pt.load(self.pdbfile)
    tmptop.top.set_reference(tmptop[0])
    self.parms["atmpro"] = utils.GetProteinByTraj(tmptop); 
    self.parms["atmlig"] = tmptop.top.select(":LIG"); 
    prot_lig_idxlst = (self.parms["atmpro"]+1).astype(str).tolist() + (self.parms["atmlig"]+1).astype(str).tolist(); 
    pro_lig_mask = "@"+",".join(prot_lig_idxlst);
    self.parms["atmsol"] = tmptop.top.select("!"+pro_lig_mask); 
    if len(self.parms["atmpro"]) > 0:
      self.parms['atom_groups'].append("protein"); 
    if len(self.parms["atmlig"]) > 0:
      self.parms['atom_groups'].append("ligand"); 
    if len(self.parms["atmsol"]) > 0:
      self.parms['atom_groups'].append("solvent"); 
    print(self.parms)
    
    parmkeys = self.parmdic.keys(); 
    if ("MASK_PRO" in parmkeys): 
      self.parms["atmpro"] = tmptop.top.select(self.parmdic["MASK_PRO"]); 
      if "protein" not in self.parms['atom_groups']:
        self.parms['atom_groups'].append("protein"); 
    if ("MASK_LIG" in parmkeys): 
      self.parms["atmlig"] = tmptop.top.select(self.parmdic["MASK_LIG"]); 
      if "ligand" not in self.parms['atom_groups']:
        self.parms['atom_groups'].append("ligand"); 
    if ("MASL_SOL" in parmkeys): 
      self.parms["atmsol"] = tmptop.top.select(self.parmdic["MASL_SOL"]); 
      if "solvent" not in self.parms['atom_groups']:
        self.parms['atom_groups'].append("solvent"); 

    """ 
      Could put this to a separate function 
      Basic numbers : lattice_nr, lattice_len Could only be defined by user
      Ways to shift (center) / scale a lattice
    """ 
    
    if "LATTICE_POINTS" in parmkeys: 
      self.parms["lattice_nr"] = self.parmdic["LATTICE_POINTS"]; 
      self.updatecell();       
      
    if "CELL_LENGTH" in parmkeys:
      self.parms["lattice_len"] = self.parmdic["CELL_LENGTH"] / (self.parms["lattice_nr"]-1);
      print("Settting the lattice_length", self.parms["lattice_len"]); 
      self.updatecell(); 
      
    # Either use center or shift a cell 
    # Firstly set the shift 
    # Set the shift back to array [0,0,0] after updating the cell.
    if "CENTERMASK" in parmkeys:
      thecenter = pt.center_of_mass(tmptop, mask=self.parmdic["CENTERMASK"])[0];
      self.parms["shift"] = thecenter - np.mean(self.parms["points3D"], axis=0); 
      self.updatecell(); 
      self.parms["shift"] = np.array([0,0,0])
    elif "CENTER" in parmkeys:
      self.parms["shift"] = np.array(self.parmdic["CENTER"]) - np.mean(self.parms["points3D"], axis=0); 
      self.updatecell(); 
      self.parms["shift"] = np.array([0,0,0])
    elif "CELL_SHIFT" in parmkeys:
      self.parms["shift"] = self.parmdic["CELL_SHIFT"]; 
      self.updatecell(); 
      self.parms["shift"] = np.array([0,0,0])
      
    """ Trajectory loading the shifting """
    if "STRIDE" in parmkeys:
      self.parms["stride"] = self.parmdic["STRIDE"]; 
    if "CUTOFF" in parmkeys:
      self.parms["search_cutoff"] = self.parmdic["CUTOFF"]; 
    
    # Avoid direct modification of key variables. 
    self._load_parms(); 
    
    """ Features to compute """
    self.features = list(self.parmdic["DESCRIPTORS"].keys()); 
    if "element" in [i.lower() for i in self.features]: 
      self.ELEMENTPARMS = self.parmdic["DESCRIPTORS"]["ELEMENT"]
    if "charge" in [i.lower() for i in self.features]: 
      self.FF_PRO = self.parmdic["FF_PRO"]; 
      self.FF_SOL = self.parmdic["FF_SOL"]; 
      self.FF_LIG = self.parmdic["FF_LIG"]; 
      self.CHARGEPARMS = self.parmdic["DESCRIPTORS"]["CHARGE"]
      
    if "entropy" in [i.lower() for i in self.features]: 
      self.ENTROPYPARMS = self.parmdic["DESCRIPTORS"]["ENTROPY"]
    

  def load_settings(self, file):
    """
      Directly load the featurizer from a json file 
      Not the focus of current development as well as the self._load_parms
    """
    with open(file, "r") as file1: 
      self.parms = json.load(file1);
    self._load_parms(); 
    
  def _load_parms(self):
    """
      Load the input parameters from the dictionary , Avoid direct assignment 
    """
    self.lattice_number = self.parms["lattice_nr"]; 
    self.lattice_length = self.parms["lattice_len"]; 
    self.cell_length = self.parms["cell_len"]; 
    self.index3D = self.parms["index3D"]; 
    self.points3D = self.parms["points3D"]; 
    self.distances = self.parms["distances"]; 
    self.cellcenter = self.parms["cellcenter"]; 
    self.atom_groups = self.parms['atom_groups']; 
    self.stride = self.parms['stride']; 
    self.search_cutoff = self.parms['search_cutoff']; 
    
  def updatecell(self):
    print("Setting the box")
    self.parms["cell_len"]   = self.parms["lattice_len"] * self.parms["lattice_nr"]; 
    self.parms["index3D"]    = self.get_points(self.parms["lattice_nr"]);  
    self.parms["points3D"]   = self.get_points(self.parms["lattice_nr"]) * self.parms["lattice_len"]; 
    self.parms["points3D"]   += self.parms["shift"]; 
    self.parms["distances"]  = np.array(range(self.parms["lattice_nr"]**3)); 
    self.parms["cellcenter"] = np.mean(self.parms["points3D"], axis = 0).reshape(1,3); 
    
  def updatesearchlist(self):
    # TODO; 
    ligmask = self.parmdic["MASK_LIG"]; 
    ligcenter = pt.center_of_mass(self.traj, ligmask); 
    
    len_diagonal = self.parms["cell_len"] * np.sqrt(3) + 3; 
    print("ligand center", ligcenter[0], len_diagonal, np.sqrt(3)); 
    self.parms["searchlist"] = self.searchlist = self.traj.top.select(f"{ligmask}<@{len_diagonal/2}"); 
    
    
  def get_point_by_distance(self, point, length):
    d0 = int(point/length**2)
    d1 = int((point - d0*length**2)/length)
    d3 = int(point - d0*length**2 - d1*length)
    return [d0, d1, d3]
  
  def get_points(self, length):
    x=[]; 
    for i in range(length**3):
      x.append(self.get_point_by_distance(i,length))
    return np.array(x).astype(int)

  def points_to_3D(self, thearray, dtype=float):
    if len(self.distances) != len(thearray):
      print("Cannot match the length of the array to the 3D cuboid"); 
      return np.array([0])
    tempalte  = np.zeros((self.lattice_number, self.lattice_number, self.lattice_number)).astype(dtype);
    for ind in self.distances:
      array_3Didx = tuple(self.index3D[ind]); 
      tempalte[array_3Didx] = thearray[ind]
    return tempalte

  def NormalizeMass(self, array, parm = 9, x0=7, slope=0.015):
    line1 = 1/(1+np.e**(-array+x0))
    baseNr = 1/(1+np.e**(-parm+x0))
    line2 = baseNr + (array-parm)*slope
    status1 = array <= parm
    status2 = array > parm
    template = np.zeros(array.shape)
    template[status1] = line1[status1]
    template[status2] = line2[status2]
    return template

  def get_entropy(self, arr):
    unique, counts = np.unique(arr, return_counts=True)
    return entropy(counts)

  def init_group(self, groupname, mask):
#       "coorpro" : np.array([]), 
#       "coorlig" : np.array([]), 
#       "coorsol" : np.array([]), 
    
    atom_sel = self.trajpdb.top.select(mask);
    if len(atom_sel) == 0: 
      print(f"Warning: There is no atom selected in the group {groupname}, skipping......"); 
      return 
    else: 
      print(f"Group Name: {groupname}; Atoms: {len(atom_sel)} ")
      self.selections[groupname]  = self.trajpdb.top.select(mask);
      self.coordinates[groupname] = self.trajpdb.xyz[0][self.selections[groupname]]

  def featurize_element(self):
    st_elm = time.perf_counter(); 
    print("Featurizing element"); 
    pdb_atomic_names  = np.array([i.name for i in self.traj.top.atoms]).astype(str); 
    pdb_atomic_numbers = np.array([i.atomic_number for i in self.traj.top.atoms]).astype(int); 
    pdb_residue_names = np.array([self.reslist[i.resid] for i in self.traj.top.atoms]).astype(str); 
    
    # Initialize the container of the descriptors
    # 1. Sequentially process each selected frames.
    self.atom_name  = []; 
    self.res_name   = []; 
    self.atom_mass = []; 
    # self.norm_mass = []; 
    # self.gauss_mass = []; 
    for i in range(len(self.frames)):
      thisxyz = self.traj.xyz[i]; 
      # Thirdly: Extract coordinates within the cutoff, atom index and
      selxyz = thisxyz[self.searchlist]; 
      # Fourthly: restrain real candidates
      cand_status = distance_matrix(selxyz, self.cellcenter) <= self.search_cutoff; 
      cand_status = cand_status.squeeze(); 
      cand_index  = self.searchlist[cand_status]; 
      cand_xyz    = selxyz[cand_status]; 
      cand_distmatrix = distance_matrix(self.points3D, cand_xyz)
      cand_diststatus = cand_distmatrix < 1.75
      # cand_distmatrix < 3.75

      mins = np.min(cand_distmatrix, axis=1)
      idx_lst = [np.where(cand_distmatrix[m] == mins[m])[0][0] if np.any(cand_diststatus[m,:]) else -1 for m in range(len(mins))]
      candlst = [cand_index[m] if m>=0 else -1 for m in idx_lst]

      # Atomic name-based; 
      atom_name_frameN = [pdb_atomic_names[m]  if m>0 else False for m in candlst]; 
      atom_name_frameN = self.points_to_3D(atom_name_frameN, dtype=str); 
      self.atom_name.append(atom_name_frameN)
      
      # Residue name-based; 
      res_name_frameN  = [pdb_residue_names[m] if m>0 else False for m in candlst]; 
      res_name_frameN  = self.points_to_3D(res_name_frameN, dtype=str); 
      self.res_name.append(res_name_frameN)
      
      # Atomic mass-based; 
      atom_mass_frameN = [pdb_atomic_numbers[m] if m>0 else 0 for m in candlst]; 
      atom_mass_frameN = self.points_to_3D(atom_mass_frameN); 
      self.atom_mass.append(atom_mass_frameN)
    
    for i in self.atom_mass:
      print(i.shape)
    # Convert the list to a numpy array. 
    self.atom_name = np.array(self.atom_name).astype(str);
    self.res_name  = np.array(self.res_name).astype(str);
    self.atom_mass = np.array(self.atom_mass); 
    # self.norm_mass = np.array(self.norm_mass);
    # self.gauss_mass= np.array(self.gauss_mass);
    
    loadtime = time.perf_counter() - st_elm;
    print(f"Element: processed {self.frameNr} frames; Time Total: {loadtime:.2f} seconds; Time Avg: {loadtime/self.frameNr:.2f};")

  def featurize_charge(self):
#     self.features.append(feature)
    print(f"FF protein, {self.FF_PRO}, FF solvent: {self.FF_SOL}, FF ligand: {self.FF_LIG}")
    print("Reading forcefield files")
    reader = ffreader(self.FF_PRO)
    waitlist = list(set([i.upper() for i in self.reslist])) + list(reader.residuemap.keys())
    reader.addFF(self.FF_LIG, waitlist=waitlist)
    reader.addFF(self.FF_SOL, waitlist=waitlist)

    st_chg = time.perf_counter(); 
    self.atom_charge  = {}; 
    self.gauss_charge = {}; 
    for sel in self.selections.keys():
      theshape = self.atom_name[sel].shape
      chargearr = np.zeros(theshape)
      tmpgrp_atom_charge  = []; 
      tmpgrp_gauss_charge = []; 
      for fnr in range(len(self.atom_name[sel])):
        atomnamearr = [self.atom_name[sel][fnr][tuple(self.index3D[i])] for i in self.distances];
        resnamearr  = [self.res_name[sel][fnr][tuple(self.index3D[i])]  for i in self.distances];
        print(f"dealing with the selection {sel}",resnamearr)
        chargearr   = [reader.getAtomCharge(i, j) if (i != "False" and i != False) else 0 for i,j in zip(resnamearr, atomnamearr)]
        chargearr   = self.points_to_3D(chargearr)
        tmpgrp_atom_charge.append(chargearr); 
        tmpgrp_gauss_charge.append(gaussian_filter(chargearr, sigma=1)); 
      self.atom_charge[sel] = np.array(tmpgrp_atom_charge)
      self.gauss_charge[sel] = np.array(tmpgrp_gauss_charge)
      print(self.atom_charge[sel])
    loadtime = time.perf_counter() - st_chg;
    print(f"Charge: featurized {self.frameNr} frames, took {loadtime:.2f} seconds; Avg: {loadtime/self.frameNr:.2f}")
  
  def featurize_entropy(self):
    st_etp = time.perf_counter(); 
    # Local context: get the average entropy within a period of time. 
    print("Featurizing entropy")
    # TODO: set a proper cutoff to determine very little occupied cells.
    
    if "THRESHOLD" in self.ENTROPYPARMS.keys() and isinstance(self.ENTROPYPARMS["THRESHOLD"], float):
      occupancy_threshold = self.ENTROPYPARMS["THRESHOLD"]
    else:
      occupancy_threshold = 0.0;
    print("Occupancy_threshold", occupancy_threshold)
    
    entropy_values = np.zeros((self.lattice_number, self.lattice_number, self.lattice_number));
    idx_template = [[] for i in self.distances]; 

    for frame in self.frames:
      # Thirdly: Extract coordinates within the cutoff, atom index and
      # Reload the corresponding period of time. 
      
      frameed = (frame+10 if frame+10 < self.frame_total else self.frame_total)
      framest = (frame-10 if frame-10 > 0 else 0)
      indices = range(framest, frameed, 1); 
      print(f"Frame: {frame}; Using indices: {indices}; "); 
      tmptraj = pt.load(self.trajfile, top=self.pdbfile, frame_indices=indices); 
      tmptraj.top.set_reference(tmptraj[0]); 
      
      thisxyz = self.traj.xyz[i];
#       self.traj.top.set_reference(self.traj[i]); 
      selidx = self.traj.top.select(f":LIG<@{self.search_cutoff}"); 
      selxyz = thisxyz[selidx]; 

      # Get atoms within the box formed by box's smallest point and greatest point
      sel_distmatrix_max = distance_matrix(self.points3D + self.cell_length/2, selxyz)
      sel_distmatrix_min = distance_matrix(self.points3D - self.cell_length/2, selxyz) 
      sel_status_max = sel_distmatrix_max < np.sqrt(3)*self.cell_length; 
      sel_status_min = sel_distmatrix_min < np.sqrt(3)*self.cell_length; 
      summary = sel_status_max * sel_status_min; 

      # Interate through all of the grid points
      # Set pre-exit and add zero just to make sure the list value is greater than frame number
      for p in range(len(self.distances)):
        Nratoms = np.count_nonzero(summary[p])
        if Nratoms >0:
          pointp = self.points3D[p]; 
          upper = pointp+self.cell_length/2
          lower = pointp-self.cell_length/2
          sel_ndxs  = selidx[np.where(summary[p] == True)[0]]
          sel_points = thisxyz[sel_ndxs]
          up_status = upper - sel_points > 0
          lw_status = sel_points - lower > 0
          ov_status = np.all(up_status*lw_status, axis=1)
          if True not in ov_status:
            idx_template[p].append(0)
            continue
          for s, tmpidx in zip(ov_status, sel_ndxs):
            if s == True:
              idx_template[p].append(tmpidx)
        else:
          idx_template[p].append(0)
    # If there is only one value in the list, the entropy will be 0 
    # Hence, there will be a 0 when initialize the list 
    entropy_arr = [self.get_entropy(_) if len(set(_)) > occupancy_threshold*self.frameNr else 0 for _ in idx_template]
    self.entropy = self.points_to_3D(entropy_arr)
    self.gauss_entropy = gaussian_filter(self.entropy, sigma=1)
    time_etp = time.perf_counter() - st_etp;
    print(f"Entropy: featurized {self.frameNr} frames, took {time_etp:.2f} seconds; Avg: {time_etp/self.frameNr:.2f}")

    print(f"Frame Number: {self.frameNr}, occupancy threshold {occupancy_threshold}")
    print(f"The averaged entropy is {np.mean(self.entropy):.2f}, Gaussian filtered entropy is {np.mean(self.gauss_entropy):.2f}")
    print(f"The max entropy is {np.max(self.entropy):.2f}, Gaussian filtered entropy is {np.max(self.gauss_entropy):.2f}")
    print(f"The min entropy is {np.min(self.entropy):.2f}, Gaussian filtered entropy is {np.min(self.gauss_entropy):.2f}")
    print(f"The standard deviation of entropy is {np.std(self.entropy):.2f}, Gaussian filtered entropy is {np.std(self.gauss_entropy):.2f}")

  def featurize_hydrophb(self):
    pass
  
  def featurize(self, settings={}):
    features = [i.lower() for i in self.features]
    for feature in features:
      if feature == 'element':
        self.featurize_element(); 
      elif feature == 'entropy':
        self.featurize_entropy(); 
      elif feature == 'charge':
        self.featurize_charge(); 
      else: 
        print(f"Decriptor {feature} is not a standard descriptor yet. ")

  def alignCenter(self, refCenter):
    diff = np.array(refCenter) - np.mean(self.points3D, axis=0); 
    self.points3D = self.points3D + diff; 
    
  def shift(self, shift):
    self.points3D = self.points3D + np.array(shift)
    
  def scaleToLength(self, refLength):
    scaleFactor = refLength / self.length3D;
    self.cell_length = self.cell_length * scaleFactor; 
    diff = self.points3D - self.index3D
    self.points3D = diff + self.index3D * scaleFactor; 
    
  def scaleByFactor(self, scaleFactor):
    self.cell_length = self.cell_length * scaleFactor; 
    self.points3D = self.points3D * scaleFactor; 

  def save(self, filename):
    with open(filename, "wb") as tmpfile:
      data_to_save={
        "frameNr": self.frameNr,
        "frameList": self.frameList,
        "atom_groups": self.atom_groups, 
        "distances": self.distances, 
        "length3D": self.length3D,
        "index3D": self.index3D,
        "points3D": self.points3D,
        "features": self.features,
        "atomic_names": self.pdb_atomic_names,
        "atomic_number": self.pdb_atomic_numbers,
       }
      if "element" in self.features:
        data_to_save["atom_mass"] = self.atom_mass
        data_to_save["norm_mass"] = self.norm_mass
        data_to_save["gauss_mass"] = self.gauss_mass
      if "entropy" in self.features:
        data_to_save["entropy"] = self.entropy; 
        data_to_save["gauss_entropy"]=self.gauss_entropy; 
      if "charge" in self.features:
        data_to_save["atom_charge"] = self.atom_charge
        data_to_save["gauss_charge"] = self.gauss_charge
      pickle.dump(data_to_save ,tmpfile, protocol=pickle.HIGHEST_PROTOCOL)

            
# ATOM_GROUPS = {"protein":":1-221", "ligand":":LIG", "solvent":":T3P,CL-,K+"}

FEATURIZER_SETTINGS = {
  # Mask of components 
  "MASK_PRO" : ":1-221",
  "MASK_LIG" : ":LIG", 
  "MASL_SOL" : ":T3P,CL-,K+", 
  
  # FF of components
  "FF_PRO" : "./Forcefield/charmm36_nowaters.xml",
  "FF_SOL" : "./Forcefield/test_wat.xml",
  "FF_LIG"  : "./tests/featurizer_test1_lig.xml",
  
  # POCKET SETTINGS
  "CELL_LENGTH" : 10,       # Unit: Angstorm (Need scaling)
  "LATTICE_POINTS" : 16,     # Unit: 1 (Number of lattice in one dimension)
  "CELL_SHIFT" : [1,2,3],   # Either CENTERMASK, CENTER or CELL_SHIFT (mask>center>shift) 
#   "CENTER" : [6,6,6], 
  "CENTERMASK" : ":LIG",
  
  # IO SETTINGS
  "STRIDE": 1,             # Unit: frames  
  
  # SEARCH SETTINGS
  "CUTOFF": 18, 
  
  # DESCRIPTOR SETTINGS
  "DESCRIPTORS"  : {
    "ELEMENT":{"ACTIVE":True, }, 
#     "CHARGE":{"ACTIVE":True, }, 
    "ENTROPY":{"ACTIVE":True, "THRESHOLD": 0.05, "INTERVAL": 8},
  }, 
}


amberxml = "/home/miemie/Dropbox/Documents/BetaPose/Forcefield/ff14SB.xml"
charmmxml = "/home/miemie/Dropbox/Documents/BetaPose/Forcefield/charmm36_nowaters.xml"


topfile="/home/miemie/Dropbox/Documents/BetaPose/tests/featurizer_test1.pdb"; 
trajfile="/home/miemie/Dropbox/Documents/BetaPose/tests/featurizer_test1.nc"; 


featurizer = featurizer_3d(topfile, trajfile, FEATURIZER_SETTINGS); 

# featurizer.scaleToLength(18); 
featurizer.featurize()

# featurizer.save("/tmp/test_featurizer_3D.pkl")


# print(featurizer.index3D)
# print(featurizer.distances)
# print(featurizer.points3D)


import time, builtins, tempfile, datetime, os, functools, copy
import numpy as np
import pytraj as pt

# other modules
from scipy.stats import entropy
# from scipy.ndimage import gaussian_filter
# from scipy.spatial import distance_matrix
from matplotlib import cm
import matplotlib.pyplot as plt

# local modules
from . import representations, data_io
from . import CONFIG, printit

_clear = CONFIG.get("clear", False);
_verbose = CONFIG.get("verbose", False);

"""
Configurations needed for this module
CONFIG["clear"]
CONFIG["verbose"]
"""


def cgenff_reader(filename):
  with open(filename) as file1:
    lst = list(filter(lambda i: re.match(r"^ATOM.*!", i), file1))
  theatom = [i.strip("\n").split()[1] for i in lst]
  atomtype = [i.strip("\n").split()[2] for i in lst]
  charge = [float(i.strip("\n").split()[3]) for i in lst]
  penalty = [float(i.strip("\n").split()[-1]) for i in lst]
  return {"name": theatom, "type": atomtype, "charge": charge, "penalty": penalty}


def lig_xml(dic, write_file=False, source=False):
  root = ET.Element('ForceField')
  info = ET.SubElement(root, 'Info')
  info_date = ET.SubElement(info, "date")
  info_date.text = str(date.today())
  if source != False:
    info_file = ET.SubElement(info, 'source')
    info_file.text = source

  data_lig = ET.SubElement(root, 'LIG')
  for i in range(len(dic["name"])):
    tmpattrib = {
      "name": dic["name"][i],
      "type": dic["type"][i],
      "charge": str(dic["charge"][i]),
      'penalty': str(dic["penalty"][i]),
    }
    tmpatom = ET.SubElement(data_lig, 'ATOM', attrib=tmpattrib)

  ligxml_str = ET.tostring(root, encoding="unicode")
  dom = minidom.parseString(ligxml_str)
  ligxml_str = dom.toprettyxml()

  if write_file != False:
    with open(write_file, "w") as file1:
      file1.write(ligxml_str)
  return ligxml_str


class obsolete_featurizer_3d:
  def __init__(self, pdbFile, trajFile, grid_length, atom_groups, search_cutoff=18, stride=1):
    # Initialize the featurizer object
    print("Initializing the featurizer object......")
    self.length3D = grid_length
    self.index3D = self.get_points(grid_length)
    self.distances = np.array(range(self.length3D ** 3));
    self.points3D = self.index3D
    self.atom_groups = atom_groups
    self.cell_length = 1;
    # Load trajectory
    self.pdbfile = pdbFile;
    self.trajpdb = pt.load(trajFile, top=pdbFile, stride=stride);
    self.frameNr = self.trajpdb.n_frames;
    self.frameList = np.arange(1, self.frameNr * stride + 1, stride);
    self.reslist = [i.name for i in self.trajpdb.top.residues]
    self.pdb_atomic_names = np.array([i.name for i in self.trajpdb.top.atoms]).astype(str)
    self.pdb_atomic_numbers = np.array([i.atomic_number for i in self.trajpdb.top.atoms]).astype(int)
    self.pdb_residue_names = np.array([self.reslist[i.resid] for i in self.trajpdb.top.atoms]).astype(str)
    print(self.pdb_residue_names)
    self.search_cutoff = search_cutoff;

    self.coordinates = {}
    self.selections = {}
    for i in atom_groups.keys():
      self.init_group(i, atom_groups[i])
    if 'ligand' in atom_groups.keys():
      com0 = pt.center_of_mass(self.trajpdb, atom_groups["ligand"], frame_indices=[0]).squeeze();
      print(f"Using ligand to align 3D curve {np.round(com0, 2)}")
      self.alignBy = "ligand"
    else:
      com0 = pt.center_of_mass(self.trajpdb, atom_groups[atom_groups.keys()[0]], frame_indices=[0]).squeeze();
      print(f"Using {atom_groups.keys()[0]} to align 3D curve {com0}")
      self.alignBy = atom_groups.keys()[0]
    self.alignCenter(com0)
    self.curveCenter = np.mean(self.points3D, axis=0).reshape(1, 3);

  def get_point_by_distance(self, point, length):
    # Get the 3D coordinate of a point in the 3D grid
    d0 = int(point / length ** 2)
    d1 = int((point - d0 * length ** 2) / length)
    d3 = int(point - d0 * length ** 2 - d1 * length)
    return [d0, d1, d3]

  def get_points(self, length):
    # Get the 3D coordinates of all points in the 3D grid
    x = []
    for i in range(length ** 3):
      x.append(self.get_point_by_distance(i, length))
    return np.array(x).astype(int)

  def points_to_3D(self, thearray, dtype=float):
    # Convert a 1D array to a 3D array
    if len(self.distances) != len(thearray):
      print("Cannot match the length of the array to the 3D cuboid");
      return np.array([0])
    tempalte = np.zeros((self.length3D, self.length3D, self.length3D)).astype(dtype);
    for ind in self.distances:
      array_3Didx = tuple(self.index3D[ind]);
      tempalte[array_3Didx] = thearray[ind]
    return tempalte

  def Norm_mass_array(self, array, parm=9, x0=7, slope=0.015):
    line1 = 1 / (1 + np.e ** (-array + x0))
    baseNr = 1 / (1 + np.e ** (-parm + x0))
    line2 = baseNr + (array - parm) * slope
    status1 = array <= parm
    status2 = array > parm
    template = np.zeros(array.shape)
    template[status1] = line1[status1]
    template[status2] = line2[status2]
    return template

  def get_entropy(self, arr):
    # The function to calculate the entropy of an array
    unique, counts = np.unique(arr, return_counts=True)
    return entropy(counts)

  def init_group(self, groupname, mask):
    atom_sel = self.trajpdb.top.select(mask);
    if len(atom_sel) == 0:
      print(f"Warning: There is no atom selected in the group {groupname}, skipping......");
      return
    else:
      print(f"Group Name: {groupname}; Atoms: {len(atom_sel)} ")
      self.selections[groupname] = self.trajpdb.top.select(mask);
      self.coordinates[groupname] = self.trajpdb.xyz[0][self.selections[groupname]]

  def featurize(self, features, settings={}):
    self.features = [];
    features = [i.lower() for i in features]
    for feature in features:
      if feature.lower() == 'element':
        print("Featurizing element")
        self.features.append(feature)
        st_elm = time.perf_counter();
        # Initialize the container of the descriptors
        self.atom_mass = {};
        self.norm_mass = {};
        self.gauss_mass = {};
        self.atom_name = {};
        self.res_name = {}

        # Firstly, Sequentially process each frames
        for sel in self.selections.keys():
          # Secondly, sequentially process each selection
          self.atom_mass[sel] = [];
          self.norm_mass[sel] = [];
          self.gauss_mass[sel] = [];
          self.atom_name[sel] = [];
          self.res_name[sel] = [];
          for i in range(len(self.trajpdb)):
            thisxyz = self.trajpdb.xyz[i];
            # Thirdly: Extract coordinates within the cutoff, atom index and
            selidx = self.selections[sel];
            selxyz = thisxyz[selidx];
            # Fourthly: restrain real candidates
            cand_status = distance_matrix(selxyz, self.curveCenter) <= self.search_cutoff;
            cand_status = cand_status.squeeze();
            cand_index = selidx[cand_status];
            cand_xyz = selxyz[cand_status];
            cand_distmatrix = distance_matrix(self.points3D, cand_xyz)
            cand_diststatus = cand_distmatrix < 1.75
            # cand_distmatrix < 3.75

            mins = np.min(cand_distmatrix, axis=1)
            idx_lst = [np.where(cand_distmatrix[m] == mins[m])[0][0] if np.any(cand_diststatus[m, :]) else -1 for m in
                       range(len(mins))]
            candlst = [cand_index[m] if m >= 0 else -1 for m in idx_lst]

            atom_name_frameN = [self.pdb_atomic_names[m] if m > 0 else False for m in candlst];
            res_name_frameN = [self.pdb_residue_names[m] if m > 0 else False for m in candlst];
            atom_name_frameN = self.points_to_3D(atom_name_frameN, dtype=str);
            res_name_frameN = self.points_to_3D(res_name_frameN, dtype=str);

            atom_mass_frameN = [self.pdb_atomic_numbers[m] if m > 0 else 0 for m in candlst];
            atom_mass_frameN = self.points_to_3D(atom_mass_frameN);
            norm_mass_frameN = self.Norm_mass_array(atom_mass_frameN)
            gauss_mass_frameN = gaussian_filter(norm_mass_frameN, sigma=1)

            self.atom_mass[sel].append(atom_mass_frameN)
            self.norm_mass[sel].append(norm_mass_frameN)
            self.gauss_mass[sel].append(gauss_mass_frameN)
            self.atom_name[sel].append(atom_name_frameN)
            self.res_name[sel].append(res_name_frameN)

          self.atom_mass[sel] = np.array(self.atom_mass[sel]);
          self.norm_mass[sel] = np.array(self.norm_mass[sel]);
          self.gauss_mass[sel] = np.array(self.gauss_mass[sel]);
          self.atom_name[sel] = np.array(self.atom_name[sel]).astype(str);
          self.res_name[sel] = np.array(self.res_name[sel]).astype(str);
        loadtime = time.perf_counter() - st_elm;
        print(
          f"Element: featurized {self.frameNr} frames, took {loadtime:.2f} seconds; Avg: {loadtime / self.frameNr:.2f}")

      elif feature.lower() == 'charge':
        self.features.append(feature)
        print("Reading forcefield files")
        reader = ffreader(settings["forcefield_ligand"])
        waitlist = list(set([i.upper() for i in self.reslist])) + list(reader.residuemap.keys())
        reader.addFF(settings["forcefield_protein"], waitlist=waitlist)
        reader.addFF(settings["forcefield_solvent"], waitlist=waitlist)

        st_chg = time.perf_counter();
        self.atom_charge = {};
        self.gauss_charge = {};
        for sel in self.selections.keys():
          theshape = self.atom_name[sel].shape
          chargearr = np.zeros(theshape)
          tmpgrp_atom_charge = [];
          tmpgrp_gauss_charge = [];
          for fnr in range(len(self.atom_name[sel])):
            atomnamearr = [self.atom_name[sel][fnr][tuple(self.index3D[i])] for i in self.distances];
            resnamearr = [self.res_name[sel][fnr][tuple(self.index3D[i])] for i in self.distances];
            print(f"dealing with the selection {sel}", resnamearr)
            chargearr = [reader.getAtomCharge(i, j) if (i != "False" and i != False) else 0 for i, j in
                         zip(resnamearr, atomnamearr)]
            chargearr = self.points_to_3D(chargearr)
            tmpgrp_atom_charge.append(chargearr);
            tmpgrp_gauss_charge.append(gaussian_filter(chargearr, sigma=1));
          self.atom_charge[sel] = np.array(tmpgrp_atom_charge)
          self.gauss_charge[sel] = np.array(tmpgrp_gauss_charge)
          print(self.atom_charge[sel])
        loadtime = time.perf_counter() - st_chg;
        print(
          f"Charge: featurized {self.frameNr} frames, took {loadtime:.2f} seconds; Avg: {loadtime / self.frameNr:.2f}")

      elif feature.lower() == 'entropy':
        self.features.append(feature)
        print("Featurizing entropy")
        # TODO: set a proper cutoff to determine very little occupied cells.
        if "entropy_threshold" in settings.keys():
          occupancy_threshold = settings["entropy_threshold"]
        else:
          occupancy_threshold = 0.0

        st_etp = time.perf_counter();
        entropy_values = np.zeros((self.length3D, self.length3D, self.length3D));
        idx_template = [[] for i in self.distances]

        for i in range(len(self.trajpdb)):
          # Thirdly: Extract coordinates within the cutoff, atom index and
          thisxyz = self.trajpdb.xyz[i];
          self.trajpdb.top.set_reference(self.trajpdb[i]);
          selidx = self.trajpdb.top.select(f":LIG<@{self.search_cutoff}");
          selxyz = thisxyz[selidx];

          sel_distmatrix_max = distance_matrix(self.points3D + self.cell_length / 2, selxyz)
          sel_distmatrix_min = distance_matrix(self.points3D - self.cell_length / 2, selxyz)
          sel_status_max = sel_distmatrix_max < np.sqrt(3) * self.cell_length;
          sel_status_min = sel_distmatrix_min < np.sqrt(3) * self.cell_length;
          summary = sel_status_max * sel_status_min;

          # Interate through all of the grid points
          # Set pre-exit and add zero just to make sure the list value is greater than frame number
          for p in range(len(self.distances)):
            Nratoms = np.count_nonzero(summary[p])
            if Nratoms > 0:
              pointp = self.points3D[p];
              upper = pointp + self.cell_length / 2
              lower = pointp - self.cell_length / 2
              sel_ndxs = selidx[np.where(summary[p] == True)[0]]
              sel_points = thisxyz[sel_ndxs]
              up_status = upper - sel_points > 0
              lw_status = sel_points - lower > 0
              ov_status = np.all(up_status * lw_status, axis=1)
              if True not in ov_status:
                idx_template[p].append(0)
                continue
              for s, tmpidx in zip(ov_status, sel_ndxs):
                if s == True:
                  idx_template[p].append(tmpidx)
            else:
              idx_template[p].append(0)
        # If there is only one value in the list, the entropy will be 0
        # Hence, there will be a 0 when initialize the list
        entropy_arr = [self.get_entropy(_) if len(set(_)) > occupancy_threshold * self.frameNr else 0 for _ in
                       idx_template]
        self.entropy = self.points_to_3D(entropy_arr)
        self.gauss_entropy = gaussian_filter(self.entropy, sigma=1)
        time_etp = time.perf_counter() - st_etp;
        print(
          f"Entropy: featurized {self.frameNr} frames, took {time_etp:.2f} seconds; Avg: {time_etp / self.frameNr:.2f}")

        print(f"Frame Number: {self.frameNr}, occupancy threshold {occupancy_threshold}")
        print(
          f"The averaged entropy is {np.mean(self.entropy):.2f}, Gaussian filtered entropy is {np.mean(self.gauss_entropy):.2f}")
        print(
          f"The max entropy is {np.max(self.entropy):.2f}, Gaussian filtered entropy is {np.max(self.gauss_entropy):.2f}")
        print(
          f"The min entropy is {np.min(self.entropy):.2f}, Gaussian filtered entropy is {np.min(self.gauss_entropy):.2f}")
        print(
          f"The standard deviation of entropy is {np.std(self.entropy):.2f}, Gaussian filtered entropy is {np.std(self.gauss_entropy):.2f}")
      else:
        print(f"Decriptor {feature} is not a standard descriptor yet. ")

  def alignCenter(self, refCenter):
    diff = np.array(refCenter) - np.mean(self.points3D, axis=0);
    self.points3D = self.points3D + diff

  def shift(self, shift):
    self.points3D = self.points3D + np.array(shift)

  def scaleToLength(self, refLength):
    scaleFactor = refLength / self.length3D;
    self.cell_length = self.cell_length * scaleFactor;
    diff = self.points3D - self.index3D
    self.points3D = diff + self.index3D * scaleFactor;

  def scaleByFactor(self, scaleFactor):
    self.cell_length = self.cell_length * scaleFactor;
    self.points3D = self.points3D * scaleFactor;

  def save(self, filename):
    with open(filename, "wb") as tmpfile:
      data_to_save = {
        "frameNr": self.frameNr,
        "frameList": self.frameList,
        "atom_groups": self.atom_groups,
        "distances": self.distances,
        "length3D": self.length3D,
        "index3D": self.index3D,
        "points3D": self.points3D,
        "features": self.features,
        "atomic_names": self.pdb_atomic_names,
        "atomic_number": self.pdb_atomic_numbers,
      }
      if "element" in self.features:
        data_to_save["atom_mass"] = self.atom_mass
        data_to_save["norm_mass"] = self.norm_mass
        data_to_save["gauss_mass"] = self.gauss_mass
      if "entropy" in self.features:
        data_to_save["entropy"] = self.entropy;
        data_to_save["gauss_entropy"] = self.gauss_entropy;
      if "charge" in self.features:
        data_to_save["atom_charge"] = self.atom_charge
        data_to_save["gauss_charge"] = self.gauss_charge
      pickle.dump(data_to_save, tmpfile, protocol=pickle.HIGHEST_PROTOCOL)


class obsolete_feature_3d_reader:
  def __init__(self, pickleFile):
    # Generate the 2D/3D hilbert curve
    with open(pickleFile, "rb") as file1:
      featuredic = pickle.load(file1)
      print(featuredic.keys());
      self.distances = featuredic["distances"];
      self.length3D = featuredic["length3D"];
      self.points3D = featuredic["points3D"];
      self.index3D = featuredic["index3D"];
      self.features = featuredic["features"];

      self.atom_groups = featuredic["atom_groups"];
      self.frameList = featuredic["frameList"]
      self.frameNr = featuredic["frameNr"]
      if "element" in self.features:
        self.atom_mass = featuredic["atom_mass"]
        self.norm_mass = featuredic["norm_mass"]
        self.gauss_mass = featuredic["gauss_mass"]
      if "entropy" in self.features:
        self.entropy = featuredic["entropy"]
        self.gauss_entropy = featuredic["gauss_entropy"]
      if "charge" in self.features:
        self.atom_charge = featuredic["atom_charge"];
        self.gauss_charge = featuredic["gauss_charge"];

  def selectData(self, maintype, select_group, subtype="gauss"):
    if maintype == "element":
      if subtype == "atom":
        data = self.atom_mass[select_group];
      elif subtype == "norm":
        data = self.norm_mass[select_group];
      elif subtype == "gauss":
        data = self.gauss_mass[select_group];
      else:
        print(f"Not Found the subtype {subtype}");
        data = self.gauss_mass[select_group];
    elif maintype == "entropy":
      if subtype == "gauss":
        data = self.gauss_entropy;
      else:
        data = self.entropy;
    elif maintype == "charge":
      if subtype == "atom":
        data = self.atom_charge[select_group];
      elif subtype == "gauss":
        data = self.gauss_charge[select_group];
      else:
        print(f"Not Found the subtype {subtype}");
        data = self.gauss_charge[select_group];
    return data

  def scatter3D(self, maintype, select_group, indice=0, subtype="gauss", cmap="Blues", threshold=0.1):
    thedata = self.selectData(maintype, select_group, subtype=subtype);
    if maintype == "element":
      thedata = thedata[indice]

    fig = plt.figure();
    ax = fig.add_subplot(projection='3d');
    plt.ion();

    thecmap = cm.get_cmap(cmap)
    print(thedata.shape)
    for i in self.distances:
      theindex = tuple(self.index3D[i]);
      theposition = self.points3D[i];
      v = thedata[theindex]
      thecolor = thecmap(v)
      # print(f"point: {i}, value: {v}, color: {thecolor}")
      if v > threshold:
        ax.scatter(*theposition, color=thecolor)

  def gen_pdbstr(self, coordinates, elements=None, bf=[]):
    if elements == None:
      elements = ["Du"] * len(coordinates)
    if len(coordinates) != len(bf):
      print("length not aligned")
    pdbline = ""
    tempstr = "ATOM      1  Du  TMP     1       0.000   0.000   0.000  1.00  0.00";
    coordinates = np.round(coordinates, decimals=3)
    for i in range(len(coordinates)):
      if len(bf) != 0:
        bfval = bf[i]
      else:
        bfval = 0.00
      point = coordinates[i];
      elem = elements[i];
      tmpstr = "".join(["{:>8}".format(i) for i in point]);
      tmpstr = "".join([f"{i:>8}" for i in point]);
      thisline = f"ATOM  {i:>5}  {elem:<3}{tempstr[16:30]}{tmpstr}{tempstr[54:60]}{round(bfval, 2):>6}\n"
      pdbline += thisline
    return pdbline

  def filter_coor(self, maintype, select_group, threshold, mode="gt", indice=0, subtype="gauss"):
    thedata = self.selectData(maintype, select_group, subtype=subtype);
    if maintype == "element" or maintype == "charge":
      thedata = thedata[indice]
    if mode == "gt":
      status = thedata > float(threshold)
    elif mode == "lt":
      status = thedata < float(threshold)
    elif mode == "mid":
      threshold1 = float(threshold.split(",")[0])
      threshold2 = float(threshold.split(",")[1])
      status1 = thedata >= threshold1
      status2 = thedata <= threshold2
      status = status1 * status2
    elif mode == "out":
      threshold1 = float(threshold.split(",")[0])
      threshold2 = float(threshold.split(",")[1])
      status1 = thedata < threshold1
      status2 = thedata > threshold2
      status = np.logical_or(status1, status2)
    filtered = thedata[status]
    print(f"there are {np.count_nonzero(status)} non-zero values")
    print(
      f"Filtered data: mean:{np.mean(filtered):2f}, Std:{np.std(filtered):2f}, Max:{np.max(filtered):2f}, Min:{np.min(filtered):2f}")
    coorlist = []
    for i in self.distances:
      theindex = tuple(self.index3D[i]);
      if status[theindex] == True:
        coorlist.append(self.points3D[i])
    return np.array(coorlist), filtered

  def write_pdb(self, pdblines, pdbfile="./test.pdb"):
    with open(pdbfile, "w") as file1:
      file1.write(pdblines)
    return pdbfile

########################################################







