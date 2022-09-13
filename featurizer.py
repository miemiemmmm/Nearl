from hilbertcurve.hilbertcurve import HilbertCurve
import os
import imageio
import numpy as np
import pandas as pd
import nglview as nv
import matplotlib.pyplot as plt


from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy

def cgenff_reader(filename):
  with open(filename) as file1:
    lst = list(filter(lambda i: re.match(r"^ATOM.*!", i), file1))
  theatom  = [i.strip("\n").split()[1] for i in lst]
  atomtype = [i.strip("\n").split()[2] for i in lst]
  charge   = [float(i.strip("\n").split()[3]) for i in lst]
  penalty  = [float(i.strip("\n").split()[-1]) for i in lst]
  return {"name":theatom, "type":atomtype, "charge":charge, "penalty":penalty}

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
    tmpattrib={
      "name":dic["name"][i], 
      "type": dic["type"][i], 
      "charge": str(dic["charge"][i]), 
      'penalty': str(dic["penalty"][i]),
    }
    tmpatom = ET.SubElement(data_lig, 'ATOM', attrib = tmpattrib)

  ligxml_str = ET.tostring(root , encoding="unicode")
  dom = minidom.parseString(ligxml_str)
  ligxml_str = dom.toprettyxml()

  if write_file != False :
    with open(write_file, "w") as file1: 
      file1.write(ligxml_str)
  return ligxml_str


class featurizer_3d:
  def __init__(self,pdbFile, trajFile, grid_length, atom_groups, search_cutoff=18, stride=1):
    print("Initializing the featurizer object......")
    self.length3D = grid_length    
    self.index3D = self.get_points(grid_length)
    self.distances = np.array(range(self.length3D**3));
    self.points3D = self.index3D
    self.atom_groups = atom_groups
    self.cell_length = 1; 
    # Load trajectory
    self.pdbfile = pdbFile; 
    self.trajpdb = pt.load(trajFile, top=pdbFile, stride=stride); 
    self.frameNr = self.trajpdb.n_frames; 
    self.frameList = np.arange(1, self.frameNr*stride+1, stride); 
    self.reslist = [i.name for i in self.trajpdb.top.residues]
    self.pdb_atomic_names  = np.array([i.name for i in self.trajpdb.top.atoms]).astype(str)
    self.pdb_atomic_numbers = np.array([i.atomic_number for i in self.trajpdb.top.atoms]).astype(int)        
    self.pdb_residue_names = np.array([self.reslist[i.resid] for i in self.trajpdb.top.atoms]).astype(str)
    print(self.pdb_residue_names)
    self.search_cutoff = search_cutoff; 

    self.coordinates = {}
    self.selections   = {}
    for i in atom_groups.keys():
      self.init_group(i, atom_groups[i])
    if 'ligand' in atom_groups.keys():
      com0 = pt.center_of_mass(self.trajpdb, atom_groups["ligand"], frame_indices=[0]).squeeze(); 
      print(f"Using ligand to align 3D curve {np.round(com0,2)}")
      self.alignBy = "ligand"
    else: 
      com0 = pt.center_of_mass(self.trajpdb, atom_groups[atom_groups.keys()[0]], frame_indices=[0]).squeeze();
      print(f"Using {atom_groups.keys()[0]} to align 3D curve {com0}")
      self.alignBy = atom_groups.keys()[0]
    self.alignCenter(com0)
    self.curveCenter = np.mean(self.points3D, axis = 0).reshape(1,3); 

  def get_point_by_distance(self, point, length):
    d0 = int(point/length**2)
    d1 = int((point - d0*length**2)/length)
    d3 = int(point - d0*length**2 - d1*length)
    return [d0, d1, d3]
  
  def get_points(self, length):
    x=[]
    for i in range(length**3):
      x.append(self.get_point_by_distance(i,length))
    return np.array(x).astype(int)

  def points_to_3D(self, thearray, dtype=float):
    if len(self.distances) != len(thearray):
      print("Cannot match the length of the array to the 3D cuboid"); 
      return np.array([0])
    tempalte  = np.zeros((self.length3D, self.length3D, self.length3D)).astype(dtype);
    for ind in self.distances:
      array_3Didx = tuple(self.index3D[ind]); 
      tempalte[array_3Didx] = thearray[ind]
    return tempalte

  def Norm_mass_array(self, array, parm = 9, x0=7, slope=0.015):
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
    atom_sel = self.trajpdb.top.select(mask);
    if len(atom_sel) == 0: 
      print(f"Warning: There is no atom selected in the group {groupname}, skipping......"); 
      return 
    else: 
      print(f"Group Name: {groupname}; Atoms: {len(atom_sel)} ")
      self.selections[groupname]  = self.trajpdb.top.select(mask);
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
        self.atom_mass  = {}; 
        self.norm_mass  = {}; 
        self.gauss_mass = {}; 
        self.atom_name  = {}; 
        self.res_name   = {}

        # Firstly, Sequentially process each frames 
        for sel in self.selections.keys():
          # Secondly, sequentially process each selection
          self.atom_mass[sel] = []; 
          self.norm_mass[sel] = []; 
          self.gauss_mass[sel] = []; 
          self.atom_name[sel]  = []; 
          self.res_name[sel]   = []; 
          for i in range(len(self.trajpdb)):
            thisxyz = self.trajpdb.xyz[i]; 
            # Thirdly: Extract coordinates within the cutoff, atom index and
            selidx = self.selections[sel]; 
            selxyz = thisxyz[selidx]; 
            # Fourthly: restrain real candidates
            cand_status = distance_matrix(selxyz, self.curveCenter) <= self.search_cutoff; 
            cand_status = cand_status.squeeze(); 
            cand_index  = selidx[cand_status]; 
            cand_xyz    = selxyz[cand_status]; 
            cand_distmatrix = distance_matrix(self.points3D, cand_xyz)
            cand_diststatus = cand_distmatrix < 1.75
            # cand_distmatrix < 3.75

            mins = np.min(cand_distmatrix, axis=1)
            idx_lst = [np.where(cand_distmatrix[m] == mins[m])[0][0] if np.any(cand_diststatus[m,:]) else -1 for m in range(len(mins))]
            candlst = [cand_index[m] if m>=0 else -1 for m in idx_lst]

            atom_name_frameN = [self.pdb_atomic_names[m]  if m>0 else False for m in candlst]; 
            res_name_frameN  = [self.pdb_residue_names[m] if m>0 else False for m in candlst]; 
            atom_name_frameN = self.points_to_3D(atom_name_frameN, dtype=str); 
            res_name_frameN  = self.points_to_3D(res_name_frameN, dtype=str); 

            atom_mass_frameN = [self.pdb_atomic_numbers[m] if m>0 else 0 for m in candlst]; 
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
          self.gauss_mass[sel]= np.array(self.gauss_mass[sel]);
          self.atom_name[sel] = np.array(self.atom_name[sel]).astype(str);
          self.res_name[sel]  = np.array(self.res_name[sel]).astype(str);
        loadtime = time.perf_counter() - st_elm;
        print(f"Element: featurized {self.frameNr} frames, took {loadtime:.2f} seconds; Avg: {loadtime/self.frameNr:.2f}")

      elif feature.lower() == 'charge':
        self.features.append(feature)
        print("Reading forcefield files")
        reader = ffreader(settings["forcefield_ligand"])
        waitlist = list(set([i.upper() for i in self.reslist])) + list(reader.residuemap.keys())
        reader.addFF(settings["forcefield_protein"], waitlist=waitlist)
        reader.addFF(settings["forcefield_solvent"], waitlist=waitlist)

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

          sel_distmatrix_max = distance_matrix(self.points3D+self.cell_length/2, selxyz)
          sel_distmatrix_min = distance_matrix(self.points3D-self.cell_length/2, selxyz) 
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


class feature_3d_reader:
  def __init__(self, pickleFile):
    # Generate the 2D/3D hilbert curve
    with open(pickleFile, "rb") as file1:
      featuredic = pickle.load(file1)
      print(featuredic.keys()); 
      self.distances = featuredic["distances"]; 
      self.length3D  = featuredic["length3D"];
      self.points3D  = featuredic["points3D"]; 
      self.index3D   = featuredic["index3D"];
      self.features  = featuredic["features"];

      self.atom_groups = featuredic["atom_groups"]; 
      self.frameList = featuredic["frameList"]
      self.frameNr   = featuredic["frameNr"]
      if "element" in self.features:
        self.atom_mass  = featuredic["atom_mass"]
        self.norm_mass  = featuredic["norm_mass"]
        self.gauss_mass = featuredic["gauss_mass"]
      if "entropy" in self.features:
        self.entropy = featuredic["entropy"]
        self.gauss_entropy = featuredic["gauss_entropy"]
      if "charge" in self.features:
        self.atom_charge  = featuredic["atom_charge"]; 
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
      elements = ["Du"]*len(coordinates)
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
      elem  = elements[i]; 
      tmpstr = "".join(["{:>8}".format(i) for i in point]); 
      tmpstr = "".join([f"{i:>8}" for i in point]); 
      thisline = f"ATOM  {i:>5}  {elem:<3}{tempstr[16:30]}{tmpstr}{tempstr[54:60]}{round(bfval,2):>6}\n"
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
    print(f"Filtered data: mean:{np.mean(filtered):2f}, Std:{np.std(filtered):2f}, Max:{np.max(filtered):2f}, Min:{np.min(filtered):2f}")
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


