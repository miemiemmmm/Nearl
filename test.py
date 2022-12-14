import pytraj as pt 
import numpy as np 
from BetaPose import utils, cluster

class TRAJ:
  def __init__(self, trajfile, pdbfile):
    # print("Generate a new traj")
    self.traj = pt.load(trajfile, top=pdbfile);
    self.traj.top.set_reference(self.traj[0])

  def cluster_pairwise(self, mask, **kwarg):
    if "countermask" in kwarg.keys():
      countermask = kwarg["kwarg"]
      pdist, y = utils.PairwiseDistance(self.traj, f"{mask}&!@H=", f"{countermask}&!@H=", use_mean=True);
    else:
      pdist, y = utils.PairwiseDistance(self.traj, f"{mask}&!@H=", f"{mask}<@6&!{mask}&@C,CA,CB,N,O", use_mean=True);
    clusters = cluster.ClusterAgglomerative(pdist, 10);
    cluster_rand = cluster.RandomPerCluster(clusters, number=1);
    self.frames = cluster_rand;
    return self.frames

  def cluster_rmsd(self):
    self.traj = self.traj;

  def slicetraj(self, frameindices):
    frameindices = np.array(frameindices).astype(int)
    self.traj = pt.Trajectory(self.traj[frameindices])

  def cluster(self, method="", **kwarg):
    if len(method) == 0:
      self.cluster_pairwise(**kwarg);
    elif (method == "distance"):
      pass
    return self.frames

  def updatesearchlist(self, mask, cutoff):
    self.searchlist = self.traj.top.select(f"{mask}<@{cutoff}");
    return self.searchlist;


class TrajectoryLoader: 
  def __init__(self, trajs, tops):
    if isinstance(trajs, str):
      self.trajs = [trajs]; 
      self.tops  = [tops]; 
    elif isinstance(trajs, list):
      self.trajs = trajs; 
      self.tops  = tops; 
  def __iter__(self):
    return self.__loadtrajs(self.trajs, self.tops); 
  
  def __loadtrajs(self, trajs, tops):
    for traj, top in zip(trajs, tops):
      yield TRAJ(traj, top)


class feature:
  def __init__(self):
    print("here")

  def __display__(self):
    print(self.traj);

  def __str__(self):
    return self.__class__.__name__

  def hook(self, featurizer):
    self.featurizer = featurizer
    self.top = featurizer.traj.top

  def run(self, trajectory):
    """
      update interval
      self.traj.superpose arguments.
      updatesearchlist arguments.
    """
    self.traj = trajectory.traj;
    self.feature_array=[];
    for index, frame in enumerate(self.traj):
      theframe = self.traj[index];
      if index % 1 == 0:
        refframe = pt.Frame(theframe);
        self.searchlist = trajectory.updatesearchlist(":MDL" , 18);
        self.traj.top.set_reference(theframe);
        self.traj.superpose(ref=refframe, mask="@CA")

      feature_frame = self.forward(self.traj[index]);
      self.feature_array.append(feature_frame);
    self.feature_array = np.array(self.feature_array);
    return self.feature_array;


import pandas as pd 
import requests, json, os, tempfile

class ACGUIKIT_REQUESTS:
  def __init__(self, url): 
    self.JOBID = ""; 
    self.url = url; 
    
  def initiate(self, jobid): 
    assert len(jobid) ==8, "Please provide a valid 8-character session ID ";
    self.JOBID = jobid
    emptypdb = "ATOM      1  CH3 LIG A   1      -8.965  24.127  -8.599  1.00  0.00\nEND"
    self.submitPDB(emptypdb, mode="str"); 
  
  def recall(self, jobid):
    """
      Primarily to obtain the session ligand and protein structure
    """
    assert len(jobid) == 8, "Please provide a valid 8-character session ID ";
    data = { 'cmd': 'recallSession', 'JOBID': jobid}; 
    response = requests.post(self.url, data=data); 
    if response.status_code==200:
      ret = json.loads(response.text); 
      self.protein = ret["pdbfile"]; 
      self.ligand = ret["molfile"]; 
      self.JOBID = jobid; 
      return ret
    else:
      print("Recall session failed"); 
      return;
    
  def listTraj(self): 
    assert len(self.JOBID) == 8, "Please provide a valid 8-character session ID "; 
    data = {'cmd': 'sendtraj','JOBID': self.JOBID,'querymode': '7'}; 
    response = requests.post(self.url, data=data); 
    if response.status_code==200:
      ret = json.loads(response.text)
      for key, val in json.loads(ret["Params"]).items(): 
        atomnr = val["atomnr"] or 0; 
        waternr = val["watnr"] or 0; 
        date = val["gendate"]; 
        eng = val["prodeng"]; 
        interval = val["outinterval"] or 0; 
        nsteps = val["nrsteps"] or 0; 
        ensemble = val["ensemble"]; 
        status = val["exitmsg"]
        print(f"{key}: {ensemble:3s}|{int(atomnr):6d}|{int(waternr):6d}|{int(interval):6d}|{int(nsteps):8d}|{eng:6s}|{date:18s}|{status}")
      return ret
    else:
      print("List trajectory failed"); 
      return
    
  def recallTraj(self, trajid): 
    assert len(self.JOBID) == 8, "Please provide a valid 8-character session ID "; 
    data = { 'cmd': 'sendtraj', 'JOBID': self.JOBID, 'querymode': '13', 'traj_id': trajid }; 
    response = requests.post(self.url, data=data); 
    
    if response.status_code==200:
      ret = json.loads(response.text)
      # print(ret.keys())
      return ret
    else:
      print("Recall trajectory failed"); 
      return
    
  def submitPDB(self, pdbtoken, pdbcode="USER", water="T3P", mode="file"):
    assert len(self.JOBID) == 8, "Please provide a valid 8-character session ID "; 
    if mode == "str":
      pdbstr = pdbtoken;
    elif os.path.isfile(pdbtoken):
      with open(pdbtoken, "r") as file1:
        pdbstr = file1.read();
    else:
      print("Please provide a valid pdb file or string"); 
      return
    data = {
      'cmd': 'deposittarget',
      'water': water, 
      'hisdef': 'HID', 
      'ligpdb': '', 
      'ligname': '', 
      'target': pdbstr, 
      'targetname': pdbcode, 
      'JOBID': self.JOBID, 
      'unsuppres': '', 
    }
    response = requests.post(self.url, data=data)
    if response.status_code == 200:
      dic = json.loads(response.text)
      return dic
    else:
      print("Submit PDB failed"); 
      return
    
  def submitMOL2(self, mol2token, mode="file"):
    assert len(self.JOBID) == 8, "Please provide a valid 8-character session ID "; 
    if mode == "str":
      mol2str = mol2token;
    elif os.path.isfile(mol2token):
      with open(mol2token, "r") as file1:
        mol2str = file1.read();
    else: 
      print("Please provide a valid mol2 file or string"); 
      return 
    data = f'cmd=depositligand&ligandmol2={mol2str}&JOBID={self.JOBID}';
    response = requests.post(self.url, data=data);
    if response.status_code == 200:
      dic = json.loads(response.text)
      return dic
    else:
      print("Submit MOL2 failed"); 
      return
    
  def prepareSession(jobid, parms={}):
    """
      After uploading the protein PDB file and ligand MOL2, prepare the session
    """
    session_info = self.recall(self.JOBID); 
    if isinstance(session_info, dict):
      datadict = {'cmd': 'preptarget', 'water': '', 'nwaters': '0', 'fullpdb': self.protein, 'JOBID': self.JOBID,
       'waterchoice': 'T3P', 'hischoice': 'HID', 'chainsel': 'none', 'ligand': 'none', 'ligmol2': self.ligand,
       'ligsdf': '', 'maxloopl': '0', 'nrsteps': '5000', 'mini_mode': '3', 'mini_grms': '0.01',
       'sc_polar': '1.0', 'sc_impsolv': '1.0', 'pdb_tolerance_a': '20.0', 'pdb_tolerance_b': '0.75+1.25',
       'appendix': '# comment', 'unsuppres': '', 'OBpH': '7.4', 'OBpercept': '5'
      }
      for i in parms.keys():
        if i in datadict.keys():
          datadict[i] = parms[i];

      data = "";
      for key, val in datadict.items():
        data += f"{key}={val}&";
      data = data.strip("&");

      response = requests.post(self.url, data=data);
      if response.status_code == 200:
        dic = json.loads(response.text)
        status = dic["status"];
        print(f"System preparation exit status is {status}", response.status_code,  response.url, );
        return dic
      else:
        return
    else:
      print("Fatal: Failed to query the session info"); 
      return
    
  def distLig(self, theid, mode="session", ligname="LIG"):
    """
      Calculate the COM distance from protein to ligand
      Protein uses CA atoms ; Ligand use ALL atoms
      Have 2 modes: session/file
    """
    assert len(self.JOBID) == 8, "Please provide a valid 8-character session ID "; 
    import pytraj as pt
    if mode == "session":
      from BetaPose import session_prep
      from rdkit import Chem
      from scipy.spatial import distance_matrix
      import re
      import numpy as np
      with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1, tempfile.NamedTemporaryFile("w", suffix=".mol2") as file2:
        session = session_prep.RecallSession(theid)
        file1.write(session["pdbfile"]);
        protcom = pt.center_of_mass(pt.load(file1.name), "@CA");
        try:
          # Mol2 could successfully be parsed in pytraj
          file2.write(session["molfile"]);
          traj = pt.load(file2.name)
          ligcom = pt.center_of_mass(pt.load(file2.name));
        except Exception as e:
          # Directly calculate the COM of the ligand 
          # print(f"Error occurred while calculating the Ligand COM: {e}")
          atoms = session["molfile"].split("@<TRIPOS>ATOM\n")[1].split("@<TRIPOS>")[0];
          atoms = [i.strip().split() for i in atoms.strip("\n").split("\n")];
          coord = np.array([i[2:5] for i in atoms]).astype(np.float32);
          atomtypes = [re.sub(r"[0-9]", "", i[1]) for i in atoms];
          masses = [];
          for i in atomtypes:
            try:
              m = Chem.Atom(i).GetMass()
              masses.append(m);
            except:
              masses.append(0);
          com = np.average(coord, axis=0, weights=masses)
          ligcom = np.array([com])
        return distance_matrix(ligcom, protcom).item();
    elif mode == "traj": 
      ret = self.recallTraj(theid); 
      if len(ret["PDBFile"]) > 66: 
        with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1:
          file1.write(ret["PDBFile"]); 
          traj = pt.load(file1.name);
          dist = pt.distance(traj, f"@CA  :{ligname}")
          return dist.item()
      else: 
        return None
    elif mode == "file":
      traj = pt.load(theid);
      dist = pt.distance(traj, f"@CA  :{ligname}")
      return dist.item()
    else:
      return None
  
