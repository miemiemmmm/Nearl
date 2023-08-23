import json, re, requests, tempfile, time, datetime
import pytraj as pt 
import numpy as np 
from . import utils

class MDPost:
  def __init__(self):
    self.__data = {
      'cmd': 'runsmdsim',
      'JOBID': '',
      'sim_ifexp': 'true',
      'sim_ifequil': 'false',
      'sim_ifimp': 'false',
      'sim_nrcopy': '1',
      'prod_outputformat': 'netcdf',
      'prod_outputgroup': 'all',
      'prod_nrsteps': '5000000',
      'prod_outinterval': '5000',
      'prod_prodeng': 'gmxmd',
      'prod_ensemble': 'npt',
      'prod_temp': '298',
      'prod_pressure': '1.00',
      'prod_timestep': '0.002',
      'sim_ifrestart': 'false',
      'sim_rstfrom': 'false',
      'sim_haslig': 'true',
      'sim_ifreconly': 'false',
      'sim_slotsel': '0',
      'sim_initbox': '-1',
      'sim_neuiontype': 'kcl',
      'sim_solmod': 'tip3p',
      'sim_ionconc': '0.15',
      'sim_forcefield': 'charmm36',
      'sim_equilnr': '6',
      'equil0_ensemble': 'em',
      'equil0_eng': 'gmxem',
      'equil0_temp': '0',
      'equil0_timestep': '0',
      'equil0_nrsteps': '5000',
      'equil1_ensemble': 'nvt',
      'equil1_eng': 'gmxmd',
      'equil1_temp': '150',
      'equil1_timestep': '0.002',
      'equil1_nrsteps': '5000',
      'equil1_tctime': '0.1',
      'equil2_ensemble': 'nvt',
      'equil2_eng': 'gmxmd',
      'equil2_temp': '200',
      'equil2_timestep': '0.002',
      'equil2_nrsteps': '5000',
      'equil2_tctime': '0.1',
      'equil3_ensemble': 'nvt',
      'equil3_eng': 'gmxmd',
      'equil3_temp': '249',
      'equil3_timestep': '0.002',
      'equil3_nrsteps': '5000',
      'equil3_tctime': '0.1',
      'equil4_ensemble': 'nvt',
      'equil4_eng': 'gmxmd',
      'equil4_temp': '298',
      'equil4_timestep': '0.002',
      'equil4_nrsteps': '5000',
      'equil4_tctime': '0.1',
      'equil5_ensemble': 'npt',
      'equil5_eng': 'gmxmd',
      'equil5_temp': '298',
      'equil5_timestep': '0.002',
      'equil5_nrsteps': '20000',
      'equil5_tctime': '0.1',
    }
  def __str__(self):
    return self.__data.__str__()
  @property
  def data(self):
    return self.__data
  @property
  def jobid(self):
    return self.__data["JOBID"]
  @jobid.setter
  def jobid(self, val):
    val = str(val).strip().replace(" ", "");
    if len(val) != 8:
      print(f"Not a valid jobid: 12 characters rather than {len(val)}")
      return
    else:
      self.__data["JOBID"] = val;
  @property
  def nrsteps(self):
    return self.__data["prod_nrsteps"]
  @nrsteps.setter
  def nrsteps(self, val):
    if int(val):
      self.__data["prod_nrsteps"] = val
    else:
      print(f"Please use a valid number as the number of steps")
  @property
  def interval(self):
    return self.__data["prod_outinterval"]
  @interval.setter
  def interval(self, val):
    if int(val):
      self.__data["prod_outinterval"] = val
    else:
      print(f"Please use a valid number as the trajectory output interval")
  @property
  def nrcopy(self):
    return self.__data["sim_nrcopy"]
  @nrcopy.setter
  def nrcopy(self, val):
    if int(val):
      self.__data["sim_nrcopy"] = val
    else:
      print(f"Please use a valid number as the parallel execution")
  @property
  def timestep(self):
    return self.__data["prod_timestep"]
  @timestep.setter
  def timestep(self, val, force=False):
    if float(val) > 0.1 and not force:
      print(f"The unit of the timestep is in ps and the given value seems extremely high. Use data.timestep = ({val}, force=True) to force skipping the value check. ")
    elif float(val):
      self.__data["prod_timestep"] = val
    else:
      print(f"Please use a valid number as the timestep")
  @property
  def date(self):
    time_now = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S'); 
    # self.date = time_now; 
    return time_now; 

  def log(func):
    def func_(self, *args, **kwarg):
      print(f"{self.date} : Simulation {self.jobid} - ", end="")
      func(self, *args, **kwarg)
    return func_

  @log
  def print_(self, *args, color="r", **kwarg):
    text = " ".join(args); 
    if color == "r":
      print(f'\033[91m{text}', **kwarg); 
    elif color == "g":
      print(f'\033[92m{text}', **kwarg); 
    elif color == "b":
      print(f'\033[94m{text}', **kwarg); 
    else:
      print(f'{text}', **kwarg);

  @log
  def print_(self, *args, color="r", **kwarg):
    text = " ".join(args);
    print(f'{text}', **kwarg);

  def update(self, parms):
    for key, value in parms.items():
      if key in self.data.keys():
        self.data[key] = value;
 
  def submit(self, url="http://130.60.168.149/fcgi-bin/ACyang.fcgi"):
    time.sleep(int(time.perf_counter().__str__()[-3:])/3000); 
    if len(self.__data["JOBID"]) != 8: 
      self.print_(f"Please set a proper job ID with a fixed length 8. Found <{self.jobid}>")
      return
    else: 
      # print(f"Submitting the MD simulation job: {self}"); 
      response = requests.post(url, data = self.__data); 
      if response.status_code == 200: 
        self.result = json.loads(response.text);
        self.data["result"] = self.result; 
        if self.result["status"] == 1: 
          self.print_(f"Simulation job is successfully complete. ", end="", color="g"); 
          if "traj_id" in self.result.keys():
            traj_id = self.result["traj_id"]; 
            print(f"The latest trajectory ID is {traj_id}"); 
          else:
            print("");
          msg = self.result["msg"]; 
          self.print_(f"The return message is: <{msg}>", color="g"); 
        else: 
          self.print_(f"Simulation job is failed due to the following reason: ", end=""); 
          print(self.result["msg"]); 
        return self.data
      else: 
        self.data["result"] = "Failed"
        self.print_(f"Simulation {self.jobid}: The job seems to failed due to the network problem. "); 
        return self.data

def RecallSession(jobid):
  """
  Primarily to obtain the session ligand and protein structure
  """
  data = {
      'cmd': 'recallSession',
      'JOBID': jobid,
  }
  response = requests.post('http://130.60.168.149/fcgi-bin/ACyang.fcgi', data=data)
  if response.status_code==200: 
    return json.loads(response.text)
  else: 
    return false

def SubmitPDB(pdbfile, jobid, pdbcode="USER", water="T3P", mode="file"):
  if mode == "str":
    pdbstr = pdbfile; 
  else: 
    with open(pdbfile, "r") as file1: 
      pdbstr = file1.read(); 
  data = {
      'cmd': 'deposittarget',
      'water': water,
      'hisdef': 'HID',
      'ligpdb': '',
      'ligname': '',
      'target': pdbstr,
      'targetname': pdbcode,
      'JOBID': jobid,
      'unsuppres': '',
  }
  response = requests.post('http://130.60.168.149/fcgi-bin/ACyang.fcgi', data=data)
  if response.status_code == 200: 
    dic = json.loads(response.text)
    # print("Finished the submission of PDB: ", response.status_code,  response.url, response.text.strip("\n")); 
    return dic
  else: 
    return False 

def SubmitMOL2(mol2file, jobid, mode="file"):
  if mode == "str": 
    mol2str = mol2file; 
  else: 
    with open(mol2file, "r") as file1: 
      mol2str = file1.read(); 
  data = f'cmd=depositligand&ligandmol2={mol2str}&JOBID={jobid}'; 
  response = requests.post('http://130.60.168.149/fcgi-bin/ACyang.fcgi', data=data); 
  if response.status_code == 200: 
    dic = json.loads(response.text)
    # print("Finished the submission of MOL2: ", response.status_code,  response.url, response.text.strip("\n")); 
    return dic
  else: 
    return False 

def PrepareSession(jobid, parms={}):
  """
  After uploading the protein PDB file and ligand MOL2, prepare the session
  """
  session_info = RecallSession(jobid); 
  if isinstance(session_info, dict):
    pdbfile = session_info["pdbfile"]; 
    molfile = session_info["molfile"]; 
    datadict = {'cmd': 'preptarget', 'water': '', 'nwaters': '0', 'fullpdb': pdbfile, 'JOBID': jobid, 
     'waterchoice': 'T3P', 'hischoice': 'HID', 'chainsel': 'none', 'ligand': 'none', 'ligmol2': molfile, 
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

    response = requests.post('http://130.60.168.149/fcgi-bin/ACyang.fcgi', data=data); 
    if response.status_code == 200: 
      dic = json.loads(response.text)
      status = dic["status"]; 
      print(f"System preparation exit status is {status}", response.status_code,  response.url, ); 
      return dic
    else: 
      return False 
  else: 
    print("Fatal: Failed to query the session info")
    return False 

def RunSimpleEquil(sessid, nrsteps=3000):
  """
    A simple equilibration process
  """
  data = {
    'cmd': 'runsmdsim',
    'JOBID': sessid,
    'sim_ifequil': 'true',
    'sim_ifimp': 'false',
    'sim_ifexp': 'false',
    'sim_ifrestart': 'false',
    'sim_equilsteps': nrsteps,
    'sim_eqsctor': '0',
    'sim_neuiontype': 'kcl',
    'sim_ionconc': '0.15',
    'sim_eqslotsel': '0'
  }
  response = requests.post('http://130.60.168.149/fcgi-bin/ACyang.fcgi', data=data); 
  if response.status_code == 200:
    dic = json.loads(response.text); 
    print("Finished the running of equilibration job: ", response.status_code,  response.url);
    return dic
  else:
    return {}

def EquilToSession(sessid, nrsteps=10000):

  equilmol2 = GetEquilMOL2(sessid);
  equilpdb  = GetEquilPDB(sessid); 
  sessionpdb = GetSessionPDB(sessid); 
  if not isinstance(equilpdb, str) or len(equilpdb) <= 0: 
    return
  if not isinstance(sessionpdb, str) or len(sessionpdb) <= 0: 
    return
  with tempfile.NamedTemporaryFile(suffix=".pdb") as file1, tempfile.NamedTemporaryFile(suffix=".pdb") as file2: 
    file1.write(bytes(equilpdb, encoding="utf-8")); 
    file2.write(bytes(sessionpdb, encoding="utf-8")); 
    utils.NormalizePDB(file2.name, file1.name, file1.name); 
    with open(file1.name, "r") as normpdb: 
      equilpdb = normpdb.read(); 
      
  equilpdb = equilpdb.replace("HSD", "HID"); 
  equilpdb = equilpdb.replace("HSE", "HIE"); 
  equilpdb = equilpdb.replace("HSP", "HIP"); 
  equilpdb = re.sub(".*LIG.*\n", "", equilpdb); 
  
  pdb_state = SubmitPDB(equilpdb, sessid, water="T3P", mode="str");
  if isinstance(pdb_state, bool) and pdb_state == False:
    return
  mol2_state = SubmitMOL2(equilmol2, sessid, mode="str");
  if isinstance(mol2_state, bool) and mol2_state == False:
    return
  parms={
    "JOBID" : sessid,
    "nrsteps" : nrsteps, 
  }
  if "T3P" in equilpdb: 
    wat_str = [i for i in equilpdb.split("\n") if re.search("HETATM.*T3P.*", i)]
    wat_str = "\n".join(wat_str)
    wat_result = re.findall("OW.*T3P", equilpdb)
    parms["nwaters"] = len(wat_result); 
    parms["water"] = wat_str
    
  prep_keys = ['water', 'nwaters', 'fullpdb', 'JOBID', 'waterchoice', 'hischoice',
               'chainsel', 'ligand', 'ligmol2', 'ligsdf', 'maxloopl', 'nrsteps',
               'mini_mode', 'mini_grms', 'sc_polar', 'sc_impsolv', 'pdb_tolerance_a', 'pdb_tolerance_b',
               'appendix', 'unsuppres', 'OBpH', 'OBpercept']
  prep_parms = {}; 
  for i in parms.keys():
    if i in prep_keys:
      prep_parms[i] = parms[i]; 
  prep_state = PrepareSession(sessid, parms=prep_parms)
  if isinstance(prep_state, bool) and prep_state == False:
    return
  print(f"Finished the re-preparation of session {sessid}")

def GetSessionPDB(jobid):
  session_info = RecallSession(jobid);
  return session_info["pdbfile"]
def GetSessionMOL2(jobid):
  session_info = RecallSession(jobid);
  return session_info["molfile"]
def GetEquil(sessionID):
  if len(sessionID) != 8: 
    print(f"The length of the session should be 8 rather then the given {len(sessionID)}")
    return {}
  data = { 'cmd': 'sendtraj', 'JOBID': sessionID, 'querymode': '4', }; 
  response = requests.post('http://130.60.168.149/fcgi-bin/ACyang.fcgi', data=data); 
  if response.status_code == 200:
    try: 
      result = json.loads(response.text); 
      status = result["status"];
      if status == 1:
        print(f"Equilbrated structure retrieval exit status is {status}", response.status_code,  response.url);
        return result
      else: 
        print("Error Message from the server: ", result["msg"])
        return result
    except: 
      print("Failed to parse string as json. Please make sure that the target session has gone through anequilibrated process")
      return {}
  else:
    return {}
def GetEquilPDB(sessionID): 
  ret = GetEquil(sessionID); 
  if "PDBFile" in ret.keys():
    return ret["PDBFile"]; 
  else: 
    return {}; 
def GetEquilMOL2(sessionID):
  ret = GetEquil(sessionID);
  if "MOL2File" in ret.keys():
    return ret["MOL2File"];
  else:
    return {};

def PrepNewSession(parms):
  """
  Wrapper of PrepareSession function to submit a request to prepare a new session. 
  Available settings: 
    ['water', 'nwaters', 'fullpdb', 'JOBID', 'waterchoice', 'hischoice', 
    'chainsel', 'ligand', 'ligmol2', 'ligsdf', 'maxloopl', 'nrsteps', 
    'mini_mode', 'mini_grms', 'sc_polar', 'sc_impsolv', 'pdb_tolerance_a', 'pdb_tolerance_b', 
    'appendix', 'unsuppres', 'OBpH', 'OBpercept']
  A simple example: 
  >>> parms={
    "jobid" : "C4001CTU", 
    "pdbcode" : "1CTU", 
    "pdbfile" : "/home/miemie/Dropbox/PhD/project_MD_ML/PDBbind_v2020_refined/1ctu/1ctu_protein.pdb", 
    "mol2file" : "/home/miemie/Dropbox/PhD/project_MD_ML/PDBbind_v2020_refined/1ctu/1ctu_ligand.mol2",
    "nrsteps":1000,
  }
  >>> PrepNewSession(parms)
  """
  if "pdbcode" in parms.keys():
    pdbcode = parms["pdbcode"]
  else:
    pdbcode = "USER"
  print("Preparing the session", parms["jobid"], "; PDB code: ",  parms["jobid"]); 
  pdb_state = SubmitPDB(parms["pdbfile"], parms["jobid"], pdbcode=pdbcode, water="T3P"); 
  if isinstance(pdb_state, bool) and pdb_state == False: 
    return
  mol2_state = SubmitMOL2(parms["mol2file"], parms["jobid"]); 
  if isinstance(mol2_state, bool) and mol2_state == False: 
    return
  prep_keys = ['water', 'nwaters', 'fullpdb', 'JOBID', 'waterchoice', 'hischoice', 
               'chainsel', 'ligand', 'ligmol2', 'ligsdf', 'maxloopl', 'nrsteps', 
               'mini_mode', 'mini_grms', 'sc_polar', 'sc_impsolv', 'pdb_tolerance_a', 'pdb_tolerance_b', 
               'appendix', 'unsuppres', 'OBpH', 'OBpercept']
  prep_parms = {}
  for i in parms.keys():
    if i in prep_keys:
      prep_parms[i] = parms[i]
  
  prep_state = PrepareSession(parms["jobid"], parms=prep_parms)
  if isinstance(prep_state, bool) and prep_state == False: 
    return
  print("Finished the preparation of session ", parms["jobid"])

def getSeqCoord(filename):
  """
    Extract residue CA <coordinates> and <sequence> from PDB chain
  """
  from Bio.PDB.Polypeptide import three_to_one
  traj = pt.load(filename)
  resnames = [i.name for i in traj.top.residues]; 
  trajxyz = traj.xyz[0]; 
  retxyz = [];
  retseq = ""; 
  for atom in traj.top.atoms: 
    if atom.name == "CA":
      try: 
        resname = resnames[atom.resid]
        resxyz = trajxyz[atom.index]
        retseq += three_to_one(resname)
        retxyz.append(resxyz)
      except: 
        pass
  return np.array(retxyz), retseq

def CompareStructures_Obs(pdbcode, sessexp=None):
  """
    Compare the PDB structure before and after then session preparation
    Functions: 
      Extract residue <coordinates> and <sequence> from PDB chain
        Uses the coordinates of the CA atom as the center of the residue
        Skip unknown residues
  """
  from tmtools import tm_align; 
  pdbcode = pdbcode.lower(); 
  with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1: 
    file1.write(utils.fetch(pdbcode))
    coord1, seq1 = getSeqCoord(file1.name); 
  if sessexp != None:
    sessioncode = sessexp(pdbcode)
    print(f"The session ID is : {sessioncode}")
  else: 
    sessioncode = f"C400{pdbcode.upper()}"
  with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1: 
    file1.write(RecallSession(sessioncode)["pdbfile"])
    coord2, seq2 = getSeqCoord(file1.name); 
  print(f"Coordinate set 1 shaped {coord1.shape} ; Sequence shape {len(seq1)}")
  print(f"Coordinate set 2 shaped {coord2.shape} ; Sequence shape {len(seq2)}")
  result = tm_align(coord1, coord2, seq1, seq2)
  print(f"TM_Score: Chain1 {result.tm_norm_chain1} ; Chain2 {result.tm_norm_chain2}")
  if max([result.tm_norm_chain1, result.tm_norm_chain2]) > 0.8:
    print("TM_Score: Good match")
  else: 
    print("TM_Score: Bad match")
  return max([result.tm_norm_chain1, result.tm_norm_chain2]); 

