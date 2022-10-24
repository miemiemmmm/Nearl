import requests
import json 

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

def SubmitPDB(pdbfile, jobid, pdbcode="USER", water="T3P"):
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
    print("Finished the submission of PDB: ", response.status_code,  response.url, response.text); 
    return dic
  else: 
    return False 

def SubmitMOL2(mol2file, jobid):
  with open(mol2file, "r") as file1: 
    mol2str = file1.read(); 
  data = f'cmd=depositligand&ligandmol2={mol2str}&JOBID={jobid}'; 
  response = requests.post('http://130.60.168.149/fcgi-bin/ACyang.fcgi', data=data); 
  if response.status_code == 200: 
    dic = json.loads(response.text)
    print("Finished the submission of MOL2: ", response.status_code,  response.url, response.text); 
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

def GetSessionPDB(jobid):
  session_info = RecallSession(jobid);
  return session_info["pdbfile"]
def GetSessionMOL2(jobid):
  session_info = RecallSession(jobid);
  return session_info["molfile"]
  
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


