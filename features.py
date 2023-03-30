class feature:
  def __str__(self):
    return self.__class__.__name__
  def hook(self, featurizer):
    self.featurizer = featurizer
    self.top = featurizer.traj.top
  def getbox(self): 
    return self.featurizer.__points3d
  def getcenter(self): 
    return self.featurizer.center; 
  
  def run(self, trajectory):
    """
    update interval
    self.traj.superpose arguments. 
    updatesearchlist arguments. 
    """
    sttime = time.perf_counter();
    self.traj = trajectory.traj; 
    self.feature_array=[]; 
    for index, frame in enumerate(self.traj):
      # Update search list and shift the box
      if index % self.featurizer.interval == 0:
        self.featurizer.translate()
        self.searchlist = trajectory.updatesearchlist(index, self.featurizer.keymask , 18); 
      feature_i = self.forward(self.traj[index]); 
      self.feature_array.append(feature_i); 
    self.feature_array = np.array(self.feature_array); 
    print(f"Feature {self.__class__.__name__}: {round(time.perf_counter()-sttime, 3)} seconds")
    return self.feature_array; 

class massfeature(feature):
  def __init__(self):
    super(massfeature, self).__init__()
  def forward(self, frame): 
    """
      1. Get the atomic feature
      2. Update the feature 
    """
    pdb_atomic_numbers = np.array([i.atomic_number for i in self.top.atoms]).astype(int); 
    thisxyz = frame.xyz; 
    selxyz = thisxyz[self.searchlist]; 
    self.atom_mass = []; 
    
    cand_status = (distance_matrix(selxyz, self.featurizer.center.reshape(1,3)) <= self.featurizer.cutoff); 
    cand_status = cand_status.squeeze(); 
    cand_xyz    = selxyz[cand_status]; 
    cand_distmatrix = distance_matrix(self.featurizer.points3d, cand_xyz)
    cand_diststatus = cand_distmatrix < 1.75
    
    cand_index  = self.searchlist[cand_status]; 
    mins = np.min(cand_distmatrix, axis=1); 
    idx_lst = [np.where(cand_distmatrix[m] == mins[m])[0][0] if np.any(cand_diststatus[m,:]) else -1 for m in range(len(mins))]; 
    candlst = [cand_index[m] if m>=0 else -1 for m in idx_lst]; 
    
    atom_mass_frameN = [pdb_atomic_numbers[m] if m > 0 else 0 for m in candlst];
    atom_mass_frameN = self.featurizer.points_to_3D(atom_mass_frameN); 
    return np.array(atom_mass_frameN); 

