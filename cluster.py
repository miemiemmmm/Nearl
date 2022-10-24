from sklearn.cluster import AgglomerativeClustering
import numpy as np 
import pytraj as pt 

def ClusterAgglomerative(pdist, clusternr):
  """
    Cluster the distance values to {clusternr} classes
    [0 0 3 ... 2 3 2]  # 10 classes
  """
  hc = AgglomerativeClustering(n_clusters=clusternr, affinity = 'euclidean', linkage = 'ward'); 
  y_hc = hc.fit_predict(pdist.T); 
  return y_hc

def RandomPerCluster(cluster, number=1):
  """
    Choose {number} data points from each cluster
  """
  retlst = []
  for gp in np.unique(cluster):
    gp1 = [i for i,j in enumerate(cluster) if j == gp]; 
    randidx = np.random.choice(gp1, number)
    if number ==1: 
      retlst.append(int(randidx))
    elif number >1: 
      retlst += list(randidx)
  retlst.sort()
  return retlst

def RMSDClusters(cluster, pdbfile, trajfile):
  """
    Caluclate the RMSD values of the each clusters
    Examine the quality of clusters
  """
  rmsdlst=[]
  for gp in np.unique(cluster):
    gp1 = [i for i,j in enumerate(cluster) if j == gp]; 
    traj1 = pt.load(trajfile, top=pdbfile, frame_indices=gp1);
    traj1.superpose("@CA")
    print(f"Cluster {gp}: Selected frames: {len(gp1)} frames "); 
    #     print("==>", tm_scores[gp1])
    #     print("==>", np.std(tm_scores[gp1]))
    gprmsd = pt.rmsd(traj1, ":LIG&!@H="); 
    gpgyr = pt.radgyr(traj1, ":LIG&!@H="); 
    print(f"RMSD: {gprmsd.mean()}-{gprmsd.std()}\nROG: {gpgyr.mean()}-{gpgyr.std()}")
    rmsdlst.append(gprmsd.mean())
  print(f"Average RMSD: {np.mean(rmsdlst)}\n")
  return rmsdlst
  
def RMSDGroup(frames, pdbfile, trajfile):
  """
    Caluclate the RMSD values of certain frames 
    Examine the quality of clusters
  """
  traj1 = pt.load(trajfile, top=pdbfile, frame_indices=frames);
  rmsds = pt.rmsd(traj1, ":LIG&!@H="); 
  print(f">>> RMSD: {rmsds.mean()}-{rmsds.std()}\n")
  return rmsds
