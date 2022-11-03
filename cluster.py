from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
import numpy as np 
import time
import pandas as pd
from sgt import SGT

def Agglomerative(data, clusternr, return_cluster=False):
  """
    Cluster the distance values to {clusternr} classes
    [0 0 3 ... 2 3 2]  # 10 classes
  """
  hc = AgglomerativeClustering(n_clusters=clusternr, affinity = 'euclidean', linkage = 'ward', compute_distances=True);
  y_hc = hc.fit_predict(data);
  # print(f"Predictions: {len(y_hc)} ; Cluster Number: {len(set(y_hc))}")
  if return_cluster:
    return y_hc, hc
  else:
    return y_hc

def _KMeans(data, clusternr):
  spectral = KMeans(n_clusters=clusternr);
  labels = spectral.fit_predict(data);
  return labels

def MBKMeans(data, clusternr):
  spectral = MiniBatchKMeans(n_clusters=clusternr);
  labels = spectral.fit_predict(data);
  return labels

def RandomPerCluster(cluster, number=1):
  """
    Choose {number} data points from each cluster
  """
  retlst = [];
  cluster = np.array(cluster); 
  for c,count in zip(*np.unique(cluster, return_counts=True)):
    gp1 = [i for i,j in enumerate(cluster) if j == c];
    if number < count:
      randidx = np.random.choice(gp1, number, replace=False);
    else:
      randidx = np.random.choice(gp1, count, replace=False);
    if number ==1:
      retlst.append(int(randidx))
    elif number >1:
      retlst += list(randidx)
  retlst.sort()
  print(f"Random choice per cluster returned {len(retlst)} values")
  return retlst

def RMSDClusters(cluster, pdbfile, trajfile):
  """
    Caluclate the RMSD values of the each clusters
    Examine the quality of clusters
  """
  import pytraj as pt 
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


####################################################################################################
##################################### Sentence to vector model #####################################
####################################################################################################

def LoadModel(trained_model):
  """
    Load a sent2vec model
  """
  import sent2vec
  start = time.perf_counter()
  model = sent2vec.Sent2vecModel()
  try:
    model.load_model(trained_model)
  except Exception as e:
    print(e)
  print(f'Model successfully loaded. {time.perf_counter() - start:.2f} second used');
  sentence = SentencePreprocess('Check the model with a test sentence')
  sentence_vector = model.embed_sentence(sentence)
  print(f"Test sentence: {sentence} ; Shape of the vector: {sentence_vector.shape}")
  return model

def SentencePreprocess(text):
  from nltk import word_tokenize
  from nltk.corpus import stopwords
  from string import punctuation
  text = text.replace('/', ' / ')
  text = text.replace('.-', ' .- ')
  text = text.replace('.', ' . ')
  text = text.replace('\'', ' \' ')
  stop_words = set(stopwords.words('english'))
  try:
    text = text.lower()
  except:
    text = text.to_string()
    text = text.lower()
  tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
  return ' '.join(tokens)

def SequenceEmbed(seqs):
  """
    Convert the sequence to the a fixed length vector
  """
  st_time = time.perf_counter();
  listseq=[[idx,[j for j in i]] for idx,i in enumerate(seqs)];
  listseq = pd.DataFrame(listseq, columns=['id', 'sequence']);
  sgt = SGT(kappa = 10, lengthsensitive = False, flatten=True);
  embedding = sgt.fit_transform(listseq);
  embedding = embedding.set_index('id');
  print(f"Sequence Embedding takes {time.perf_counter() - st_time:.3f} seconds with shape of {embedding.shape}")
  return embedding

def SentenceEmbed(sents):
  """
    Vectorize the sentences
  """
  retvec = [];
  st_time = time.perf_counter();
  for sent in sents:
    sent = SentencePreprocess(sent);
    retvec.append(model.embed_sentence(sent));
  retvec = np.array(np.squeeze(retvec));
  print(f"Sentence Embedding takes {time.perf_counter() - st_time:.3f} seconds with shape of {embedding.shape}")
  return retvec


