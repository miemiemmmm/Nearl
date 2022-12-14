import pandas as pd
import numpy as np 
from sklearn.cluster import AgglomerativeClustering

import time, sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

def load_model(trained_model):
  start = time.perf_counter()
  model = sent2vec.Sent2vecModel()
  try:
      model.load_model(trained_model)
  except Exception as e:
      print(e)
  end = time.perf_counter()
  print(f'Model successfully loaded. {end-start:.2f} second used \n'); 
  sentence = preprocess_sentence(
      'If, this sentence, no punctuation, then, \
      the preprocess_sentence function, work good. \
      The correct sentence_vector.shape is (1, 600), now check: \n')
  sentence_vector = model.embed_sentence(sentence)
  print(sentence, sentence_vector.shape)
  return model


def preprocess_sentence(text):
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


def tovec(sents):
  retvec = []
  for sent in sents:
    sent = preprocess_sentence(sent); 
    retvec.append(model.embed_sentence(sent)); 
  retvec = np.array(np.squeeze(retvec)); 
  print(f"Return shape of the vector is {retvec.shape}")
  return retvec


modelfile= "/home/miemie/Downloads/wiki_unigrams.bin"
modelfile = "/home/miemie/Downloads/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
model = load_model(modelfile);

title_csv = "/home/miemie/Dropbox/PhD/project_MD_ML/PDBbind_v2020_refined/index/titles.csv"
table = pd.read_csv(title_csv, index_col=0)
# print(table)

vec = tovec(table.title)

clusters = ClusterAgglomerative(vec, 100)
# print(clusters)
# print(clusters.shape)
# print(np.unique(clusters, return_counts=True))
choice = RandomPerCluster(clusters, 1); 

for i in table.title[choice]:
  print(i)


