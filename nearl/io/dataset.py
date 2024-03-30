import os, time
import h5py
import torch
import numpy as np 
import multiprocessing as mp

from .. import config, printit

def readdata(input_file, keyword, theslice):
  with h5py.File(input_file, "r") as h5file:
    ret_data = h5file[keyword][theslice]
  return ret_data

def split_array(input_array, batch_size): 
  # For the N-1 batches, the size is uniformed and only variable for the last batch
  bin_nr = (len(input_array) + batch_size - 1) // batch_size 
  if len(input_array) == batch_size * bin_nr: 
    return np.array_split(input_array, bin_nr)
  else:
    final_batch_size = len(input_array) % batch_size
    if bin_nr-1 == 0: 
      return [input_array[-final_batch_size:]]
    else:
      return np.array_split(input_array[:-final_batch_size], bin_nr-1) + [input_array[-final_batch_size:]]
    
class Dataset: 
  def __init__(self, files, grid_dim, label_key="label", feature_keys=[], benchmark = False): 
    self.size = np.array([grid_dim, grid_dim, grid_dim], dtype=int)
    self.FILELIST = files
    self.sample_sizes = []
    self.total_entries = 0
    self.BENCHMARK = bool(benchmark)
    
    # Check the existence of the feature keys in the file
    for i in feature_keys:
      if i not in h5py.File(files[0], "r").keys():
        raise KeyError(f"Feature key {i} is not in the file")
    self.feature_keys = feature_keys
    self.channel_nr = len(feature_keys)
    if label_key not in h5py.File(files[0], "r").keys():
      raise KeyError(f"Label key {label_key} is not in the file")
    self.label_key = label_key

    for filename in self.FILELIST:
      if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")
      with h5py.File(filename, "r") as hdf:
        label_nr = hdf[label_key].shape[0]
        if config.verbose(): 
          printit(f"Found {label_nr} labels in {filename}")
        self.sample_sizes.append(label_nr)
        self.total_entries += label_nr

    if config.verbose() or self.BENCHMARK or config.debug():
      printit(f"Total number of samples: {sum(self.sample_sizes)}")

    self.file_map = np.full(self.total_entries, 0, dtype=np.uint32)
    self.position_map = np.full(self.total_entries, 0,dtype=np.uint64)
    

    tmp_count = 0
    for fidx, filename in enumerate(self.FILELIST):
      # Full the file map and position map
      label_nr = self.sample_sizes[fidx]
      self.file_map[tmp_count:tmp_count+label_nr] = fidx
      self.position_map[tmp_count:tmp_count+label_nr] = np.arange(label_nr)
      tmp_count += label_nr

  def __len__(self):
    return self.total_entries
  
  def position(self, index):
    return self.position_map[index]
  
  def filename(self, index):
    return self.FILELIST[self.file_map[index]]

  def __getitem__(self, index):
    # Get the file
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    
    data = torch.zeros([self.channel_nr] + self.size.tolist(), dtype=torch.float32)
    for i, key in enumerate(self.feature_keys):
      data[i] = torch.from_numpy(readdata(self.filename(index), key, self.position(index)))
    label = torch.tensor([readdata(self.filename(index), self.label_key, self.position(index))])
    return data, label
  
  def mini_batch_task(self, index):
    tasks = []
    for i in range(len(self.feature_keys)):
      tasks.append((self.filename(index), self.feature_keys[i], np.s_[self.position(index)]))
    tasks.append((self.filename(index), self.label_key, np.s_[self.position(index)]))
    return tasks
  

  def mini_batches(self, batch_nr=None, batch_size=None, shuffle=True, process_nr=24, **kwargs):
    """
    """
    
    indices = np.arange(self.total_entries)
    if shuffle:
      np.random.shuffle(indices)
    if batch_nr is not None: 
      batch_size = self.total_entries // batch_nr
      batches = split_array(indices, batch_size)
    elif batch_size is not None:
      batches = split_array(indices, batch_size)
    else:
      raise ValueError("Either batch_nr or batch_size should be specified")
  
    if config.verbose() or self.BENCHMARK or config.debug():
      printit(f"Iterating the dataset: {len(batches)} batches with batch size {batch_size}. Using {process_nr} processes.")

    pool = mp.Pool(process_nr)
    
    taskset_size = self.channel_nr+1
    st = time.perf_counter()
    for batch_idx, batch in enumerate(batches): 
      data = torch.zeros([len(batch), self.channel_nr] + self.size.tolist(), dtype=torch.float32)
      label = torch.zeros([len(batch), 1], dtype=torch.float32)
      
      taskset = []
      for i in batch:
        taskset += self.mini_batch_task(i)
      results = pool.starmap(readdata, taskset)
      
      for i in range(len(batch)):
        for j in range(self.channel_nr):
          data[i, j] = torch.from_numpy(results[i*taskset_size+j])
        _label = np.array([results[i*taskset_size+self.channel_nr]])
        label[i] = torch.from_numpy(_label)
      
      if self.BENCHMARK:
        mps = (time.perf_counter() - st)/len(batch) * 1000
        sps = len(batch)/(time.perf_counter()-st)
        time_remaining = (len(batches) - batch_idx) * (time.perf_counter() - st)
        printit(f"Batch {batch_idx:4d} ({batch_size} entries): MPS: {mps:8.2f}; SPS: {int(sps):6d}; Time left: {time_remaining:8.2f} seconds")
      st = time.perf_counter()
      yield data, label


# dataset = dataloader(files, 32, 
#   feature_keys=["mass", "mass_distinct_count", "partial_charge_negative", "partial_charge_negative"], 
#   label_key="label", 
#   benchmark=True
# )