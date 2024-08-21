import os, time
import h5py
import torch
import numpy as np 
import multiprocessing as mp

from .. import config, printit

__all__ = [
  "Dataset",
  "data_augment",
  "readdata",
]

# Flip by batch with 5 dimensions (batch, channel, x, y, z)
FLIP_DICT_BATCH_LEVEL_TORCH = {
  0: [],
  1: [2],         # Flip along the X-axis, skipping the batch and channel dimensions
  2: [3],         # Flip along the Y-axis
  3: [4],         # Flip along the Z-axis
  4: [2, 3],
  5: [2, 4],
  6: [3, 4],
  7: [2, 3, 4]
}


FLIP_DICT_GRID_LEVEL = {
  0: [],         # No flip
  1: [0],        # Flip along the X-axis
  2: [1],        # Flip along the Y-axis
  3: [2],        # Flip along the Z-axis
  4: [0, 1],     # Flip along the X and Y axes
  5: [0, 2],     # Flip along the X and Z axes
  6: [1, 2],     # Flip along the Y and Z axes
  7: [0, 1, 2],  # Flip along the X, Y, and Z axes
}


def rand_flip_axis(level):
  rand_flip = int(np.random.random()*1000) % 8
  if level == "batch":
    flip = FLIP_DICT_BATCH_LEVEL_TORCH[rand_flip]
  else:
    flip = FLIP_DICT_GRID_LEVEL[rand_flip]
  return flip


def rand_translate(factor=1):
  rand_trans = np.random.randint(low=-1*factor, high=1*factor + 1, size=3)
  return rand_trans


def readdata(input_file, keyword, theslice):
  """
  A simple wrapper function to read the data from the HDF5 file.

  Parameters
  ----------
  input_file : str
    The HDF5 file path.
  keyword : str
    The keyword to read from the HDF5 file.
  theslice : slice
    The slice to read from the HDF5 file.

  Returns
  -------
  np.ndarray
    The data read from the HDF5 file.
  """
  with h5py.File(input_file, "r") as h5file:
    ret_data = h5file[keyword][theslice]
  return ret_data


def readlabel(input_file, position):
  with h5py.hdffile(input_file, "r") as h5file:
    ret_data = h5file["label"][position]
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



def data_augment(batch_array, translation_factor=0, add_noise=False): 
  """
  Batch data are in the shape of (batch_size:0, channel:1, x:2, y:3, z:4)

  Parameters
  ----------
  batch_array : torch.Tensor
    The batch data to augment.
  translation_factor : int
    The factor of translation for data augmentation, defaults to 2.
  add_noise : bool
    Whether to add noise to the data or not, defaults to False.
  
  Notes
  -----
  Data augmentation includes:
  1. Flip along the axes
  2. Rotate along the axes
  3. Translate along the axes
  4. Add noise to the data
  """
  # Flip the data
  flip_axes = rand_flip_axis("batch")
  for axis in flip_axes:
    batch_array = batch_array.flip(dims=(axis,))

  # Rotate the data
  for axis in range(3): 
    do_rotation = np.random.choice([0, 1])
    if do_rotation:
      k = np.random.choice([1, 2, 3])
      if axis == 0: 
        # Rotate along the X-axis
        batch_array = torch.rot90(batch_array, k, [3, 4]) 
      elif axis == 1:
        # Rotate along the Y-axis
        batch_array = torch.rot90(batch_array, k, [2, 4]) 
      else:
        # Rotate along the Z-axis
        batch_array = torch.rot90(batch_array, k, [2, 3]) 
  
  if translation_factor > 0:
    trans = rand_translate(factor=translation_factor)
    # Handling translation edge effects by filling the 'new' space with zeros
    for i, shift in enumerate(trans):
      if shift != 0:
        batch_array = torch.roll(batch_array, shifts=shift, dims=(i+2))
        if shift > 0 and i == 0:
          batch_array[:, :, :shift, :, :] = 0
        elif shift < 0 and i == 0:
          batch_array[:, :, shift:, :, :] = 0
        elif shift > 0 and i == 1:
          batch_array[:, :, :, :shift, :] = 0
        elif shift < 0 and i == 1:
          batch_array[:, :, :, shift:, :] = 0
        elif shift > 0 and i == 2:
          batch_array[:, :, :, :, :shift] = 0
        elif shift < 0 and i == 2:
          batch_array[:, :, :, :, shift:] = 0
    
  if config.verbose() or config.debug():
    msg = f"Flipped axes: {flip_axes}; "
    if translation_factor > 0:
      msg += "Translation: {trans}"
    printit(msg)

  # Apply a gaussian noise to the array
  if add_noise:
    batch_array += torch.randn_like(batch_array) * torch.max(batch_array) * 0.1
  return batch_array


class Dataset: 
  """
  A dataset class that reads data from HDF5 files.

  Parameters
  ----------
  files : list
    A list of HDF5 file paths.
  grid_dim : int
    The dimension of the grid.
  label_key : str
    The key of the label in the HDF5 file.
  feature_keys : list
    The keys of the features in the HDF5 file.

  """
  # TODO Check if the Label is not defined in the file
  def __init__(self, files, grid_dim=None, label_key="label", feature_keys=[]): 
    # self.size = np.array([grid_dim, grid_dim, grid_dim], dtype=int)
    self.FILELIST = files
    self.sample_sizes = []
    self.total_entries = 0
    
    self.feature_keys = feature_keys
    self.channel_nr = len(feature_keys)
    self.label_key = label_key
    self.size = None

    for filename in self.FILELIST:
      if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")
      with h5py.File(filename, "r") as hdf: 
        # Set up the grid size
        if grid_dim is None and self.size is None: 
          # Read from the file
          self.size = np.array(hdf["featurizer_parms"]["dimensions"][:])
        else:
          if isinstance(grid_dim, (int, float)):
            self.size = np.array([grid_dim, grid_dim, grid_dim], dtype=int)
          elif isinstance(grid_dim, (list, tuple, np.ndarray)):
            self.size = np.array(grid_dim, dtype=int)
        # Check the existence of the feature keys in the file
        for k in feature_keys:
          if k not in hdf.keys():
            raise KeyError(f"Feature key {k} is not in the h5 file: {filename}")
        if self.label_key not in [i for i in hdf.keys()]:
          self.label_key = None
          self.label_dtype = int
          # raise KeyError(f"Label key {label_key} is not in the h5 file: {filename}")
        else: 
          # Get the native dtype of the hdf label and set to numpy's dtype
          self.label_dtype = np.dtype(hdf[label_key].dtype)
          print(f"Label dtype: {self.label_dtype}")

        if self.label_key is None:
          label_nr = hdf[feature_keys[0]].shape[0]
        else: 
          label_nr = hdf[label_key].shape[0]
        if config.verbose(): 
          printit(f"Found {label_nr} labels in {filename}")
        self.sample_sizes.append(label_nr)
        self.total_entries += label_nr

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
    """
    Get the total number of entries in the dataset.
    """
    return self.total_entries
  
  def position(self, index):
    """
    Get the internal position (to the corresponding HDF file) of the entry at the global index in the dataset.

    Parameters
    ----------
    index : int
      The global index of the entry.
    """
    return self.position_map[index]
  
  def filename(self, index):
    """
    Get the HDF file name of the entry at the global index in the dataset.

    Parameters
    ----------
    index : int
      The global index of the entry.

    """
    return self.FILELIST[self.file_map[index]]

  def __getitem__(self, index):
    """
    Get the data and label of the entry at the global index in the dataset.

    Parameters
    ----------
    index : int
      The global index of the entry.

    """
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    
    data = torch.zeros([self.channel_nr] + self.size.tolist(), dtype=torch.float32)
    for i, key in enumerate(self.feature_keys):
      data[i] = torch.from_numpy(readdata(self.filename(index), key, self.position(index)))
    if self.label_key is None:
      label = torch.tensor([0])
    else: 
      label = torch.tensor([readdata(self.filename(index), self.label_key, self.position(index))])
    return data, label
  
  def mini_batch_task(self, index):
    """
    Get the tasks describing the location of the data and the label of the entry at the global index in the dataset.

    Parameters
    ----------
    index : int
      The global index of the entry.
    
    """
    tasks = []
    for i in range(len(self.feature_keys)):
      tasks.append((self.filename(index), self.feature_keys[i], np.s_[self.position(index)]))
    return tasks
  

  def mini_batches(self, batch_nr=None, batch_size=None, shuffle=True, process_nr=24, 
                   augment=False, augment_translation=0, augment_add_noise=False, **kwargs):
    """
    Built-in generator to iterate the dataset in mini-batches. 

    Parameters
    ----------
    batch_nr : int
      The number of batches to split the dataset into, mutually exclusive with batch_size. 
    batch_size : int
      The size of each batch, mutually exclusive with batch_nr.
    shuffle : bool
      Whether to shuffle the dataset or not, defaults to True.
    process_nr : int
      The number of processes to use for reading the data, defaults to 24.
    augment : bool
      Whether to augment the data or not, defaults to False. TODO: Implement data augmentation.
      If True is set, the following parameters are used.
    augment_translation : int
      The factor of translation for data augmentation, defaults to 0.
    augment_add_noise : bool
      Whether to add noise to the data or not, defaults to False.

    Yields
    ------
    tuple
      The data and label of the mini-batch.

    """
    indices = np.arange(self.total_entries)
    if shuffle:
      np.random.shuffle(indices)
    
    if batch_size is not None:
      batches = split_array(indices, batch_size)
    elif batch_nr is not None: 
      batch_size = self.total_entries // batch_nr
      batches = split_array(indices, batch_size)
    else:
      raise ValueError("Either batch_nr or batch_size should be specified")
  
    if config.verbose():
      printit(f"Iterating the dataset: {len(batches)} batches with batch size {batch_size}. Using {process_nr} processes.")
    
    result_shape = [-1, self.channel_nr] + self.size.tolist()
    
    with mp.Pool(process_nr) as pool:
      for batch in batches: 
        taskset = []
        for i in batch:
          taskset += self.mini_batch_task(i)
        results = pool.starmap(readdata, taskset)

        # Convert and reshape to the target numpy array 
        data_numpy = np.array(results, dtype=np.float32).reshape(result_shape)
        # Read the labels
        if self.label_key is None:
          labels = np.zeros(len(batch), dtype=np.float32)
        else:
          labels = pool.starmap(readdata, [(self.filename(i), self.label_key, self.position(i)) for i in batch])
        labels_numpy = np.array(labels, dtype=self.label_dtype).reshape(-1, 1)

        data = torch.from_numpy(data_numpy)
        label = torch.from_numpy(labels_numpy)

        if augment: 
          # TODO: implement it if data augment is needed 
          data = data_augment(data, translation_factor=augment_translation, add_noise=augment_add_noise)
        yield data, label
