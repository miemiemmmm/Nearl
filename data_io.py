import os
import h5py as h5 
import numpy as np 

class hdf_operator:
  def __init__(self, filename, new=False): 
    if new or (not os.path.isfile(filename)): 
      self.hdffile = h5.File(filename, "w")
    else: 
      self.hdffile = h5.File(filename, "a")
    
  def list_datasets(self):
    """
    List all available datasets in the HDF5 file.
    """
    datasets = []
    def find_datasets(name, obj):
      if isinstance(obj, h5.Dataset):
        datasets.append(name)
    self.hdffile.visititems(find_datasets)
    return datasets
  
  def data(self, key):
    dset = self.hdffile[key]
    return dset
  def close(self):
    self.hdffile.close()
  
  def column_names(self, dataset_name):
    """
    Get the column names of a dataset.
    Args:
    dataset_name (str): The name of the dataset.
    Returns:
    list of str: A list of column names for the specified dataset.
    """
    dataset = self.data(dataset_name)
    if isinstance(dataset.dtype, np.dtype):
      return dataset.dtype.names
    else:
      raise ValueError(f"{dataset_name} is not a structured dataset with column names.")
  
  def mask_entries(self, dataset_name, boolean_mask):
    """
    Delete a set of entries in a given dataset using a boolean array.
    NOTE: might be slow when dealing with ultra large files
    Args:
    dataset_name (str): The name of the dataset.
    boolean_mask (array-like): A boolean array indicating which entries to keep.
    """
    boolean_mask = np.bool_(boolean_mask)
    if dataset_name not in self.list_datasets(): 
      raise Exception(f"Not found any dataset named {dataset_name}")
      
    # Retrieve the dataset
    dataset = self.data(dataset_name)
    shape_before = dataset.shape; 
    
    # Create a new dataset without the specified entries
    new_data = dataset[:][boolean_mask]
    new_dataset = self.create_dataset(f"{dataset_name}_temp", new_data)
    shape_after = new_dataset.shape; 
    # Copy attributes from the original dataset to the new one
    for key, value in dataset.attrs.items():
      new_dataset.attrs[key] = value

    # Remove the original dataset and rename the new one
    self.hdffile.pop(dataset_name, None);
    self.hdffile.move(f"{dataset_name}_temp", dataset_name); 
    print(f"Successfully masked {np.count_nonzero(boolean_mask)} entries; Shape: {shape_before} -> {shape_after}")
    
  def remove_entry(self, dataset_name, index):
    """
    Remove an entry from the specified dataset by index.

    Args:
    dataset_name (str): The name of the dataset.
    index (int): The index of the entry to remove.
    """
    dataset = self.data(dataset_name)
    shape_before = dataset.shape; 
    data = np.asarray(dataset)
    data = np.delete(data, index, axis=0)
    # Resize the dataset and overwrite with the new data
    dataset.resize(data.shape)
    dataset[...] = data
    shape_after = dataset.shape; 
    print(f"Successfully Delete the entry {index}; Shape: {shape_before} -> {shape_after}")

  def create_dataset(self, data_key, thedata, columns=[], **kwarg): 
    theshape = thedata.shape; 
    maxshape = [i for i in theshape]; 
    maxshape[0] = None; 
    maxshape = tuple(maxshape); 
    dset = self.hdffile.create_dataset(data_key, data=thedata, compression="gzip", maxshape=maxshape, **kwarg)
    print(f"Created Dataset: {data_key}"); 
    return dset
  
  def create_table(self, data_key, thedata, columns=[], **kwarg): 
    if thedata.dtype.names: 
      names = [name.encode('utf-8') for name in thedata.dtype.names]
    else: 
      names = columns; 
    self.create_dataset(data_key, thedata, **kwarg); 
    if len(columns) > 0: 
      dset.attrs['columns'] = names
      
  def delete_dataset(self, dataset_name):
    if dataset_name in self.hdffile:
      self.hdffile.pop(dataset_name, None); 
      print(f"Dataset '{dataset_name}' has been deleted.")
    else:
      print(f"Warning: Dataset '{dataset_name}' does not exist.")

  
  def append_entry(self, dataset_name, newdata):
    dset = self.data(dataset_name);
    current_shape = dset.shape;
    # Calculate the new shape after appending the new data
    new_shape = (current_shape[0] + newdata.shape[0], *current_shape[1:])
    # Resize the dataset to accommodate the new data
    dset.resize(new_shape)
    # Append the new data to the dataset
    dset[-newdata.shape[0]:] = newdata
    print(f"Appended {newdata.shape[0]} entries to {dataset_name}")
    
  def draw_structure(self):
    print("####### HDF File Structure #######")
    def print_structure(name, obj):
      if isinstance(obj, h5.Group):
        print(f"$ /{name:10s}/")
      else:
        print(f"$ /{name:10s}: Shape-{obj.shape}")
    self.hdffile.visititems(print_structure)
    print("##### END HDF File Structure #####")
    
  def alter_entry(self, dataset_name, index, new_data):
    """
    Alter a specific entry in the dataset by index.
    Args:
      dataset_name (str): The name of the dataset.
      index (int): The index of the entry to be modified.
      new_data (tuple): The new data to replace the existing entry.
    """
    dataset = self.data(dataset_name)
    if 0 <= index < len(dataset):
      dataset[index] = new_data
    else:
      print(f"Index {index} is out of range for dataset '{dataset_name}'.")
