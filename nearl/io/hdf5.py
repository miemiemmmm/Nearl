import os
import h5py as h5 
import numpy as np 

"""
#################################
# Build a database from scratch #
#################################
from BetaPose.io import hdf5
from BetaPose.utils import utils; 
import numpy as np 
hdf = hdf5.hdf_operator("/tmp/test.h5", new=True)

# Complex 
datatype1 = [("atom_number", int), ("carbon_number", int), ("donor_number", int), ("acceptor_number", int),
             ("positive_charge", float), ("negative_charge", float), ("pseudo_lj", float), ("pseudo_elec", float),
             ("surface_area", float), ("volume", float), ("mean_radius", float), ("convex_ratio", float)]*6
data1 = np.zeros((1, 73)); 
data1 = hdf5.array2dataset(data1, [("repr_md5", "S32")] + [(f"repr{(i/7)+1}_{(i%7)+1}", float) for i in range(72)]); 
hdf.create_dataset("repr_form", data1); 
# hdf.append_entry("repr_form", np.zeros((16, 72))); 


datatype2 = [("ID",int), ("repr_md5", "S32"), ("fingerprint",float,(33,600))]
zero_fpfh = np.zeros((1, 33,600))
data2 = np.array([(0, utils.get_hash(), zero_fpfh)], dtype=object)
data2 = hdf5.array2dataset(data2, datatype2)
hdf.create_table("FPFH", data2, dtype=datatype2)


datatype3 = [("ID",int), ("repr_md5", "S32"), ("feature_mass",float,(12,12,12))]
feature_0 = np.zeros((1, 12,12,12))
data3 = np.array([(0, utils.get_hash(), feature_0)], dtype=object)
data3 = hdf5.array2dataset(data3, datatype3)
hdf.create_table("feature_mass", data3)

hdf.draw_structure()
hdf.close()
"""

REPR_TYPES = [("atom_number", int), ("carbon_number", int), ("donor_number", int), ("acceptor_number", int),
             ("positive_charge", float), ("negative_charge", float), ("pseudo_lj", float), ("pseudo_elec", float),
             ("surface_area", float), ("volume", float), ("mean_radius", float), ("convex_ratio", float)]

def array2dataset(data, dtypes): 
  """
  Convert a NumPy array to a structured array with specified data types
  """
  length = data.shape[0]; 
  template = np.zeros(length, dtype=dtypes)
  
  for i in range(len(template)): 
    template[i] = tuple(data[i])
  return template

class hdf_operator(h5.File):
  def __init__(self, filename, default_flag="r", read_only=False, append=False, new=False, overwrite=False):
    if read_only:
      print("Read-only mode enabled.")
      flag = "r";
    elif append:
      print(f"Appending to the file {filename}")
      flag = "a";
    elif overwrite or new:
      print(f"Force writing to the file {filename}")
      flag = "w"
    elif not os.path.isfile(filename):
      print(f"File {filename} does not exist, creating a new one.")
      flag = "w"
    else:
      print(f"Undefine reading mode, default to read-only.")
      flag = "r"
    super().__init__(filename, flag);

  # __enter__ and __exit__ are used to enable the "with" statement
  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def data(self, key):
    dset = self[key];
    return np.asarray(dset)
  def dtype(self,key):
    dset = self[key];
    return dset.dtype
    
  def list_datasets(self):
    """
    List all available datasets in the HDF5 file.
    """
    datasets = []
    def find_datasets(name, obj):
      if isinstance(obj, h5.Dataset):
        datasets.append(name)
    self.visititems(find_datasets)
    return datasets

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
    new_dataset = self.dump_dataset(f"{dataset_name}_temp", new_data)
    shape_after = new_dataset.shape; 
    # Copy attributes from the original dataset to the new one
    for key, value in dataset.attrs.items():
      new_dataset.attrs[key] = value

    # Remove the original dataset and rename the new one
    self.pop(dataset_name, None);
    self.move(f"{dataset_name}_temp", dataset_name);
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

  def dump_dataset(self, data_key, thedata, columns=[], **kwarg):
    theshape = np.asarray(thedata).shape;
    maxshape = [i for i in theshape];
    maxshape[0] = None;
    maxshape = tuple(maxshape); 
    dset = self.create_dataset(data_key, data=thedata, compression="gzip", maxshape=maxshape, **kwarg)
    print(f"Created Dataset: {data_key}"); 
    return dset

  def temporary_dump(self, datalist, filename):
    if ("a" in self.mode or "w" in self.mode):
      raise "Please check the read mode to make the HDF file is writable or not"
    dataobj = np.array(thedata, dtype=object);
    np.save(filename, dataobj)

  def create_heterogeneous(self, data_key, thedata, **kwarg):
    newgroup = self.create_group(data_key);
    for idx, data in enumerate(thedata):
      newgroup.create_dataset(f"{data_key}_{str(idx)}", data=data, **kwarg);
  def append_heterogeneous(self, data_key, thedata, **kwarg):
    if data_key in self:
      thegroup = self[data_key];
      current_datasets = len(thegroup)
    else:
      raise ValueError(f"Dataset {data_key} does not exist.")
    for idx, data in enumerate(thedata, start=current_datasets):
      thegroup.create_dataset(f"{data_key}_{idx}", data=data, **kwarg);

  def create_table(self, data_key, thedata, columns=[], **kwarg):
    if thedata.dtype.names: 
      names = [name.encode('utf-8') for name in thedata.dtype.names]
    else: 
      names = columns; 
    self.create_dataset(data_key, thedata, **kwarg); 
    if len(columns) > 0: 
      dset.attrs['columns'] = names

  def delete_dataset(self, dataset_name):
    if dataset_name in self.list_datasets():
      self.pop(dataset_name, None);
      print(f"Dataset '{dataset_name}' has been deleted.")
    else:
      print(f"Warning: Dataset '{dataset_name}' does not exist.")
  def delete_heterogeneous(self, group_name):
    if group_name in self:
      if isinstance(self[group_name], h5.Group):
        del self[group_name]
        print(f"Group '{group_name}' has been deleted.")
      else:
        print(f"Warning: '{group_name}' is not a group.")
    else:
      print(f"Warning: Group '{group_name}' does not exist.")

  def delete_entry(self, name):
    if name in self:
      del self[name]
      print(f"Entry '{name}' has been deleted.")
    else:
      print(f"Warning: Entry '{name}' does not exist.")
  
  def append_entry(self, dataset_name, newdata):
    dset = self[dataset_name];
    current_shape = dset.shape;
    # Calculate the new shape after appending the new data
    new_shape = (current_shape[0] + newdata.shape[0], *current_shape[1:])
    # Resize the dataset to accommodate the new data
    dset.resize(new_shape)
    # Append the new data to the dataset
    dset[current_shape[0]:new_shape[0]] = newdata;
    print(f"Appended {newdata.shape[0]} entries to {dataset_name}")

  def draw_structure(self, depth=0):
    """
    Draw the structure of the HDF file. By default, only display the first level of the file (depth=0).
    Args:
      depth (int): The depth of the file structure to display.
    """
    print("############## HDF File Structure ##############")
    def print_structure(name, obj):
      current_depth = name.count('/');
      if current_depth > depth:  # If the current item's depth exceeds the limit, return early
        return
      if isinstance(obj, h5.Group):
        print(f"$ /{name:20s}/Heterogeneous Group");
      else:
        print(f"$ /{name:20s}: Shape-{obj.shape}");
    self.visititems(print_structure);
    print("############ END HDF File Structure ############")
    
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


def temporary_dump(datalist, filename):
  dataobj = np.array(datalist, dtype=object);
  np.save(filename, dataobj)
