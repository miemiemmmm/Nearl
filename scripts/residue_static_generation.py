import time, os
import numpy as np
import pandas as pd 
import multiprocessing as mp
import feater 
import feater.voxelize
import feater.utils

# Read PDB coordinates
FEATER_DATASET = "/Weiss/FEater_Dual_PDBHDF/TrainingSet_Dataset.h5"
TARGET_NR = 1000
CLASS_NR = 400
CPU_NR = 16
BATCH_NR = 1024
START_BATCH = 0
OUTPUT_DIR = "/Matter/nearl_single_static"
WEIGHT_TYPE="mass"  # Types "mass" or "radius"


# 4 foulders for static/dynamic and single/dual dataset. 
# /Matter/nearl_dual_dynamic
# /Matter/nearl_dual_static
# /Matter/nearl_single_dynamic
# /Matter/nearl_single_static

###############################################################################
np.random.seed(1543)
SUMCHECK = 718479870712
MASS2RADII = {
  16: 1.52,  # O
  12: 1.70,  # C
  14: 1.55,  # N
  1: 1.20,   # H
  32: 1.80,  # S
  31: 1.80,  # P
  0: 1.40,   # None
}
listfile = os.path.join(OUTPUT_DIR, "index_label.csv")
h5outfile = os.path.join(OUTPUT_DIR, "voxel.h5")


# Check each tag and select the corresponding data
total_nr = 0
selected_final = []
selected_labels = []
selected_restypes = []
with feater.io.hdffile(FEATER_DATASET, "r") as hdffile: 
  labels = np.array(hdffile["label"])
  for tagi in range(CLASS_NR): 
    locations = np.where(labels == tagi)
    locations = locations[0]
    if len(locations) < TARGET_NR:
      _target_nr = len(locations)
      print(f"Setting the target number to {len(locations)} because of the small number of samples")
    else:
      _target_nr = TARGET_NR
    selected = np.random.choice(locations, _target_nr, replace=False)
    total_nr += _target_nr
    selected.sort()
    selected_final += selected.tolist()
    selected_labels += [tagi] * _target_nr
    selected_restypes += [feater.LAB2RES_DUAL[tagi]] * _target_nr

if SUMCHECK != np.sum(selected_final): 
  print(f"Sum check failed {np.sum(selected_final)} != {SUMCHECK}")
  exit(1)
else:
  print(f"Selected {len(selected_final)} entries for the static dataset for {CLASS_NR} classes")

# Write out the list of residues for the test 
st = time.perf_counter()
df = pd.DataFrame({
    "label": selected_labels,
    "index": selected_final,
    "residue_type": selected_restypes
  }
)
df.to_csv(listfile, index=False)

def process_residue(hdffile, index, settings): 
  with feater.io.hdffile(hdffile, "r") as hdf: 
    crd_start = hdf["coord_starts"][index]
    crd_end = hdf["coord_ends"][index]
    coord = hdf["coordinates"][crd_start: crd_end]
    # TODO: weight could be atom radii and direct atom mass
    weights = hdf["elements"][crd_start: crd_end] # Get atom mass for determination of atom type
    # Direct atom mass 
    if WEIGHT_TYPE == "mass":
      weights = np.array(weights, dtype=np.float64)
    elif WEIGHT_TYPE == "radius":
      weights = np.array([MASS2RADII[i] for i in weights], dtype=np.float64)
    
    label = hdf["label"][index]

  dims = np.array(settings["dims"], dtype=int)
  boxsize = float(settings["boxsize"])
  spacing = float(boxsize / dims[0])
  cutoff = float(settings["cutoff"])
  sigma = float(settings["sigma"])
    
  ret_data = feater.voxelize.interpolate(coord, weights, dims, spacing = spacing, cutoff = cutoff, sigma = sigma)
  ret_data = ret_data.reshape(dims)
  return ret_data


# Featurization metadata
VOX_SETTINGS = {
  "dims": [32, 32, 32],
  "boxsize": 16.0,
  "cutoff": 8.0,
  "sigma": 1.5
}

if START_BATCH == 0:
  with feater.io.hdffile(h5outfile, "w") as f: 
    feater.utils.add_data_to_hdf(f, "dimensions", VOX_SETTINGS["dims"], dtype=np.int32, maxshape=[3])  
    feater.utils.add_data_to_hdf(f, "cutoff", np.array([VOX_SETTINGS["cutoff"]], dtype=np.float32), maxshape=[1])
    feater.utils.add_data_to_hdf(f, "sigma", np.array([VOX_SETTINGS["sigma"]], dtype=np.float32), maxshape=[1])
    feater.utils.add_data_to_hdf(f, "boxsize", np.array([VOX_SETTINGS["boxsize"]], dtype=np.float32), maxshape=[1])
else: 
  with feater.io.hdffile(h5outfile, "a") as f: 
    if "dimensions" not in f.keys():
      feater.utils.add_data_to_hdf(f, "dimensions", VOX_SETTINGS["dims"], dtype=np.int32, maxshape=[3])  
    if "cutoff" not in f.keys():
      feater.utils.add_data_to_hdf(f, "cutoff", np.array([VOX_SETTINGS["cutoff"]], dtype=np.float32), maxshape=[1])
    if "sigma" not in f.keys():
      feater.utils.add_data_to_hdf(f, "sigma", np.array([VOX_SETTINGS["sigma"]], dtype=np.float32), maxshape=[1])
    if "boxsize" not in f.keys():
      feater.utils.add_data_to_hdf(f, "boxsize", np.array([VOX_SETTINGS["boxsize"]], dtype=np.float32), maxshape=[1])


# Batch generation and batch saving of the static voxel features 
pool = mp.Pool(CPU_NR)
batches = np.array_split(selected_final, BATCH_NR)
label_batchs = np.array_split(selected_labels, BATCH_NR)
st = time.perf_counter()
for b in range(START_BATCH, len(batches)):
  batchi = batches[b]
  tasks = [(FEATER_DATASET, index, VOX_SETTINGS) for index in batchi]
  results = pool.starmap(process_residue, tasks)
  print(f"Finished batch {b+1}/{BATCH_NR} with {len(batchi)} residues; Used {time.perf_counter() - st:.2f} seconds")
  result_buffer = np.array(results, dtype=np.float32)
  label_buffer = label_batchs[b]
  print("Result: ", result_buffer.shape)
  print("Labels: ", label_buffer.shape)
  
  st = time.perf_counter() 
  # Dump the batch results to the disk 
  with feater.io.hdffile(h5outfile, "a") as f:
    feater.utils.add_data_to_hdf(f, "voxel", result_buffer, dtype=np.float32, chunks=True, maxshape=(None, 32, 32, 32), compression="gzip", compression_opts=4)
    feater.utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)

pool.close()
pool.join()
print("Done")



