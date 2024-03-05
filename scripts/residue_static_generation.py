import time, os, argparse, json
import numpy as np
import pandas as pd 
import multiprocessing as mp
import feater 
import feater.voxelize
import feater.utils

np.random.seed(1543)

MASS2RADII = {
  16: 1.52,  # O
  12: 1.70,  # C
  14: 1.55,  # N
  1: 1.20,   # H
  32: 1.80,  # S
  31: 1.80,  # P
  0: 1.40,   # None
}


def process_residue(hdffile, index, weight_type, settings): 
  with feater.io.hdffile(hdffile, "r") as hdf: 
    crd_start = hdf["coord_starts"][index]
    crd_end = hdf["coord_ends"][index]
    coord = hdf["coordinates"][crd_start: crd_end]
    weights = hdf["elements"][crd_start: crd_end] # Get atom mass for determination of atom type
    # Weight type to be: direct "atom mass" or "atom radii" 
    if weight_type == "mass":
      weights = np.array(weights, dtype=np.float64)
    elif weight_type == "radius":
      weights = np.array([MASS2RADII[i] for i in weights], dtype=np.float64)

  dims = np.array(settings["dims"], dtype=int)
  boxsize = float(settings["boxsize"])
  spacing = float(boxsize / dims[0])
  cutoff = float(settings["cutoff"])
  sigma = float(settings["sigma"])
    
  ret_data = feater.voxelize.interpolate(coord, weights, dims, spacing = spacing, cutoff = cutoff, sigma = sigma)
  ret_data = ret_data.reshape(dims)
  return ret_data


def split_dataset(datafile, trainingset_nrs, output_folder): 
  # Initialize the list of selected residues
  selected_final = []
  selected_labels = []
  selected_restypes = []
  for tnr in trainingset_nrs:
    selected_final.append([])
    selected_labels.append([])
    selected_restypes.append([])
  
  # Read the HDF file and select the residues
  with feater.io.hdffile(datafile, "r") as hdffile: 
    labels = np.array(hdffile["label"])
    # Partition the labels by the expected residue number
    needed_nr = np.sum(trainingset_nrs)
    print("Needed number of labels: ", needed_nr)
    fractions = trainingset_nrs / needed_nr
    class_nr = len(np.unique(labels))
    
    # Main loop for the selection of the residues in each class
    for tagi in range(class_nr): 
      locations = np.where(labels == tagi)
      locations = locations[0]
      sample_bins = fractions * len(locations)
      sample_bins_cumsum = [0] + np.cumsum(sample_bins).astype(int).tolist() 
      
      if len(locations) > needed_nr:
        # In the case that the number of available residues is more than the target number: Randomly select the residues
        for bin_idx, target_nr in enumerate(trainingset_nrs):
          bini = locations[sample_bins_cumsum[bin_idx]:sample_bins_cumsum[bin_idx+1]]   # Elements fall into the each bin
          selected = np.random.choice(bini, target_nr, replace=False)
          selected.sort()
          selected_final[bin_idx] += selected.tolist()
          selected_labels[bin_idx] += [tagi] * target_nr
          selected_restypes[bin_idx] += [feater.LAB2RES_DUAL[tagi]] * target_nr
      else: 
        # In the case that the number of available residues is less than the target number: Just select all of them
        print(f"Class {tagi} has {len(locations)} residues; Selecting all of them ...")
        for bin_idx, target_nr in enumerate(trainingset_nrs):
          bini = locations[sample_bins_cumsum[bin_idx]:sample_bins_cumsum[bin_idx+1]]   # Elements fall into the each bin
          selected_final[bin_idx] += bini.tolist()
          selected_labels[bin_idx] += [tagi] * len(bini)
          selected_restypes[bin_idx] += [feater.LAB2RES_DUAL[tagi]] * len(bini)    

  if len(output_folder) > 0:
    # Write out the list of residues for the test 
    for i in range(len(trainingset_nrs)):
      df = pd.DataFrame(
        {
          "label": selected_labels[i],
          "index": selected_final[i],
          "residue_type": selected_restypes[i]
        }
      )
      csv_out = os.path.join(os.path.abspath(output_folder), f"dataset{i}.csv")
      print(f"Rank {i} has {len(selected_final[i])} entries; Writing to {csv_out} ...")
      df.to_csv(csv_out, index=False)
  else: 
    print(f"Rank {i} has {len(selected_final[i])} entries; Skipping the writing ...")


def parse_args():
  parser = argparse.ArgumentParser(description="Voxelization of the static dataset")
  parser.add_argument("-f", "--datasetfile", type=str, default="", help="Dataset to use")
  parser.add_argument("-c", "--cpu_nr", type=int, default=16, help="Number of CPUs to use")
  parser.add_argument("-b", "--batch_nr", type=int, default=1024, help="Number of batches to use")
  parser.add_argument("--start_batch", type=int, default=0, help="Start batch")
  parser.add_argument("-o", "--output_dir", type=str, default="/tmp/", help="Output directory")
  parser.add_argument("--weight_type", type=str, default="mass", help="Weight type")
  parser.add_argument("--target_nr", type=int, default=1000, help="Target number of residues")
  args = parser.parse_args()
  if len(args.datasetfile) == 0:
    raise ValueError("Dataset file is not given")
  return args


def process_dataset(csvfile, source_data, output_hdf, weight_type, start_batch, batch_nr, cpu_nr):
  df = pd.read_csv(csvfile)
  selected_final = df["index"].values.tolist()
  selected_labels = df["label"].values.tolist()
  if start_batch == 0:
    with feater.io.hdffile(output_hdf, "w") as f: 
      feater.utils.add_data_to_hdf(f, "dimensions", VOX_SETTINGS["dims"], dtype=np.int32, maxshape=[3])  
      feater.utils.add_data_to_hdf(f, "cutoff", np.array([VOX_SETTINGS["cutoff"]], dtype=np.float32), maxshape=[1])
      feater.utils.add_data_to_hdf(f, "sigma", np.array([VOX_SETTINGS["sigma"]], dtype=np.float32), maxshape=[1])
      feater.utils.add_data_to_hdf(f, "boxsize", np.array([VOX_SETTINGS["boxsize"]], dtype=np.float32), maxshape=[1])
  else:
    with feater.io.hdffile(output_hdf, "a") as f: 
      if "dimensions" not in f.keys():
        feater.utils.add_data_to_hdf(f, "dimensions", VOX_SETTINGS["dims"], dtype=np.int32, maxshape=[3])  
      if "cutoff" not in f.keys():
        feater.utils.add_data_to_hdf(f, "cutoff", np.array([VOX_SETTINGS["cutoff"]], dtype=np.float32), maxshape=[1])
      if "sigma" not in f.keys():
        feater.utils.add_data_to_hdf(f, "sigma", np.array([VOX_SETTINGS["sigma"]], dtype=np.float32), maxshape=[1])
      if "boxsize" not in f.keys():
        feater.utils.add_data_to_hdf(f, "boxsize", np.array([VOX_SETTINGS["boxsize"]], dtype=np.float32), maxshape=[1])
  
  # Initialize the pool for parallel processing
  pool = mp.Pool(cpu_nr)
  batches = np.array_split(selected_final, batch_nr)
  label_batchs = np.array_split(selected_labels, batch_nr)
  st = time.perf_counter()
  for b in range(start_batch, len(batches)):
    batchi = batches[b]
    tasks = [(source_data, index, weight_type, VOX_SETTINGS) for index in batchi]
    results = pool.starmap(process_residue, tasks)
    print(f"Finished batch {b+1}/{batch_nr} with {len(batchi)} residues; Used {time.perf_counter() - st:.2f} seconds")
    result_buffer = np.array(results, dtype=np.float32)
    label_buffer = label_batchs[b]
    
    st = time.perf_counter() 
    # Dump the batch results to the disk 
    with feater.io.hdffile(output_hdf, "a") as f:
      feater.utils.add_data_to_hdf(f, "voxel", result_buffer, dtype=np.float32, chunks=True, maxshape=(None, 32, 32, 32), compression="gzip", compression_opts=4)
      feater.utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)
  pool.close()
  pool.join()


###############################################################################
if __name__ == "__main__":
  args = parse_args()

  # Read PDB coordinates
  FEATER_DATASET = args.datasetfile
  OUTPUT_DIR = args.output_dir
  # FEATER_DATASET = "/Weiss/FEater_Dual_PDBHDF/TrainingSet_Dataset.h5"
  # OUTPUT_DIR = "/Matter/nearl_dual_static"
  TARGET_NRS = [1200, 500]


  # 4 foulders for static/dynamic and single/dual dataset. 
  # /Matter/nearl_dual_dynamic
  # /Matter/nearl_dual_static
  # /Matter/nearl_single_dynamic
  # /Matter/nearl_single_static


  CPU_NR = args.cpu_nr
  BATCH_NR = args.batch_nr           # Split the dataset into 1024 batches
  # NOTE: This scripts is using add_data_to_hdf; Always use 0 for new generation 
  START_BATCH = 0           # Index of the start batch (incase of a restart); 0 for new generation
  WEIGHT_TYPE = args.weight_type  # Types of weight to pass to cuda code: either "mass" or "radius"

  with open(os.path.join(OUTPUT_DIR, f"featurize_{WEIGHT_TYPE}.txt"), "w") as f:
    setting_string = json.dumps(vars(args), indent=2)
    print("Feature generation settings:")
    print(setting_string)
    f.write(setting_string)

  # Split the dataset
  split_dataset(FEATER_DATASET, TARGET_NRS, OUTPUT_DIR)

  # Featurization metadata
  VOX_SETTINGS = {
    "dims": [32, 32, 32],
    "boxsize": 16.0,
    "cutoff": 4.0,
    "sigma": 1.5
  }

  for group_idx in range(len(TARGET_NRS)): 
    csv_file = os.path.join(OUTPUT_DIR, f"dataset{group_idx}.csv")
    h5outfile = os.path.join(OUTPUT_DIR, f"voxel{group_idx}_{WEIGHT_TYPE}.h5")
    process_dataset(csv_file, FEATER_DATASET, h5outfile, WEIGHT_TYPE, START_BATCH, BATCH_NR, CPU_NR)

  # if START_BATCH == 0:
  #   with feater.io.hdffile(h5outfile, "w") as f: 
  #     feater.utils.add_data_to_hdf(f, "dimensions", VOX_SETTINGS["dims"], dtype=np.int32, maxshape=[3])  
  #     feater.utils.add_data_to_hdf(f, "cutoff", np.array([VOX_SETTINGS["cutoff"]], dtype=np.float32), maxshape=[1])
  #     feater.utils.add_data_to_hdf(f, "sigma", np.array([VOX_SETTINGS["sigma"]], dtype=np.float32), maxshape=[1])
  #     feater.utils.add_data_to_hdf(f, "boxsize", np.array([VOX_SETTINGS["boxsize"]], dtype=np.float32), maxshape=[1])
  # else: 
  #   with feater.io.hdffile(h5outfile, "a") as f: 
  #     if "dimensions" not in f.keys():
  #       feater.utils.add_data_to_hdf(f, "dimensions", VOX_SETTINGS["dims"], dtype=np.int32, maxshape=[3])  
  #     if "cutoff" not in f.keys():
  #       feater.utils.add_data_to_hdf(f, "cutoff", np.array([VOX_SETTINGS["cutoff"]], dtype=np.float32), maxshape=[1])
  #     if "sigma" not in f.keys():
  #       feater.utils.add_data_to_hdf(f, "sigma", np.array([VOX_SETTINGS["sigma"]], dtype=np.float32), maxshape=[1])
  #     if "boxsize" not in f.keys():
  #       feater.utils.add_data_to_hdf(f, "boxsize", np.array([VOX_SETTINGS["boxsize"]], dtype=np.float32), maxshape=[1])


  # # Batch generation and batch saving of the static voxel features 
  # pool = mp.Pool(CPU_NR)
  # batches = np.array_split(selected_final, BATCH_NR)
  # label_batchs = np.array_split(selected_labels, BATCH_NR)
  # st = time.perf_counter()
  # for b in range(START_BATCH, len(batches)):
  #   batchi = batches[b]
  #   tasks = [(FEATER_DATASET, index, VOX_SETTINGS) for index in batchi]
  #   results = pool.starmap(process_residue, tasks)
  #   print(f"Finished batch {b+1}/{BATCH_NR} with {len(batchi)} residues; Used {time.perf_counter() - st:.2f} seconds")
  #   result_buffer = np.array(results, dtype=np.float32)
  #   label_buffer = label_batchs[b]
  #   print("Result: ", result_buffer.shape)
  #   print("Labels: ", label_buffer.shape)
    
  #   st = time.perf_counter() 
  #   # Dump the batch results to the disk 
  #   with feater.io.hdffile(h5outfile, "a") as f:
  #     feater.utils.add_data_to_hdf(f, "voxel", result_buffer, dtype=np.float32, chunks=True, maxshape=(None, 32, 32, 32), compression="gzip", compression_opts=4)
  #     feater.utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)

  # pool.close()
  # pool.join()
  # print("Done")



