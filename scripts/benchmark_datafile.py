import time, os, json, argparse
from nearl.io.dataset import Dataset

def benchmark_dataset_method(dset, exit_point=None, **kwargs):
  st = time.perf_counter()
  st_total = time.perf_counter()
  print(f"Started benchmarking ...")
  data_size = 0
  put_to_cuda = kwargs.get("tocuda", 0)
  batch_size = kwargs.get("batch_size", 256)
  process_nr = kwargs.get("process_nr", 12)
  verbose = kwargs.get("verbose", 0)
  
  batch_nr = (len(dset) + batch_size - 1) // batch_size
  if batch_nr * batch_size != len(dset):
    print(f"Data size is not aligned with batch size. {len(dset)} samples are extracted in total.")
    _exit_point = batch_nr - 1     # Skipt the last batch because the size is not aligned. 
  else:
    _exit_point = batch_nr
  exit_point = min(_exit_point, exit_point)  
  print(f"Benchmark Summary: batch_size {batch_size}, batch number {batch_nr}, process number {process_nr}, exit point {exit_point}.")

  for idx, datai in enumerate(dset.mini_batches(batch_size=batch_size, shuffle=True, process_nr=process_nr)): 
    data, label = datai
    if verbose:
      print(f"Batch {idx+1:5} used {time.perf_counter()-st:6.4f} s. {data.shape}")
    data_size += data.nbytes
    data_size += label.nbytes
    if put_to_cuda: 
      data.cuda()
      label.cuda()
    st = time.perf_counter()
    if (idx == exit_point-1): 
      print(f"Estimated total extraction time: {(time.perf_counter()-st_total+1e-8)/idx*batch_nr:6.2f} s. ")
      time_elapsed = time.perf_counter()-st_total
      through_put = batch_size * (idx+1) / time_elapsed       # Unit: sample/s
      throughput_per_core = through_put / process_nr          # Unit: sample/(core*s)
      digit_thoughput = data_size * 8 / time_elapsed / 1e6    # Unit: Mbps (Megabits per second)
      total_size = data_size / 1024 / 1024                    # Unit: MB (Megabytes)
      
      print(f"Data_size: {total_size:6.3f} MB; Time_elapse {time_elapsed:6.3f} s; Retrieval_rate: {digit_thoughput:6.3f} Mbps; Throughput: {through_put:6.3f} samples/s; Throughput_per_core: {throughput_per_core:6.3f} samples/(core*s);")
      break

def argument_parser():
  parser = argparse.ArgumentParser(description="Benchmarking the data extraction speed of the HDF5 dataset.")
  parser.add_argument("-f", '--input-file', type=str, help="Input file path.")
  parser.add_argument("-b", "--batch-size", default=128, type=int, help="Batch size.")
  parser.add_argument("-p", "--process-nr", default=8, type=int, help="Number of processes.")
  parser.add_argument("-e", "--exit-point", default=999999, type=int, help="Exit point.")
  parser.add_argument("-c", "--tocuda", default=0, type=int, help="Transfer data to cuda.") 
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose mode.")
  parser.add_argument("-t", "--tags", default=None, type=str, help="Tags for the output file.")
  parser.add_argument("-l", "--label-tag", default="label", type=str, help="Tag of the label.")

  args = parser.parse_args()

  # Tags check
  args.tags_list = args.tags.strip("%").split("%")
  if len(args.tags_list) == 0: 
    raise ValueError("Tags are not specified. Each tag should be separated by '%'.")

  # Input file check 
  if args.input_file is None: 
    raise ValueError("Input file is not specified.")
  if os.path.exists(args.input_file) is False: 
    raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
  
  return args

def console_interface(): 
  args = argument_parser()
  settings = vars(args)

  workernr = settings["process_nr"]
  batch_size = settings["batch_size"]
  input_file = settings["input_file"]
  exit_point = settings["exit_point"]
  print(json.dumps(settings, indent=2))
  if input_file.endswith(".h5") is True: 
    files = [input_file]
  else: 
    with open(input_file, "r") as f: 
      files = f.read().strip("\n").split("\n")
  print(f"Files: {files}")

  dset = Dataset(files, feature_keys = settings["tags_list"], label_key = settings["label_tag"])

  benchmark_dataset_method(dset, batch_size=batch_size, process_nr=workernr, exit_point=exit_point, tocuda=settings["tocuda"], verbose=settings["verbose"])

if __name__ == "__main__": 
  """
  Benchmarking the data extraction speed of the HDF5 dataset.

  Example:
    python /MieT5/BetaPose/scripts/benchmark_datafile.py -f MisatoOutput3.h5 -b 64 -t prot_NCount_obs%prot_OCount_obs -l label_pcdt
  """
  console_interface()

  
