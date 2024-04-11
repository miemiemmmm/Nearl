import os, sys, argparse, time, random, json

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import nearl
from nearl.io.dataset import Dataset

writer = SummaryWriter("/Matter/nearl_tensorboard")

def compute_correlations(labels, pred):
  # Convert tensors to NumPy arrays if they are PyTorch tensors
  if isinstance(labels, torch.Tensor):
      labels = labels.detach().cpu().numpy()
  if isinstance(pred, torch.Tensor):
      pred = pred.detach().cpu().numpy()
  labels = labels.flatten()
  pred = pred.flatten()
  # Compute Pearson's R
  pearson_corr, _ = scipy.stats.pearsonr(labels, pred)
  # Compute Spearman's rho
  spearman_corr, _ = scipy.stats.spearmanr(labels, pred)
  return pearson_corr, spearman_corr

def parse_args(): 
  parser = argparse.ArgumentParser(description="Train VoxNet")
  parser.add_argument("-train", "--training_data", type=str, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test", "--test_data", type=str, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output_folder", type=str, default="/tmp/", help="Output folder")
  
  # Pretrained model and break point restart
  parser.add_argument("--model", type=str, help="Model to use")
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")
  
  # Training parameters
  parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")

  # Other parameters
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("-w", "--data_workers", type=int, default=4, help="Number of workers for data loading")
  parser.add_argument("--manualSeed", type=int, help="Manually set seed")

  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")

  args = parser.parse_args()
  if not os.path.exists(args.training_data):
    raise FileNotFoundError(f"The training data file {args.training_data} does not exist")
  if not os.path.exists(args.test_data):
    raise FileNotFoundError(f"The test data file {args.test_data} does not exist")
  if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)

  return args


def perform_training(training_settings: dict): 
  USECUDA = training_settings["cuda"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])

  # TODO modify the dataset to use
  datatags = training_settings["datatags"]
  labeltag = training_settings["labeltag"]
  dimensions = training_settings["dimensions"]
  
  st = time.perf_counter()
  # Load the datasets
  trainingfiles = nearl.utils.check_filelist(training_settings["training_data"])
  testfiles     = nearl.utils.check_filelist(training_settings["test_data"])
  # Load the data.
  # NOTE: In Voxel based training, the bottleneck is the SSD data reading.
  # NOTE: Putting the dataset to SSD will significantly speed up the training.
  
  training_data = Dataset(trainingfiles, dimensions,
    feature_keys = datatags,
    label_key = labeltag,
  )
  test_data = Dataset(testfiles, dimensions,
    feature_keys = datatags,
    label_key = labeltag,
  )
  modelname = training_settings["model"]
  model = nearl.utils.get_model(training_settings["model"], len(datatags), 1, dimensions)
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    model.load_state_dict(torch.load(training_settings["pretrained"]))
  if USECUDA:
    model.cuda()

  optimizer = optim.Adam(model.parameters(), lr=training_settings["learning_rate"], betas=(0.9, 0.999))
  # criterion = nn.CrossEntropyLoss()  # For classification
  criterion = nn.MSELoss() 
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  
  parameter_number = sum([p.numel() for p in model.parameters()])
  training_settings["parameter_number"] = parameter_number
  print(f"Number of parameters: {parameter_number}")

  START_EPOCH = training_settings["start_epoch"]
  EPOCH_NR = training_settings["epochs"]
  BATCH_SIZE = training_settings["batch_size"]
  WORKER_NR = training_settings["data_workers"]
  for epoch in range(START_EPOCH, EPOCH_NR):
    st = time.perf_counter()
    message = f" Running the epoch {epoch+1 : >4d}/{EPOCH_NR : <4d} at {time.ctime()} "
    print(f"{message:#^80}")
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    _loss_train_cache = []
    _loss_test_cache = []
    _rmse_train_cache = []
    _rmse_test_cache = []
    _pearson_train_cache = []
    _pearson_test_cache = []
    _spearman_train_cache = []
    _spearman_test_cache = []
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      # if (batch_idx+1) % 300 == 0:   # TODO: Remove this line when production
      #   break
      train_data, train_label = batch
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()
      optimizer.zero_grad()
      model = model.train()
      pred = model(train_data)
      # print(pred.shape, train_label.shape)   # torch.Size([1024, 1]) torch.Size([1024, 1])
      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()

      # Measure the RMSE of the training set and one batch of the test set
      if (batch_idx+1) % (batch_nr//10) == 0:
        # NOTE: Evaluating accuracy every batch will casue a significant slow down.
        with torch.no_grad():
          # Measure the RMSE of the training set and one batch of the test set
          trdata, trlabel = next(training_data.mini_batches(batch_size=1000, process_nr=WORKER_NR))
          if USECUDA:
            trdata, trlabel = trdata.cuda(), trlabel.cuda()
          pred = model(trdata)
          trloss = criterion(pred, trlabel)
          trrmse = (trlabel - pred).pow(2).mean().sqrt().item()
          tr_pearson, tr_spearman = compute_correlations(trlabel, pred)

          tedata, telabel = next(test_data.mini_batches(batch_size=1000, process_nr=WORKER_NR))
          if USECUDA:
            tedata, telabel = tedata.cuda(), telabel.cuda()
          pred = model(tedata)
          teloss = criterion(pred, telabel)
          termse = (telabel - pred).pow(2).mean().sqrt().item()
          
          # te_spearman = np.corrcoef(telabel.cpu().numpy().flatten(), pred.cpu().numpy().flatten())[0, 1]
          te_pearson, te_spearman = compute_correlations(telabel, pred)
          writer.add_scalar(f"Performance/train_loss", trloss, epoch*batch_nr+batch_idx)
          writer.add_scalar(f"Performance/test_loss", teloss, epoch*batch_nr+batch_idx)
          writer.add_scalar(f"Performance/train_RMSE", trrmse, epoch*batch_nr+batch_idx)
          writer.add_scalar(f"Performance/test_RMSE", termse, epoch*batch_nr+batch_idx)
          writer.add_scalar(f"Correlation/trainR", tr_pearson, epoch*batch_nr+batch_idx)
          writer.add_scalar(f"Correlation/testR", te_pearson, epoch*batch_nr+batch_idx)
          writer.add_scalar(f"Correlation/trainRHO", tr_spearman, epoch*batch_nr+batch_idx)
          writer.add_scalar(f"Correlation/testRHO", te_spearman, epoch*batch_nr+batch_idx)
          _loss_test_cache.append(teloss.cpu().numpy())
          _loss_train_cache.append(trloss.cpu().numpy())
          _rmse_train_cache.append(trrmse)
          _rmse_test_cache.append(termse)
          _pearson_train_cache.append(tr_pearson)
          _pearson_test_cache.append(te_pearson)
          _spearman_train_cache.append(tr_spearman)
          _spearman_test_cache.append(te_spearman)
        epoch_info = f"{epoch+1}/{EPOCH_NR}:{batch_idx+1}/{batch_nr}"
        time_left = (time.perf_counter() - st) / batch_idx * (batch_nr - batch_idx)
        msg = f"Epoch {epoch_info:>15}: RMSE: {termse:.4f}/{trrmse:.4f}; Loss: {teloss:.4f}/{trloss:.4f}; Corr: {te_pearson:.4f}/{tr_pearson:.4f}; RHO: {te_spearman:.4f}/{tr_spearman:.4f}; Time left: {time_left:.0f} seconds"
        print(f"{msg}")
    scheduler.step()

    # Save the model
    model_name = "Model"+training_settings["model"]+f"_{len(datatags)}_1_{dimensions}_{epoch}.pth"
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), model_name)
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(model.state_dict(), modelfile_output)

    # Save the performance to a HDF5 file
    perf_file = os.path.join(training_settings["output_folder"], "performance.h5")
    nearl.utils.update_hdf_data(perf_file, "train_loss", np.array([np.mean(_loss_train_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))
    nearl.utils.update_hdf_data(perf_file, "test_loss", np.array([np.mean(_loss_test_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))
    nearl.utils.update_hdf_data(perf_file, "train_rmse", np.array([np.mean(_rmse_train_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))
    nearl.utils.update_hdf_data(perf_file, "test_rmse", np.array([np.mean(_rmse_test_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))
    nearl.utils.update_hdf_data(perf_file, "train_pearson", np.array([np.mean(_pearson_train_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))
    nearl.utils.update_hdf_data(perf_file, "test_pearson", np.array([np.mean(_pearson_test_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))
    nearl.utils.update_hdf_data(perf_file, "train_spearman", np.array([np.mean(_spearman_train_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))
    nearl.utils.update_hdf_data(perf_file, "test_spearman", np.array([np.mean(_spearman_test_cache)], dtype=np.float32), np.s_[epoch:epoch+1], dtype=np.float32, maxshape=(None, ))

  print("Running final inference on training set ...")
  predictions = torch.full((len(training_data), 1), 0.0, dtype=torch.float32)
  labels = torch.full((len(training_data), 1), 0.0, dtype=torch.float32)
  c = 0
  with torch.no_grad():
    for data, label in training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR):
      if USECUDA:
        data, label = data.cuda(), label.cuda()
      pred = model(data)
      predictions[c:c+len(data)] = pred
      labels[c:c+len(data)] = label
      c += len(data)
    trloss = criterion(predictions, labels)
    trrmse = (labels - predictions).pow(2).mean().sqrt().item()
    tr_pearson, tr_spearman = compute_correlations(predictions.cpu().numpy(), labels.cpu().numpy())
  print(f"Final training set performance: RMSE: {trrmse:.4f}; Loss: {trloss:.4f}; Corr: {tr_pearson:.4f}; RHO: {tr_spearman:.4f}")


  print("Running final inference on test set ...")
  predictions = torch.full((len(test_data), 1), 0.0, dtype=torch.float32)
  labels = torch.full((len(test_data), 1), 0.0, dtype=torch.float32)
  c = 0
  with torch.no_grad():
    for data, label in test_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR):
      if USECUDA:
        data, label = data.cuda(), label.cuda()
      pred = model(data)
      predictions[c:c+len(data)] = pred
      labels[c:c+len(data)] = label
      c += len(data)
      # loss.append(criterion(pred, label).cpu().numpy())
    teloss = criterion(predictions, labels)
    termse = (labels - predictions).pow(2).mean().sqrt().item()
    te_pearson, te_spearman = compute_correlations(predictions.cpu().numpy(), labels.cpu().numpy())
  print(f"Final test set performance: RMSE: {termse:.4f}; Loss: {teloss:.4f}; Corr: {te_pearson:.4f}; RHO: {te_spearman:.4f}")

  # Add some results to the training configuration
  training_settings["results"] = {
    "train_loss": trloss.item(),
    "train_rmse": trrmse,
    "train_pearson": tr_pearson,
    "train_spearman": tr_spearman,
    "test_loss": teloss.item(),
    "test_rmse": termse,
    "test_pearson": te_pearson,
    "test_spearman": te_spearman,
  }

  # Test the ranking spearman correlation
  _training_data = Dataset(trainingfiles, dimensions,
    feature_keys = datatags,
    label_key = "pk_original",
  )
  _test_data = Dataset(testfiles, dimensions,
    feature_keys = datatags,
    label_key = "pk_original",
  )
  predictions = torch.full((len(_training_data), 1), 0.0, dtype=torch.float32)
  labels = torch.full((len(_training_data), 1), 0.0, dtype=torch.float32)
  c = 0
  with torch.no_grad():
    for data, label in _training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR):
      if USECUDA:
        data, label = data.cuda(), label.cuda()
      pred = model(data)
      predictions[c:c+len(data)] = pred
      labels[c:c+len(data)] = label
      c += len(data)
    trloss = criterion(predictions, labels)
    trrmse = (labels - predictions).pow(2).mean().sqrt().item()
    tr_pearson, tr_spearman = compute_correlations(predictions.cpu().numpy(), labels.cpu().numpy())
  
  predictions = torch.full((len(_test_data), 1), 0.0, dtype=torch.float32)
  labels = torch.full((len(_test_data), 1), 0.0, dtype=torch.float32)
  c = 0
  with torch.no_grad():
    for data, label in _test_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR):
      if USECUDA:
        data, label = data.cuda(), label.cuda()
      pred = model(data)
      predictions[c:c+len(data)] = pred
      labels[c:c+len(data)] = label
      c += len(data)
      # loss.append(criterion(pred, label).cpu().numpy())
    teloss = criterion(predictions, labels)
    termse = (labels - predictions).pow(2).mean().sqrt().item()
    te_pearson, te_spearman = compute_correlations(predictions.cpu().numpy(), labels.cpu().numpy())
  
    training_settings["results"]["train_loss_original"] = trloss.item()
    training_settings["results"]["train_rmse_original"] = trrmse
    training_settings["results"]["train_pearson_original"] = tr_pearson
    training_settings["results"]["train_spearman_original"] = tr_spearman
    training_settings["results"]["test_loss_original"] = teloss.item()
    training_settings["results"]["test_rmse_original"] = termse
    training_settings["results"]["test_pearson_original"] = te_pearson
    training_settings["results"]["test_spearman_original"] = te_spearman
    print(training_settings)
    

if __name__ == "__main__":
  args = parse_args()
  SETTINGS = vars(args)

  SETTINGS["datatags"] = [
    "atomtype_hydrogen", "atomtype_carbon", 
    "atomtype_nitrogen", "atomtype_oxygen",
    "atomtype_sulfur", 
    # "aromaticity",
    # "atomic_number", "mass",

    # "ligand_annotation", "protein_annotation", 
    "obs_density_mass", "obs_distinct_atomic_number",
    # "obs_distinct_resid", 

    # "partial_charge_negative", "partial_charge_positive", 
  ]
  SETTINGS["labeltag"] = "label"
  SETTINGS["dimensions"] = 32

  _SETTINGS = json.dumps(SETTINGS, indent=2)
  print("Settings of this training:")
  print(_SETTINGS)
  
  st = time.perf_counter()
  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)

  perform_training(SETTINGS)

  # Potentially added some more configurations of the training job
  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(json.dumps(SETTINGS, indent=2))
  
  print(SETTINGS)

  print(f"Training finished, time elapsed: {time.perf_counter() - st:.2f}")


# variables:
# 	float aromaticity(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float atomic_number(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float atomtype_carbon(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float atomtype_hydrogen(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float atomtype_nitrogen(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float atomtype_oxygen(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float atomtype_sulfur(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float label(phony_dim_0) ;
# 	float ligand_annotation(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float mass(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float obs_density_mass(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float obs_distinct_atomic_number(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float obs_distinct_resid(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float partial_charge_negative(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float partial_charge_positive(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
# 	float pk_original(phony_dim_0) ;
# 	float protein_annotation(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;