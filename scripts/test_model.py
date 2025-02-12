"""
Perform benchmarking on the pre-trained models. 
"""

import os, sys, time, io
import argparse, random, json 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr, spearmanr

# Import my models 
import nearl
import nearl.utils, nearl.io, nearl.models

tensorboard_writer = None 


LOSS_FUNCTIONS = {
  "crossentropy": nn.CrossEntropyLoss, 
  "mse": nn.MSELoss, 
  "mae": nn.L1Loss, 
  "l1": nn.L1Loss, 
  "huber": nn.HuberLoss,
}


def parse_args():
  parser = argparse.ArgumentParser(description="Train the models used in the FEater paper. ")

  # Input data files 
  # parser.add_argument("-train", "--training-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test",  "--test-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-p",     "--pretrained", type=str, required=True, help="Pretrained model path")
  parser.add_argument("-m",     "--meta-information", type=str, required=True, help="The meta information file for the pretrained model")
  parser.add_argument("-o",     "--output_folder", type=str, default="/tmp/", help="The output folder to store the model and performance data")
  parser.add_argument("-w",     "--data-workers", type=int, default=12, help="Number of workers for data loading")
  
  # Miscellanenous
  parser.add_argument("--test-number", type=int, default=100_000_000, help="Number of test samples")
  parser.add_argument("--test-robustness", type=int, default=0, help="Test the robustness of the model")
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("-s", "--manualSeed", type=int, help="Manually set seed")
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--production", type=int, default=0, help="Production mode") 

  args = parser.parse_args()  
  # if not os.path.exists(args.training_data): 
  #   raise ValueError(f"The training data file {args.training_data} does not exist.")
  if not os.path.exists(args.test_data):
    raise ValueError(f"The test data file {args.test_data} does not exist.")
  
  if not os.path.exists(args.meta_information):
    raise ValueError(f"The meta data file {args.meta_information} does not exist.")

  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)
  
  return args



def perform_testing_API(settings: dict, return_model=False, silent=False): 
  """
  The variable "test_data" and "pretrained" should be in the settings.
  """
  if not settings.get("test_data", None):
    raise ValueError("The test data file is not provided.")
  elif not settings.get("pretrained", None):
    raise ValueError("The pretrained model is not provided.") 
  elif not settings.get("meta_information", None):
    raise ValueError("The meta information file is not provided.")

  with open(settings["meta_information"], "r") as f:
    meta_information = json.load(f)
    print("The original model used the test dataset:", meta_information["test_data"])
  
  meta_information.update(settings)

  USECUDA = meta_information["cuda"]
  MODEL_TYPE = meta_information["model"]
  BATCH_SIZE = meta_information["batch_size"]
  WORKER_NR = meta_information["data_workers"]
  DATATAGS = meta_information["datatags"]
  LABELTAG = meta_information["labeltag"]
  TESTNUMBER = meta_information["test_number"]

  random.seed(meta_information["seed"])
  torch.manual_seed(meta_information["seed"])
  np.random.seed(meta_information["seed"])

  testfiles = nearl.utils.check_filelist(meta_information["test_data"])
  test_data = nearl.io.dataset.Dataset(testfiles, feature_keys = DATATAGS, label_key = LABELTAG)

  model = nearl.utils.get_model(MODEL_TYPE, meta_information["input_channels"], meta_information["output_dimension"], meta_information["dimensions"])
  print(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}")
  if not silent:
    print(f"The model is: {model}")
  if return_model: 
    return model
  
  ###############################
  # Load the pretrained model
  if meta_information["pretrained"] and len(meta_information["pretrained"]) > 0:
    model.load_state_dict(torch.load(meta_information["pretrained"]))
    print(f"Pretrained model loaded from {meta_information['pretrained']}")
  else: 
    raise ValueError(f"Unexpected pretrained model {meta_information['pretrained']}")
  if USECUDA:
    model.cuda()

  if meta_information["loss_function"] not in LOSS_FUNCTIONS.keys():
    raise ValueError(f"Loss function {meta_information['loss_function']} is not supported.") 
  else: 
    criterion = LOSS_FUNCTIONS[meta_information["loss_function"]]()

  ret_pred, targets, ret_loss = nearl.utils.test_model(model, test_data, criterion, TESTNUMBER, BATCH_SIZE, USECUDA, WORKER_NR)
  ret_pred = ret_pred.flatten()
  targets = targets.flatten()
  ret_loss = ret_loss.flatten()

  if meta_information["output_dimension"] > 1: 
    accuracy = np.count_nonzero(ret_pred == targets) / len(ret_pred)
    print(f"Accuracy: {accuracy}")
  else: 
    # RMSE between ret_pred, targets
    rmse = np.sqrt(np.mean((ret_pred - targets)**2).sum())
    R = pearsonr(ret_pred, targets)
    rho = spearmanr(ret_pred, targets)
    print(f"RMSE: {rmse}, Pearson R: {R}, Spearman Rho: {rho}")
  
  return ret_pred, targets
  

def perform_testing_CLI(): 
  args = parse_args()
  SETTINGS = vars(args)
  print("Settings of this training:")
  print(json.dumps(SETTINGS, indent=2))

  with open(SETTINGS["meta_information"], "r") as f:
    meta_information = json.load(f)
  
  meta_information.update(SETTINGS)

  ret_pred, targets, ret_loss = perform_testing_API(meta_information)
  ret_pred = ret_pred.flatten()
  targets = targets.flatten()
  ret_loss = ret_loss.flatten()
  
  if meta_information["output_dimension"] > 1: 
    # Classification problem 
    accuracy = np.count_nonzero(ret_pred == targets) / len(ret_pred)
    print(f"Accuracy: {accuracy}")
  else:
    # Regression problem 
    rmse = np.sqrt(np.mean((ret_pred - targets)**2).sum())
    R = pearsonr(ret_pred, targets)
    rho = spearmanr(ret_pred, targets)
    print(f"RMSE: {rmse}, Pearson R: {R}, Spearman Rho: {rho}")
    

  # # Find matched model file and set the corresponding output column
  # print(f"Updating the performance data in {SETTINGS['output_file']}, with the pretrained model {SETTINGS['pretrained']}")
  # df = pd.read_csv(SETTINGS["output_file"], index_col=None)
  # df.loc[df["param_path"] == SETTINGS["pretrained"], "acc_test"] = accuracy_on_test
  # df.loc[df["param_path"] == SETTINGS["pretrained"], "loss_test"] = loss_on_test
  # df.loc[df["param_path"] == SETTINGS["pretrained"], "acc_train"] = accuracy_on_train
  # df.loc[df["param_path"] == SETTINGS["pretrained"], "loss_train"] = loss_on_train
  # df.to_csv(SETTINGS["output_file"], index=False)
  

if __name__ == "__main__":
  """
  Test the model with 
  """
  perform_testing_CLI()


