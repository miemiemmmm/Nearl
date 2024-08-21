"""
Perform benchmarking on the pre-trained models. 

Example: 

python3 /MieT5/MyRepos/FEater/feater/scripts/train_models.py --model convnext --optimizer adam --loss-function crossentropy \
  --training-data /Matter/feater_train_1000/dual_hilbert.txt --test-data /Weiss/FEater_Dual_HILB/te.txt --output_folder /Weiss/benchmark_models/convnext_dual_hilb/ \
  --test-number 4000 -e 120 -b 64 -w 12 --lr-init 0.0001 --lr-decay-steps 30 --lr-decay-rate 0.5 --data-type dual --cuda 1 --dataloader-type hilb --production 0
"""

import os, sys, time, io
import argparse, random, json 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import transformers

from torch.utils.tensorboard import SummaryWriter

# Import models 
from feater.models.pointnet import PointNetCls      
from feater import dataloader, utils
import feater

tensorboard_writer = None 

# For point cloud type of data, the input is in the shape of (B, 3, N)
INPUT_POINTS = 0
def update_pointnr(pointnr):
  global INPUT_POINTS
  INPUT_POINTS = pointnr
DATALOADER_TYPE = ""
def update_dataloader_type(dataloader_type):
  global DATALOADER_TYPE
  DATALOADER_TYPE = dataloader_type

OPTIMIZERS = {
  "adam": optim.Adam, 
  "sgd": optim.SGD,
  "adamw": optim.AdamW,
}

LOSS_FUNCTIONS = {
  "crossentropy": nn.CrossEntropyLoss,
}

DATALOADER_TYPES = {
  "pointnet": dataloader.CoordDataset,
  "pointnet2": dataloader.CoordDataset,
  "dgcnn": dataloader.CoordDataset,
  "paconv": dataloader.CoordDataset,

  
  "voxnet": dataloader.VoxelDataset,
  "deeprank": dataloader.VoxelDataset,
  "gnina": dataloader.VoxelDataset,

  
  "resnet": dataloader.HilbertCurveDataset,
  "convnext": dataloader.HilbertCurveDataset,
  "convnext_iso": dataloader.HilbertCurveDataset,
  "swintrans": dataloader.HilbertCurveDataset,
  "ViT": dataloader.HilbertCurveDataset,

  "coord": dataloader.CoordDataset, 
  "surface": dataloader.SurfDataset, 
}




def get_model(model_type:str, output_dim:int): 
  if model_type == "pointnet":
    model = PointNetCls(output_dim)

  elif model_type == "pointnet2":
    from feater.models.pointnet2 import get_model as get_pointnet2_model
    if DATALOADER_TYPE == "surface": 
      rads = [0.35, 0.75]  # For surface-based training
    elif DATALOADER_TYPE == "coord":
      rads = [1.75, 3.60]  # For coordinate-based data
    else: 
      raise ValueError(f"Unexpected dataloader type {DATALOADER_TYPE} for PointNet2 model; Only surface and coord are supported")
    print(f"Using the radii {rads} for the PointNet2 model")
    model = get_pointnet2_model(output_dim, normal_channel=False, sample_nr = INPUT_POINTS, rads=rads)
    
  elif model_type == "dgcnn":
    from feater.models.dgcnn import DGCNN_cls
    args = {
      "k": 20, 
      "emb_dims": 1024,
      "dropout" : 0.25,
    }
    model = DGCNN_cls(args, output_channels=output_dim)
  elif model_type == "paconv":
    # NOTE: This is a special case due to the special dependency of the PAConv !!!!
    if "/MieT5/tests/PAConv/obj_cls" not in sys.path:
      sys.path.append("/MieT5/tests/PAConv/obj_cls")
    from feater.models.paconv import PAConv
    config = {
      "k_neighbors": 20, 
      "output_channels": output_dim,
      "dropout": 0.25,
    }
    model = PAConv(config)
  elif model_type == "voxnet":
    from feater.models.voxnet import VoxNet
    model = VoxNet(output_dim)
  elif model_type == "deeprank":
    from feater.models.deeprank import DeepRankNetwork
    model = DeepRankNetwork(1, output_dim, 32)
  elif model_type == "gnina":
    from feater.models.gnina import GninaNetwork
    model = GninaNetwork(output_dim)
  elif model_type == "resnet":
    from feater.models.resnet import ResNet
    model = ResNet(1, output_dim, "resnet18")
  elif model_type == "convnext":
    from feater.models.convnext import ConvNeXt
    """
      in_chans=3, num_classes=1000, 
      depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
      drop_path_rate=0.,  layer_scale_init_value=1e-6, 
      head_init_scale=1.,
    """
    model = ConvNeXt(1, output_dim)

  elif model_type == "convnext_iso":
    from feater.models.convnext import ConvNeXt, ConvNeXtIsotropic
    model = ConvNeXtIsotropic(1, output_dim)

  elif model_type == "swintrans":
    from transformers import SwinForImageClassification, SwinConfig
    configuration = SwinConfig(
      image_size = 128, 
      num_channels = 1,
      num_labels = output_dim,
      window_size=4, 
    )
    model = SwinForImageClassification(configuration)

  elif model_type == "ViT":
    from transformers import ViTConfig, ViTForImageClassification
    configuration = ViTConfig(
      image_size = 128, 
      num_channels = 1, 
      num_labels = output_dim, 
      window_size=4, 
    )
    model = ViTForImageClassification(configuration)

  else:
    raise ValueError(f"Unexpected model type {model_type}; Only voxnet, pointnet, resnet, and deeprank are supported")
  return model


def match_data(pred, label):  
  predicted_labels = torch.argmax(pred, dim=1)
  plt.scatter(predicted_labels.cpu().detach().numpy(), label.cpu().detach().numpy(), s=4, c = np.arange(len(label))/len(label), cmap="inferno")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Predicted vs. Actual")
  buf = io.BytesIO()
  plt.savefig(buf, format="png")
  buf.seek(0)
  Image.open(buf)
  image_tensor = torchvision.transforms.ToTensor()(Image.open(buf))
  plt.clf()
  buf.close()
  return image_tensor


def test_model(model, dataset, criterion, test_number, batch_size, use_cuda=1, process_nr=32):
  test_loss = 0.0
  correct = 0
  c = 0
  c_samples = 0
  batch_nr = len(dataset) // batch_size
  with torch.no_grad():
    model.eval()
    for data, target in dataset.mini_batches(batch_size=batch_size, process_nr=process_nr):
      # Correct way to handle the input data
      # For the PointNet, the data is in the shape of (B, 3, N)
      # Important: Swap the axis to make the coordinate as 3 input channels of the data
      # print(target.unique(return_counts=True))
      if c % (batch_nr//10) == 0:
        print(f"Testing: {c}/{batch_nr} at {time.ctime()}")

      if isinstance(dataset, dataloader.CoordDataset) or isinstance(dataset, dataloader.SurfDataset):
        data = data.transpose(2, 1)  
      if use_cuda:
        data, target = data.cuda(), target.cuda()

      pred = model(data)

      if isinstance(model, PointNetCls) or isinstance(pred, tuple):
        pred = pred[0]
      
      # Get the logit if the huggingface models is used
      if isinstance(pred, transformers.file_utils.ModelOutput): 
        pred = pred.logits
      
      pred_choice = torch.argmax(pred, dim=1)
      correct += pred_choice.eq(target.data).cpu().sum().item()
      test_loss += criterion(pred, target).item()

      # Increament the counter for test sample count
      c_samples += len(data)
      c += 1
      # if c_samples >= test_number:
      #   break
    test_loss /= c
    accuracy = correct / c_samples
    
    return test_loss, accuracy


def parse_args():
  parser = argparse.ArgumentParser(description="Train the models used in the FEater paper. ")

  # Input data files 
  # parser.add_argument("-train", "--training-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test",  "--test-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-p",     "--pretrained", type=str, required=True, help="Pretrained model path")
  parser.add_argument("-m",     "--meta-information", type=str, required=True, help="The meta information file for the pretrained model")
  parser.add_argument("-o",     "--output-file", type=str, required=True, help="The output folder to store the model and performance data")
  parser.add_argument("-w",     "--data-workers", type=int, default=12, help="Number of workers for data loading")
  
  # Miscellanenous
  parser.add_argument("--test-number", type=int, default=100_000_000, help="Number of test samples")
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


def perform_testing(training_settings: dict): 
  USECUDA = training_settings["cuda"]
  MODEL_TYPE = training_settings["model"]
  BATCH_SIZE = training_settings["batch_size"]
  WORKER_NR = training_settings["data_workers"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])

  ###################
  # Load the datasets
  trainingfiles = utils.checkfiles(training_settings["training_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  if training_settings["dataloader_type"] in ("surface", "coord"):
    training_data = DATALOADER_TYPES[training_settings["dataloader_type"]](trainingfiles, target_np=training_settings["pointnet_points"])
    test_data = DATALOADER_TYPES[training_settings["dataloader_type"]](testfiles, target_np=training_settings["pointnet_points"])
  elif MODEL_TYPE == "pointnet": 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles, target_np=training_settings["pointnet_points"])
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles, target_np=training_settings["pointnet_points"])
  else: 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles)
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles)
  print(f"Training data size: {len(training_data)}; Test data size: {len(test_data)}; Batch size: {BATCH_SIZE}; Worker number: {WORKER_NR}")

  ###################
  # Load the model
  classifier = get_model(MODEL_TYPE, training_settings["class_nr"])
  print(f"Classifier: {classifier}")

  ###################
  # Load the pretrained model
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  else: 
    raise ValueError(f"Unexpected pretrained model {training_settings['pretrained']}")
  if USECUDA:
    classifier.cuda()

  ###################
  # Get loss function 
  # The loss function in the original training
  criterion = LOSS_FUNCTIONS.get(training_settings["loss_function"], nn.CrossEntropyLoss)()
  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")

  ####################
  test_number = training_settings["test_number"]
  loss_on_train, accuracy_on_train = test_model(classifier, training_data, criterion, test_number, BATCH_SIZE, USECUDA, WORKER_NR)
  loss_on_test, accuracy_on_test = test_model(classifier, test_data, criterion, test_number, BATCH_SIZE, USECUDA, WORKER_NR)
  print(f"Checking the Performance on Loss: {loss_on_test}/{loss_on_train}; Accuracy: {accuracy_on_test}/{accuracy_on_train} at {time.ctime()}")

  return loss_on_train, accuracy_on_train, loss_on_test, accuracy_on_test

  

if __name__ == "__main__":
  """
  Test the model with 

  """

  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(SETTINGS, indent=2)
  print("Settings of this training:")
  print(_SETTINGS)

  # with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
  #   f.write(_SETTINGS)

  # Read the input meta-information 
  with open(SETTINGS["meta_information"], "r") as f:
    meta_information = json.load(f)
    # del meta_information["training_data"]
    del meta_information["test_data"]
    del meta_information["pretrained"]

    update_pointnr(meta_information["pointnet_points"])
    update_dataloader_type(meta_information["dataloader_type"])

  SETTINGS.update(meta_information)  # Update the settings with the requested meta-information 

  print(SETTINGS)
  
  loss_on_train, accuracy_on_train, loss_on_test, accuracy_on_test = perform_testing(SETTINGS)


  # Find matched model file and set the corresponding output column
  print(f"Updating the performance data in {SETTINGS['output_file']}, with the pretrained model {SETTINGS['pretrained']}")
  df = pd.read_csv(SETTINGS["output_file"], index_col=None)
  df.loc[df["param_path"] == SETTINGS["pretrained"], "acc_test"] = accuracy_on_test
  df.loc[df["param_path"] == SETTINGS["pretrained"], "loss_test"] = loss_on_test
  df.loc[df["param_path"] == SETTINGS["pretrained"], "acc_train"] = accuracy_on_train
  df.loc[df["param_path"] == SETTINGS["pretrained"], "loss_train"] = loss_on_train
  df.to_csv(SETTINGS["output_file"], index=False)
  




