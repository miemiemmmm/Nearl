import os, sys, argparse, time, random, json

import scipy.stats
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import nearl
from nearl.io.dataset import Dataset

# writer = SummaryWriter("/Matter/nearl_tensorboard")
tensorboard_writer = None 


OPTIMIZERS = {
  "adam": optim.Adam, 
  "sgd": optim.SGD,
  "adamw": optim.AdamW,
}

adam_kwargs = {
  "betas": (0.9, 0.999),
}

LOSS_FUNCTIONS = {
  "crossentropy": nn.CrossEntropyLoss,
  "mse": nn.MSELoss
}

def get_model(model_type:str, input_dim:int, output_dim:int, box_size, **kwargs): 
  channel_nr = input_dim      # TODO: pass this as a parameter later 
  if model_type == "voxnet":      # Model done
    import nearl.models.model_voxnet
    voxnet_parms = {
      "input_channels": channel_nr,
      "output_dimension": output_dim,
      "input_shape": box_size,
      "dropout_rates" : [0.2, 0.3, 0.4],   # 0.25, 0.25, 0.25
    }
    return nearl.models.model_voxnet.VoxNet(**voxnet_parms)

  elif model_type == "deeprank":
    import nearl.models.model_deeprank
    return nearl.models.model_deeprank.DeepRankNetwork(channel_nr, output_dim, box_size)

  elif model_type == "gnina2017":
    import nearl.models.model_gnina
    return nearl.models.model_gnina.GninaNetwork2017(channel_nr, output_dim, box_size)

  elif model_type == "gnina2018":
    import nearl.models.model_gnina
    return nearl.models.model_gnina.GninaNetwork2018(channel_nr, output_dim, box_size)

  elif model_type == "kdeep":
    import nearl.models.model_kdeep
    return nearl.models.model_kdeep.KDeepNetwork(channel_nr, output_dim, box_size)

  elif model_type == "pafnucy":
    import nearl.models.model_pafnucy
    return nearl.models.model_pafnucy.PafnucyNetwork(channel_nr, output_dim, box_size, **kwargs)

  elif model_type == "atom3d":
    import nearl.models.model_atom3d
    return nearl.models.model_atom3d.Atom3DNetwork(channel_nr, output_dim, box_size)

  elif model_type == "resnet":
    from feater.models.resnet import ResNet
    model = ResNet(channel_nr, output_dim, "resnet18")

  elif model_type == "convnext_iso":
    from feater.models.convnext import ConvNeXtIsotropic
    model = ConvNeXtIsotropic(in_chans = channel_nr, num_classes=output_dim)

  elif model_type == "ViT":
    from transformers import ViTConfig, ViTForImageClassification
    configuration = ViTConfig(
      image_size = 128, 
      num_channels = channel_nr, 
      num_labels = output_dim, 
      window_size=4, 
    )
    model = ViTForImageClassification(configuration)

  else: 
    raise ValueError(f"Model type {model_type} is not supported")
  return model


def test_model(model, dataset, criterion, test_number, batch_size, use_cuda=1, process_nr=24):
  test_loss = 0
  count = 0
  tested_sample_nr = 0
  metric = 0
  with torch.no_grad():
    model.eval()
    for data, target in dataset.mini_batches(batch_size=batch_size, process_nr=process_nr):
      if tested_sample_nr >= test_number:
        break
      if use_cuda:
        data, target = data.cuda(), target.cuda()
      output = model(data)
      if "logits" in dir(output): 
        output = output.logits
      loss = criterion(output, target)
      test_loss += loss.item()
      count += 1
      tested_sample_nr += len(data)
      if output.shape[1] == 1: 
        # For regression
        sq_diff = (output.squeeze() - target.squeeze()).pow(2).sum().item()
        metric += sq_diff
      elif output.shape[1] > 1:
        # For classification
        accuracy = (output.argmax(dim=1) == target).sum().item()
        metric += accuracy
      else:
        raise ValueError("The target shape is not recognized")
    test_loss /= count
    if output.shape[1] == 1: 
      metric = (metric / tested_sample_nr) ** 0.5
    else: 
      metric /= count
  return test_loss, metric

def test_samples(model, dataset, criterion, test_number, batch_size, use_cuda=1, process_nr=24):
  c_sample = 0
  results = []
  targets = []
  with torch.no_grad():
    model.eval()
    for data, target in dataset.mini_batches(batch_size=batch_size, process_nr=process_nr):
      if c_sample >= test_number:
        break
      if use_cuda:
        data, target = data.cuda(), target.cuda()
      output = model(data)
      if "logits" in dir(output): 
        output = output.logits
      results.append(output.flatten())
      targets.append(target.flatten())
      c_sample += len(data)
      
  return torch.cat(results), torch.cat(targets)


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

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
  parser = argparse.ArgumentParser(description="Train a model for the given dataset")
  parser.add_argument("-m", "--model", type=str, required=True, help="Model to use")
  parser.add_argument("--optimizer", type=str, default="adam", help="The optimizer to use")
  parser.add_argument("--loss-function", type=str, default="crossentropy", help="The loss function to use")

  # Data and output
  parser.add_argument("-train", "--training_data", type=str, required=True, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test", "--test_data", type=str, required=True, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder")
  parser.add_argument("-od", "--output_dimension", type=int, required=True, help="Output dimension, 1 for regression, >1 for classification")
  parser.add_argument("-w", "--data_workers", type=int, default=12, help="Number of workers for data loading")
  parser.add_argument("--test_number", type=int, default=4000, help="Number of test samples to use")
  parser.add_argument("-t", "--tags", type=str, default="", help="Feature named delimited by '%' ")
  parser.add_argument("-l", "--labeltag", type=str, default="label", help="Label name")
  parser.add_argument("-a", "--augment", type=int, default=1, help="Augment the data")
  
  # Pretrained model and break point restart
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")
  
  # Training parameters
  parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("-lr", "--lr-init", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--lr-decay-steps", type=int, default=30, help="Decay the learning rate every n steps")
  parser.add_argument("--lr-decay-rate", type=float, default=0.5, help="Decay the learning rate by this rate")

  # Other parameters
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("--manualSeed", type=int, help="Manually set seed")
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--production", type=int, default=0, help="Production mode")

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
  # Training Parameters
  USECUDA = training_settings["cuda"]
  MODEL_TYPE = training_settings["model"]

  START_EPOCH = training_settings["start_epoch"]
  EPOCH_NR = training_settings["epochs"]
  BATCH_SIZE = training_settings["batch_size"]
  WORKER_NR = training_settings["data_workers"]
  AUGMENT = training_settings["augment"]

  datatags = training_settings["datatags"]
  labeltag = training_settings["labeltag"]
  dimensions = training_settings["dimensions"]    # TODO: this has to match the voxel or image size
  output_dims = training_settings["output_dimension"]
  
  # Set the random seed
  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])
  
  
  # Load the datasets
  trainingfiles = nearl.utils.check_filelist(training_settings["training_data"])
  testfiles     = nearl.utils.check_filelist(training_settings["test_data"])
  training_data = Dataset(trainingfiles, dimensions, feature_keys = datatags, label_key = labeltag)
  test_data = Dataset(testfiles, dimensions, feature_keys = datatags, label_key = labeltag)
  
  model = get_model(MODEL_TYPE, len(datatags), output_dims, dimensions)
  print(model)
  # Use KaiMing He's initialization
  c = 0
  for m in model.modules():
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
      print(f"Init Conv Layer {c}")
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      c += 1
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      print(f"Init BatchNorm Layer {c}")
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
      c += 1
    elif isinstance(m, nn.Linear):
      print(f"Init Linear Layer {c}")
      nn.init.normal_(m.weight, 0, 0.1)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      c += 1

  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    model.load_state_dict(torch.load(training_settings["pretrained"]))
  if USECUDA:
    model.cuda()

  # Set the optimizer and loss function
  # if training_settings["optimizer"] not in OPTIMIZERS:
  #   raise ValueError(f"Optimizer {training_settings['optimizer']} is not supported")
  if training_settings["optimizer"] == "adam":
    optimizer = OPTIMIZERS[training_settings["optimizer"]](model.parameters(), lr=training_settings["lr_init"], **adam_kwargs)
  else:
    raise ValueError(f"Optimizer {training_settings['optimizer']} is not supported")
  # optimizer = optim.Adam(model.parameters(), lr=training_settings["lr_init"], betas=(0.9, 0.999))
  
  if training_settings["loss_function"] not in LOSS_FUNCTIONS:
    raise ValueError(f"Loss function {training_settings['loss_function']} is not supported")
  criterion = LOSS_FUNCTIONS[training_settings["loss_function"]]()

  # criterion = nn.CrossEntropyLoss()  # For classification
  # criterion = nn.MSELoss() 

  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_settings["lr_decay_steps"], gamma=training_settings["lr_decay_rate"])
  
  parameter_number = sum([p.numel() for p in model.parameters()])
  training_settings["parameter_number"] = parameter_number
  print(f"Number of parameters: {parameter_number}")

  for epoch in range(0, EPOCH_NR): 
    st = time.perf_counter()
    st_training = time.perf_counter()
    if (epoch < START_EPOCH): 
      print(f"Skip the epoch {epoch}/{START_EPOCH} ...")
      scheduler.step()
      print(f"Epoch {epoch} took {time.perf_counter() - st_training:6.2f} seconds to train. Current learning rate: {get_lr(optimizer):.6f}. ")
      continue
    message = f" Running the epoch {epoch+1 : >4d}/{EPOCH_NR : <4d} at {time.ctime()} "
    print(f"{message:#^80}")
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR, augment=AUGMENT)):
      # print(f'processing batch {batch_idx}/{batch_nr}')
      train_data, train_label = batch

      # In case of nan, replace nan with 0 
      if torch.isnan(train_data).any():
        train_data[torch.isnan(train_data)] = 0

      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()
      optimizer.zero_grad()
      model = model.train()
      pred = model(train_data)

      if "logits" in dir(pred): 
        pred = pred.logits

      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()

      # Measure the RMSE of the training set and one batch of the test set
      if (batch_idx+1) % (batch_nr // 5) == 0: 
        loss_on_train, metric_on_train = test_model(model, training_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
        loss_on_test, metric_on_test = test_model(model, test_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
        jobmsg = f"Processing the block {batch_idx:>5d}/{batch_nr:<5d}; Loss: {loss_on_test:>6.4f}/{loss_on_train:<6.4f}; "
        if output_dims == 1: 
          # RMSE
          jobmsg += f"RMSE: {metric_on_test:>6.4f}/{metric_on_train:<6.4f}; "
        elif output_dims > 1:
          # Accuracy
          jobmsg += f"Accuracy: {metric_on_test:>6.4f}/{metric_on_train:<6.4f}; "
        else: 
          raise ValueError("The output dimension is not recognized")

        if batch_idx > 0: 
          time_left = (time.perf_counter() - st_training) / (batch_idx + 1) * (batch_nr - batch_idx)
          jobmsg += f"Time-left: {time_left:5.0f}s; "
        print(jobmsg)
          
        if (training_settings["verbose"] > 0) or (not training_settings["production"]): 
          # Add the performance to the tensorboard
          tensorboard_writer.add_scalar("Loss/Train", loss_on_train, epoch*batch_nr+batch_idx)
          tensorboard_writer.add_scalar("Loss/Test", loss_on_test, epoch*batch_nr+batch_idx)
          if output_dims == 1: 
            # RMSE
            tensorboard_writer.add_scalar("RMSE/Train", metric_on_train, epoch*batch_nr+batch_idx)
            tensorboard_writer.add_scalar("RMSE/Test", metric_on_test, epoch*batch_nr+batch_idx)
          elif output_dims > 1:
            # Accuracy
            tensorboard_writer.add_scalar("Accuracy/Train", metric_on_train, epoch*batch_nr+batch_idx)
            tensorboard_writer.add_scalar("Accuracy/Test", metric_on_test, epoch*batch_nr+batch_idx)

          inferred, gound_truth = test_samples(model, test_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
          pearson_on_test, spearman_on_test = compute_correlations(gound_truth, inferred)
          tensorboard_writer.add_scalar("Correlation/TestR", pearson_on_test, epoch*batch_nr+batch_idx)
          tensorboard_writer.add_scalar("Correlation/TestRho", spearman_on_test, epoch*batch_nr+batch_idx)

          scatter_fig = plt.figure()
          plt.scatter(gound_truth.cpu().numpy(), inferred.cpu().numpy()) 
          tensorboard_writer.add_figure("Scatter/Test", scatter_fig, epoch*batch_nr+batch_idx)

          inferred, gound_truth = test_samples(model, training_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
          pearson_on_train, spearman_on_train = compute_correlations(gound_truth, inferred)
          tensorboard_writer.add_scalar("Correlation/TrainR", pearson_on_train, epoch*batch_nr+batch_idx)
          tensorboard_writer.add_scalar("Correlation/TrainRho", spearman_on_train, epoch*batch_nr+batch_idx)

          # Plot the scatter to the tensorboard
          scatter_fig = plt.figure()
          plt.scatter(gound_truth.cpu().numpy(), inferred.cpu().numpy())
          tensorboard_writer.add_figure("Scatter/Train", scatter_fig, epoch*batch_nr+batch_idx)

          
          # epoch_info = f"{epoch+1}/{EPOCH_NR}:{batch_idx+1}/{batch_nr}"
          # time_left = (time.perf_counter() - st) / batch_idx * (batch_nr - batch_idx)
          # msg = f"Epoch {epoch_info:>15}: RMSE: {termse:.4f}/{trrmse:.4f}; Loss: {teloss:.4f}/{trloss:.4f}; Corr: {te_pearson:.4f}/{tr_pearson:.4f}; RHO: {te_spearman:.4f}/{tr_spearman:.4f}; Time left: {time_left:.0f} seconds"
      st = time.perf_counter()
    scheduler.step()
    print(f"Epoch {epoch} took {time.perf_counter() - st_training:6.2f} seconds to train. Current learning rate: {get_lr(optimizer):.6f}. ")
    loss_on_train, accuracy_on_train = test_model(model, training_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
    loss_on_test, accuracy_on_test = test_model(model, test_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
    print(f"Checking the Performance on Loss: {loss_on_test}/{loss_on_train}; Accuracy: {accuracy_on_test}/{accuracy_on_train} at {time.ctime()}")

    # Save the model
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"{MODEL_TYPE}_{epoch}.pth")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(model.state_dict(), modelfile_output)

    if (args.verbose > 0) or (not args.production): 
      # Add the performance to the tensorboard
      current_lr = get_lr(optimizer)
      tensorboard_writer.add_scalar("Loss/Train", loss_on_train, epoch)
      tensorboard_writer.add_scalar("Accuracy/Train", accuracy_on_train, epoch)
      tensorboard_writer.add_scalar("Loss/Test", loss_on_test, epoch)
      tensorboard_writer.add_scalar("Accuracy/Test", accuracy_on_test, epoch)
      tensorboard_writer.add_scalar("LearningRate", current_lr, epoch)

    # Save the metrics
    with h5.File(os.path.join(training_settings["output_folder"], "performance.h5") , "a") as hdffile:
      nearl.utils.update_hdf_data(hdffile, "loss_train", np.array([loss_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
      nearl.utils.update_hdf_data(hdffile, "loss_test", np.array([loss_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
      nearl.utils.update_hdf_data(hdffile, "accuracy_train", np.array([accuracy_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
      nearl.utils.update_hdf_data(hdffile, "accuracy_test", np.array([accuracy_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
    

if __name__ == "__main__":
  nearl.update_config(verbose = True, debug = True)
  nearl.update_config(verbose = False, debug = False)

  args = parse_args()
  SETTINGS = vars(args)

  SETTINGS["datatags"] =  [i.strip() for i in SETTINGS["tags"].split("%") if len(i.strip()) > 0]
  SETTINGS["input_channels"] = len(SETTINGS["datatags"])
  SETTINGS["dimensions"] = 32     # TODO: this has to be set according to the H5 dataset

  _SETTINGS = json.dumps(SETTINGS, indent=2)
  print("Settings of this training:")
  print(_SETTINGS)

  if (args.verbose > 0) or (not args.production): 
    if not os.path.exists(os.path.join(SETTINGS["output_folder"], "tensorboard")): 
      os.makedirs(os.path.join(SETTINGS["output_folder"], "tensorboard")) 
    tensorboard_writer = SummaryWriter(os.path.join(SETTINGS["output_folder"], "tensorboard"))
  
  st = time.perf_counter()
  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)

  perform_training(SETTINGS)

  # Potentially added some more configurations of the training job
  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(json.dumps(SETTINGS, indent=2))
  
  print(SETTINGS)

  print(f"Training finished, time elapsed: {time.perf_counter() - st:.2f}")

