import os, sys, argparse, time, random, json

import scipy.stats
import numpy as np
import pandas as pd
import h5py as h5
import matplotlib.pyplot as plt 
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nearl
from nearl.io.dataset import Dataset

COLORS = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
          '#ff7f00','#999999','#a65628','#f781bf']

# writer = SummaryWriter("/Matter/nearl_tensorboard")
tensorboard_writer = None 

OPTIMIZERS = {
  "sgd": optim.SGD,
  "adam": optim.Adam, 
  "adamw": optim.AdamW,
}

optim_kwargs = {
  "betas": (0.99, 0.999),
}

LOSS_FUNCTIONS = {
  "crossentropy": nn.CrossEntropyLoss, 
  "mse": nn.MSELoss, 
  "mae": nn.L1Loss, 
  "l1": nn.L1Loss, 
  "huber": nn.HuberLoss,
}


def draw_scatter(pred_data, target_data, figtype="scatter", title="", fig=None, color="blue", clean=False): 
  """
  Draw the scatter plot for the predicted and target data 
  """
  if clean:
    plt.close("all")
    plt.clf()
  # Compute the metrics 
  rmse = np.sqrt(np.mean((pred_data - target_data)**2).sum())
  R, rho = compute_correlations(target_data, pred_data)
  title += f"RMSE: {rmse:4.2f}; R: {R:4.2f}; rho: {rho:4.2f}"

  df = pd.DataFrame({"groundtruth": target_data, "predicted": pred_data})
  if fig is None: 
    f = sns.JointGrid(data = df, x="groundtruth", y="predicted", xlim=(1, 13), ylim=(1, 13))
  else:
    f = fig 
  if figtype == "scatter":
    f.ax_joint.scatter(df["groundtruth"], df["predicted"], color=color, label=title, marker="x", alpha=0.5)
  elif figtype == "kde":
    f.ax_joint.kdeplot(df["groundtruth"], df["predicted"], levels=11, cmap="inferno", thresh=0)
  else: 
    raise ValueError(f"Figure type {figtype} is not recognized") 
  bins = np.linspace(1, 13, 15)
  f.ax_marg_x.hist(df["groundtruth"], color=color, alpha=0.5, bins=bins, density=True, linewidth=1.5, edgecolor="black")
  f.ax_marg_y.hist(df["predicted"], color=color, alpha=0.5, bins=bins, density=True, orientation="horizontal", linewidth=1.5, edgecolor="black")

  # determine the color of the regression line
  import matplotlib.colors as mcolors
  rgb_color = mcolors.to_rgb(color)
  color_reg = np.array(rgb_color)   # + np.random.normal(scale=0.5, size=3)
  # Limit to 0-1
  color_reg = np.clip(color_reg, 0, 1)
  sns.regplot(data=df, x="groundtruth", y="predicted", scatter=False, color=color_reg, ax=f.ax_joint, ci=50, line_kws=dict(linewidth=2))
  f.ax_joint.legend(loc="lower right")
  # Set the 
  return f 


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
  print(labels, pred)
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
  parser.add_argument("-a", "--augment", type=int, default=0, help="Augment the data")
  
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
  if USECUDA:
    torch.cuda.manual_seed(training_settings["seed"])
  device = torch.device("cuda" if training_settings["cuda"] else "cpu")
  
  # Load the datasets
  trainingfiles = nearl.utils.check_filelist(training_settings["training_data"])
  testfiles     = nearl.utils.check_filelist(training_settings["test_data"])
  training_data = Dataset(trainingfiles, feature_keys = datatags, label_key = labeltag)
  test_data = Dataset(testfiles, feature_keys = datatags, label_key = labeltag)
  
  model = nearl.utils.get_model(MODEL_TYPE, len(datatags), output_dims, dimensions)
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
  model.to(device)

  # Set the optimizer and loss function
  if training_settings["optimizer"] not in OPTIMIZERS: 
    raise ValueError(f"Optimizer {training_settings['optimizer']} is not supported")
  optimizer = OPTIMIZERS[training_settings["optimizer"]](model.parameters(), lr=training_settings["lr_init"], **optim_kwargs)
  
  if training_settings["loss_function"] not in LOSS_FUNCTIONS:
    raise ValueError(f"Loss function {training_settings['loss_function']} is not supported")
  criterion = LOSS_FUNCTIONS[training_settings["loss_function"]]()
  print(f"Loss function: {criterion}")

  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_settings["lr_decay_steps"], gamma=training_settings["lr_decay_rate"])
  
  parameter_number = sum([p.numel() for p in model.parameters()])
  training_settings["parameter_number"] = parameter_number
  print(f"Number of parameters: {parameter_number}")
  

  def normalize_by_channel(data):
    """
    Normalize 5D data by channel.
    Assumes data shape: (samples, channels, dim1, dim2, dim3)
    """
    # Compute mean and std for each channel
    mean = torch.mean(data, dim=(0, 2, 3, 4), keepdim=True)
    std = torch.std(data, dim=(0, 2, 3, 4), keepdim=True)
    
    # Avoid division by zero
    std = torch.where(std == 0, torch.tensor(1e-6), std)
    return (data - mean) / std


  for epoch in range(0, EPOCH_NR): 
    st = time.perf_counter()
    st_training = time.perf_counter()
    if epoch < START_EPOCH: 
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
      train_data, train_label = train_data.to(device), train_label.to(device)

      #### TODO: In case of nan, replace nan with 0 
      # if torch.isnan(train_data).any() or torch.isnan(train_label).any(): 
      #   print(f"Warning: Found NaN in the training data")
      #   train_data[torch.isnan(train_data)] = 0

      optimizer.zero_grad()
      model = model.train()
      pred = model(train_data)

      if "logits" in dir(pred): 
        pred = pred.logits

      loss = criterion(pred, train_label)
      loss.backward()
      # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
      optimizer.step()

      # Measure the RMSE of the training set and one batch of the test set
      if (batch_idx+1) % (batch_nr // 3) == 0: 
        preds_tr, targets_tr, losses_tr = nearl.utils.test_model(model, training_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
        preds_te, targets_te, losses_te = nearl.utils.test_model(model, test_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
        jobmsg = f"Processing the block {batch_idx:>5d}/{batch_nr:<5d}; Loss: {np.mean(losses_te):>6.4f}({np.mean(losses_tr):<6.4f}); "

        if True in np.isnan(preds_te):
          print(f"Warning: Found NaN in the test prediction")
          continue
        elif True in np.isnan(preds_tr):
          print(f"Warning: Found NaN in the training prediction")
          continue

        if output_dims == 1: 
          # Metrics on the training set 
          preds_tr, targets_tr, losses_tr = preds_tr.flatten(), targets_tr.flatten(), losses_tr.flatten()
          rmse_tr = np.sqrt(np.mean((preds_tr - targets_tr)**2).sum())
          pearson_tr, spearman_tr = compute_correlations(targets_tr, preds_tr)

          # Metrics on the test set
          preds_te, targets_te, losses_te = preds_te.flatten(), targets_te.flatten(), losses_te.flatten()
          rmse_te = np.sqrt(np.mean((preds_te - targets_te)**2).sum())
          pearson_te, spearman_te = compute_correlations(targets_te, preds_te)
          jobmsg += f"RMSE: {rmse_te:>4.2f}({rmse_tr:<4.2f}); Pearson: {pearson_te:>4.2f}({pearson_tr:<4.2f}); spearman: {spearman_te:>4.2f}({spearman_tr:<4.2f}); "

        elif output_dims > 1:
          # Accuracy 
          accuracy_tr = np.count_nonzero(preds_tr == targets_tr) / len(preds_tr)
          accuracy_te = np.count_nonzero(preds_te == targets_te) / len(preds_te)
          jobmsg += f"Accuracy: {accuracy_te:>4.2f}/{accuracy_tr:<4.2f}; "

        else: 
          raise ValueError("The output dimension is not recognized")

        if batch_idx > 0: 
          time_left = (time.perf_counter() - st_training) / (batch_idx + 1) * (batch_nr - batch_idx)
          jobmsg += f"Time-left: {time_left:5.0f}s; "
        print(jobmsg)
          
        if (training_settings["verbose"] > 0) or (not training_settings["production"]): 
          # Add the performance to the tensorboard
          tensorboard_writer.add_scalar("Loss/Train", np.mean(losses_tr), epoch*batch_nr+batch_idx)
          tensorboard_writer.add_scalar("Loss/Test", np.mean(losses_te), epoch*batch_nr+batch_idx)
          if output_dims == 1: 
            # RMSE
            tensorboard_writer.add_scalar("Perf/RMSE", rmse_te, epoch*batch_nr+batch_idx)
            tensorboard_writer.add_scalar("Perf/RMSE*", rmse_tr, epoch*batch_nr+batch_idx)
            tensorboard_writer.add_scalar("Perf/R", pearson_te, epoch*batch_nr+batch_idx)
            tensorboard_writer.add_scalar("Perf/Rho", spearman_te, epoch*batch_nr+batch_idx)
          elif output_dims > 1:
            # Accuracy
            tensorboard_writer.add_scalar("Perf/Train", accuracy_tr, epoch*batch_nr+batch_idx)
            tensorboard_writer.add_scalar("Perf/Test", accuracy_te, epoch*batch_nr+batch_idx)

          fig_tr = draw_scatter(preds_tr, targets_tr, figtype="scatter", title="TrainSet:", color=COLORS[1])
          fig_te = draw_scatter(preds_te, targets_te, figtype="scatter", title="TestSet:", fig=fig_tr, color=COLORS[0])
          # tensorboard_writer.add_figure("Vis/Test", fig_te.fig, epoch*batch_nr+batch_idx)
          tensorboard_writer.add_figure("Dist/Results", fig_te.fig, epoch*batch_nr+batch_idx)

          parmset = list(model.named_parameters())
          for idx, p in enumerate(parmset):
            # Skip the bias type of parameters
            if "bias" in p[0]:
              continue
            pname = p[0]
            p = p[1]
            if p.grad is not None:
              grad_norm = p.grad.norm()
              tensorboard_writer.add_scalar(f"GradNorm/{pname}_{idx}", grad_norm, epoch*batch_nr+batch_idx)
              tensorboard_writer.add_histogram(f"GradHist/{pname}_{idx}", p.grad, epoch*batch_nr+batch_idx)

      st = time.perf_counter()
    scheduler.step()
    print(f"Epoch {epoch} took {time.perf_counter() - st_training:6.2f} seconds to train. Current learning rate: {get_lr(optimizer):.6f}. ")
    preds_tr, targets_tr, losses_tr = nearl.utils.test_model(model, training_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR)
    preds_te, targets_te, losses_te = nearl.utils.test_model(model, test_data, criterion, training_settings["test_number"], BATCH_SIZE, USECUDA, WORKER_NR) 
    preds_tr, targets_tr, losses_tr = preds_tr.flatten(), targets_tr.flatten(), losses_tr.flatten()
    preds_te, targets_te, losses_te = preds_te.flatten(), targets_te.flatten(), losses_te.flatten()
    loss_on_train = np.mean(losses_tr)
    loss_on_test = np.mean(losses_te)

    if True in np.isnan(preds_te):
      print(f"Warning: Found NaN in the test prediction")
      continue
    elif True in np.isnan(preds_tr):
      print(f"Warning: Found NaN in the training prediction")
      continue
    
    if output_dims == 1:
      accuracy_on_train = np.sqrt(np.mean((preds_tr - targets_tr)**2).sum())
      accuracy_on_test = np.sqrt(np.mean((preds_te - targets_te)**2).sum())
      pearson_on_train, spearman_on_train = compute_correlations(targets_tr, preds_tr)
      pearson_on_test, spearman_on_test = compute_correlations(targets_te, preds_te)
    else: 
      accuracy_on_train = np.count_nonzero(preds_tr == targets_tr) / len(preds_tr)
      accuracy_on_test = np.count_nonzero(preds_te == targets_te) / len(preds_te)

    # loss_on_test, accuracy_on_test 
    print(f"Checking the Performance on Loss: {loss_on_test}/{loss_on_train}; Accuracy: {accuracy_on_test}/{accuracy_on_train} at {time.ctime()}")

    # Save the model
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"{MODEL_TYPE}_{epoch}.pth")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(model.state_dict(), modelfile_output)
    # Save the figures 
    fig_tr = draw_scatter(preds_tr, targets_tr, figtype="scatter", title="TrainSet:", color=COLORS[1])
    fig_te = draw_scatter(preds_te, targets_te, figtype="scatter", title="TestSet:", fig=fig_tr, color=COLORS[0])
    fig_te.savefig(os.path.join(training_settings["output_folder"], f"Results_{epoch}.png"))


    if (args.verbose > 0) or (not args.production): 
      # Add the performance to the tensorboard
      current_lr = get_lr(optimizer)
      tensorboard_writer.add_scalar("Loss/Train#E", loss_on_train, epoch)
      tensorboard_writer.add_scalar("Perf/Train#E", accuracy_on_train, epoch)
      tensorboard_writer.add_scalar("Loss/Test#E",  loss_on_test, epoch)
      tensorboard_writer.add_scalar("Perf/Test#E",  accuracy_on_test, epoch)
      tensorboard_writer.add_scalar("LearningRate", current_lr, epoch)

    # Save the metrics
    with h5.File(os.path.join(training_settings["output_folder"], "performance.h5") , "a") as hdffile:
      nearl.utils.update_hdf_data(hdffile, "loss_train", np.array([loss_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
      nearl.utils.update_hdf_data(hdffile, "loss_test", np.array([loss_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
      nearl.utils.update_hdf_data(hdffile, "accuracy_train", np.array([accuracy_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
      nearl.utils.update_hdf_data(hdffile, "accuracy_test", np.array([accuracy_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
      if output_dims == 1:
        nearl.utils.update_hdf_data(hdffile, "pearsonr_train", np.array([pearson_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
        nearl.utils.update_hdf_data(hdffile, "pearsonr_test", np.array([pearson_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
        nearl.utils.update_hdf_data(hdffile, "spearmanr_train", np.array([spearman_on_train], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))
        nearl.utils.update_hdf_data(hdffile, "spearmanr_test", np.array([spearman_on_test], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None,))


if __name__ == "__main__":
  # nearl.update_config(verbose = True, debug = True)
  nearl.update_config(verbose = False, debug = False)

  args = parse_args()
  SETTINGS = vars(args)

  SETTINGS["datatags"] =  [i.strip() for i in SETTINGS["tags"].split("%") if len(i.strip()) > 0]
  SETTINGS["input_channels"] = len(SETTINGS["datatags"])
  
  # TODO: this has to be set according to the H5 dataset
  if SETTINGS["model"] in ("resnet", "convnext_iso", "ViT"):
    SETTINGS["dimensions"] = None
  else:
    with open(SETTINGS["training_data"], "r") as f:
      hdffile = f.readline().strip()
      with h5.File(hdffile, "r") as hdf:
        SETTINGS["dimensions"] = int(hdf["featurizer_parms"]["dimensions"][0])

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

