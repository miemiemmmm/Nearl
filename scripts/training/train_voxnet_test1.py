import os, sys, argparse, time, random, json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import feater
import feater.dataloader
from feater.models.voxnet import VoxNet

def parse_args(): 
  parser = argparse.ArgumentParser(description="Train VoxNet")
  parser.add_argument("-train", "--training_data", type=str, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test", "--test_data", type=str, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output_folder", type=str, default="cls", help="Output folder")
  
  # Pretrained model and break point restart
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")

  # Training parameters
  parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
  parser.add_argument("-itv", "--interval", type=int, default=50, help="How many batches to wait before logging training status")

  # Other parameters
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("--data_workers", type=int, default=4, help="Number of workers for data loading")
  parser.add_argument("--manualSeed", type=int, help="Manually set seed")

  
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--dataset", type=str, default="single", help="Dataset to use")

  args = parser.parse_args()
  if not os.path.exists(args.training_data):
    raise FileNotFoundError(f"The file {args.training_data} does not exist")
  if not os.path.exists(args.test_data):
    raise FileNotFoundError(f"The file {args.test_data} does not exist")
  if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)

  if args.dataset == "single":
    args.class_nr = 20
  elif args.dataset == "dual":
    args.class_nr = 400
  return args


# TODO
def perform_training(training_settings: dict): 
  USECUDA = training_settings["cuda"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])
  
  st = time.perf_counter()
  # Load the datasets
  trainingfiles = feater.utils.checkfiles(training_settings["training_data"])
  testfiles = feater.utils.checkfiles(training_settings["test_data"])
  # Load the data.
  # NOTE: In Voxel based training, the bottleneck is the SSD data reading.
  # NOTE: Putting the dataset to SSD will significantly speed up the training.
  training_data = feater.dataloader.VoxelDataset(trainingfiles)
  test_data = feater.dataloader.VoxelDataset(testfiles)

  classifier = VoxNet(n_classes=training_settings["class_nr"])
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  if USECUDA:
    classifier.cuda()

  optimizer = optim.Adam(classifier.parameters(), lr=training_settings["learning_rate"], betas=(0.9, 0.999))
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  
  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")
  START_EPOCH = training_settings["start_epoch"]
  EPOCH_NR = training_settings["epochs"]
  BATCH_SIZE = training_settings["batch_size"]
  INTERVAL = training_settings["interval"]
  WORKER_NR = training_settings["data_workers"]
  for epoch in range(START_EPOCH, EPOCH_NR):
    st = time.perf_counter()
    message = f" Running the epoch {epoch:>4d}/{EPOCH_NR:<4d} at {time.ctime()} "
    print(f"{message:#^80}")
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      # if (batch_idx+1) % 300 == 0:   # TODO: Remove this line when production
      #   break
      train_data, train_label = batch
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()
      optimizer.zero_grad()
      classifier = classifier.train()
      pred = classifier(train_data)
      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()

      if (batch_idx) % 50 == 0:
        # NOTE: Evaluating accuracy every batch will casue a significant slow down.
        accuracy = feater.utils.report_accuracy(pred, train_label, verbose=False)
        print(f"Processing the block {batch_idx:>5d}/{batch_nr:<5d}; Loss: {loss.item():8.4f}; Accuracy: {accuracy:8.4f}")
      
      if (batch_idx + 1) % INTERVAL == 0:
        vdata, vlabel = next(test_data.mini_batches(batch_size=10000, process_nr=WORKER_NR))
        feater.utils.validation(classifier, vdata, vlabel, usecuda=USECUDA)
        print(f"Estimated epoch time: {(time.perf_counter() - st) / (batch_idx + 1) * batch_nr:.2f} seconds")
    scheduler.step()
    # Save the model
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"voxnet_{epoch}.pth")
    conf_mtx_output  = os.path.join(os.path.abspath(training_settings["output_folder"]), f"voxnet_confmatrix_{epoch}.png")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    print(f"Performing the prediction on the test set ...")
    # Evaluate the performance in the test set 
    tdata, tlabel = next(test_data.mini_batches(batch_size=10000, process_nr=WORKER_NR))
    pred, label = feater.utils.validation(classifier, tdata, tlabel, usecuda=USECUDA)
    with feater.io.hdffile(os.path.join(training_settings["output_folder"], "performance.h5") , "a") as hdffile:
      correct = np.count_nonzero(pred == label)
      accuracy = correct / float(label.shape[0])
      feater.utils.update_hdf_by_slice(hdffile, "accuracy", np.array([accuracy], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
    # Evaluate the performance in the training set
      
    
    if training_settings["dataset"] == "single":
      print(f"Saving the confusion matrix to {conf_mtx_output} ...")
      feater.utils.confusion_matrix(pred, label, output_file=conf_mtx_output)
    else: 
      pass

if __name__ == "__main__":
  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(SETTINGS, indent=2)
  print("Settings of this training:")
  print(_SETTINGS)

  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)

  perform_training(SETTINGS)