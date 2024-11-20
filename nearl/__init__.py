import os, sys, time
from datetime import datetime
from inspect import stack as __call_stack
from builtins import print as __builtinprint

__version__ = "0.0.1"

CONFIG = {
  "tempfolder" : "/tmp/", 
  "clear" : True, 
  "verbose" : False, 
  "usegpu": True, 
  "debug" : False, 
  "reportdatetime": False
}

class config:
  @staticmethod
  def verbose():
    return CONFIG.get("verbose", False)
  @staticmethod
  def tempfolder():
    return CONFIG.get("tempfolder", "/tmp")
  @staticmethod
  def debug():
    return CONFIG.get("debug", False)
  @staticmethod
  def usegpu():
    return CONFIG.get("usegpu", False)
  @staticmethod
  def clear():
    return CONFIG.get("clear", True)
  @staticmethod
  def reportdatetime():
    return CONFIG.get("reportdatetime", False)

def update_config(dict_to_update:dict={}, **kwargs):
  for key, value in dict_to_update.items():
    if key in CONFIG:
      CONFIG[key] = value
    else:
      raise KeyError(f"Key {key} is not in the configuration file")
  
  for key, value in kwargs.items():
    if key in CONFIG:
      CONFIG[key] = value
    else:
      raise KeyError(f"Key {key} is not in the configuration file")

_start_time = time.perf_counter()

def loginfo_runtime():
  """
  Simply adding a timestamp to the log message
  """
  timestamp = time.perf_counter() - _start_time
  log_message = f"Running {timestamp:8.2f}: " 
  return log_message

def loginfo_datetime():
  """
  Simply adding a timestamp to the log message
  """
  timestamp = datetime.now().strftime('%y-%m-%dT%H:%M:%S')
  log_message = f"{timestamp}: " 
  return log_message

def loginfo_debug():
  """
  Report the calling stack of a function
  """
  timestamp = datetime.now().strftime('%y-%m-%dT%H:%M:%S')
  thestack = __call_stack()[::-1][1:-2]
  function_stack = [i.function for i in thestack]
  log_message = f"{timestamp:15s}: {'>'.join(function_stack)}: " 
  return log_message

def printit(*arg, **kwarg):
  if config.debug():
    log_msg = loginfo_debug()
  elif config.verbose() or config.reportdatetime(): 
    log_msg = loginfo_datetime()
  else: 
    log_msg = loginfo_runtime()
  
  printed = log_msg + " ".join((str(i) for i in arg))
  if "Warning" in arg[0]:
    if sys.stdout.isatty(): 
      __builtinprint(f"\033[93m{printed}\033[0m", file=sys.stderr, **kwarg)
    else:
      __builtinprint(printed, file=sys.stderr, **kwarg)
  elif "Error" in arg[0]:
    if sys.stdout.isatty():
      __builtinprint(f"\033[91m{printed}\033[0m", file=sys.stderr, **kwarg)
    else: 
      __builtinprint(printed, file=sys.stderr, **kwarg)
  else: 
    __builtinprint(printed, **kwarg)
  with open(os.path.join(os.path.abspath(config.tempfolder()), "nearl.log"), "a") as log_file:
    print(*arg, file=log_file)


def summary(): 
  printit(f"Summary: {'Clear' if config.clear() else 'Keep'} temporary files; {'Verbose' if config.verbose() else 'Silent'} mode; {'Using GPU' if config.usegpu() else 'Using CPU only'}")
  if (not os.path.exists(config.tempfolder())) or (not os.path.isdir(config.tempfolder())):
    raise OSError("The temporary folder (tempfolder) does not exist; Please check the configuration file")
  elif not os.access(config.tempfolder(), os.W_OK):
    raise OSError("The temporary folder (tempfolder) is not writable")
  
if config.verbose() or config.debug():
  summary()

def draw_call_stack():
  """
  Draw the calling stack of a function for debugging purpose
  """
  printit(f"{'Drawing Calling Stack':=^100s}")
  for frame_info in __call_stack():
    printit(f"Function: {frame_info.function:<20s} | Line: {frame_info.lineno:<5d} from File: {frame_info.filename:40}")
  printit(f"{'End Drawing Calling Stack':=^100s}")


def get_example_data(path="./"): 
  """
  Download the example data from the data repository to the target folder. 

  Notes
  -----
  Keywords of the returned dictionary: 
  MINI_TRAJSET, MINI_PDBSET, PDBBIND_REFINED, PDBBIND_GENERAL
  
  """
  if not os.path.exists(path):
    raise OSError(f"Path {path} does not exist")
  elif not os.path.isdir(path):
    raise OSError(f"Path {path} is not a directory")
  elif not os.access(path, os.W_OK):
    raise OSError(f"Path {path} is not writable")
  else:
    import subprocess 
    os.chdir(path)

    # Download the example data 
    # TODO: Add the link to the example data
    if not os.path.exists("example_data.tar.gz"):
      printit(f"Downloading example data to {path}")
      datafile_url = "https://miemiemmmm.b-cdn.net/shared_files/example_data.tar.gz" 
      subprocess.run(["wget", "--directory-prefix", path, datafile_url]) 
    else: 
      printit("The example data (example_data.tar.gz) already exists! Skip downloading...") 
    
    # Extract the compressed file 
    if os.path.exists("example_data"):
      printit("The example data folder already exists! Skip extracting...")
    else:
      printit("Extracting the example data...")
      subprocess.run(["tar", "-xf", "example_data.tar.gz"], cwd=path) 
    
    # Obtain the data paths as a dictionary 
    os.chdir("example_data")
    if not os.path.exists("data.py"):
      raise OSError(f"Data index file not found in the extracted folder {os.getcwd()}")
    import data
    paths = data.get_data()

    return paths

# from nearl import io, features, models, utils, data, commands
# from . import features, featurizer, io, utils
# from .features import Feature
# from .featurizer import Featurizer
# from .io import Dataset, Trajectory, TrajectoryLoader

