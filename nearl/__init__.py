import os
from time import perf_counter
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

# "SEGMENT_LIMIT" : 6,
# "VIEWPOINT_STANDPOINT": "self",
# "VIEWPOINT_BINS" : 24,
# "DOWN_SAMPLE_POINTS" : 1000,
# "SEGMENT_CMAP" : "inferno",
# PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

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

if config.debug() or config.verbose():
  def logit(function):
    def add_log_info(*arg, **kwarg):
      info_message = " ".join((str(i) for i in arg))
      # create log message with timestamp and function call stack for debugging purpose
      timestamp = datetime.now().strftime('%y-%m-%dT%H:%M:%S')
      function_stack = [i.function for i in __call_stack()[1:-1]]
      log_message = f"{timestamp:15s}: {'->'.join(function_stack)} said: " + info_message
      function(log_message, **kwarg)   # execute the decorated function
    return add_log_info
else: 
  _start_time = perf_counter()
  def logit(function):
    def add_log_info(*arg, **kwarg):
      info_message = " ".join((str(i) for i in arg))
      timestamp = perf_counter() - _start_time
      log_message = f"Running {timestamp:8.2f}: " + info_message
      function(log_message, **kwarg)   # execute the decorated function
    return add_log_info


@logit
def printit(*arg, **kwarg):
  __builtinprint(*arg, **kwarg)
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


# from nearl import io, features, models, utils, data
from . import features, featurizer, io
# from . import commands




