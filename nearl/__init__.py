from builtins import print as __builtinprint


__version__ = "0.0.1"


CONFIG = {
  "tempfolder" : "/tmp/", 
  "clear" : True, 
  "verbose" : False, 
  "usegpu": True, 
  "debug" : False, 
}


# TODO: SUPPORT ACCESS TO TRAJECTORY DATABASE
# TODO: THE INTEGRATION OF MOLECULAR BLOCK SEGMENTATION 
# "SEGMENT_LIMIT" : 6,
# "VIEWPOINT_STANDPOINT": "self",
# "VIEWPOINT_BINS" : 24,
# "DOWN_SAMPLE_POINTS" : 1000,

# "SEGMENT_CMAP" : "inferno",
# PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

def update_config(dict_to_update:dict={}):
  for key, value in dict_to_update.items():
    if key in CONFIG:
      CONFIG[key] = value
    else:
      raise KeyError(f"Key {key} is not in the configuration file")
  parse_config()
    
def parse_config():
  global _clear, _verbose, _tempfolder, _usegpu, _debug
  _clear = CONFIG.get("clear", True)
  _verbose = CONFIG.get("verbose", False)
  _tempfolder = CONFIG.get("tempfolder", "/tmp")
  _usegpu = CONFIG.get("usegpu", False)
  _debug = CONFIG.get("debug", False)
parse_config()


if _debug or _verbose:
  from datetime import datetime
  def logit(function):
    def add_log_info(*arg, **kwarg):
      info_message = " ".join((str(i) for i in arg))
      # create log message with timestamp and function call stack for debugging purpose
      timestamp = datetime.now().strftime('%y-%m-%dT%H:%M:%S')
      function_stack = [i.function for i in inspect.stack()[1:-1]]
      log_message = f"{timestamp:15s}: {'->'.join(function_stack)} said: " + info_message
      function(log_message, **kwarg)   # execute the decorated function
    return add_log_info
else: 
  from time import perf_counter
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


def summary(): 
  import os
  printit(f"Summary: {'Clear' if _clear else 'Keep'} temporary files; {'Verbose' if _verbose else 'Silent'} mode; {'Using GPU' if _usegpu else 'Using CPU only'}")
  if (not os.path.exists(_tempfolder)) or (not os.path.isdir(_tempfolder)):
    raise OSError("The temporary folder (tempfolder) does not exist; Please check the configuration file")
  elif not os.access(_tempfolder, os.W_OK):
    raise OSError("The temporary folder (tempfolder) is not writable")
  
summary()


from inspect import stack as __call_stack
def draw_call_stack():
  """
  Draw the calling stack of a function for debugging purpose
  """
  printit(f"{'Drawing Calling Stack':=^100s}")
  for frame_info in __call_stack.stack():
    printit(f"Function: {frame_info.function:<20s} | Line: {frame_info.lineno:<5d} from File: {frame_info.filename:40}")
  printit(f"{'End Drawing Calling Stack':=^100s}")


from nearl import io, features, models, utils, data
