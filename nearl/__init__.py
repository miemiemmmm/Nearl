import datetime, builtins, os, inspect, time
import importlib.resources as resources

########################################################

__version__ = "0.0.1"
_start_time = time.perf_counter()
configfile = resources.files("nearl").joinpath("../CONFIG.json")


CONFIG = {
  "tempfolder" : os.path.abspath("/tmp/"),
  "clear" : True, 
  "verbose" : False, 
  "usegpu": True, 
  "debug" : False, 

  "WINDOW_SIZE" : 10,
  

  # Box partition parameters : TODO 
  "SEGMENT_LIMIT" : 6,
  "VIEWPOINT_STANDPOINT": "self",
  "VIEWPOINT_BINS" : 24,
  "DOWN_SAMPLE_POINTS" : 1000,

  # SQL database parameters
  "user" : "yang",
  "dbname": "testfeatures",
  "host": "localhost",
  "password": "testyang",

  "datadir": "/MieT5/BetaPose/data",

  "SEGMENT_CMAP" : "inferno",
}



# _clear = True
# _verbose = False
# _tempfolder =  "/tmp"
# _usegpu = False
# _debug =  False


def logit(function):
  def add_log_info(*arg, **kwarg):
    info_message = " ".join((str(i) for i in arg))
    if _debug or _verbose:
      # create log message with timestamp and function call stack for debugging purpose
      timestamp = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
      function_stack = [i.function for i in inspect.stack()[1:-1]]
      log_message = f"{timestamp:15s}: {'->'.join(function_stack)} said: " + info_message
    else:
      timestamp = time.perf_counter() - _start_time
      log_message = f"Running {timestamp:8.2f}: " + info_message
    function(log_message, **kwarg)   # execute the decorated function
  return add_log_info


@logit
def printit(*arg, **kwarg):
  builtins.print(*arg, **kwarg)


def savelog(filename="", overwrite=True):
  # Save the log information to logfile
  if len(filename) == 0:
    filename = os.path.join(_tempfolder, "runtime.log")
  if (not os.path.exists(filename)) or (overwrite is True):
    printit(f"Runtime log saved to {filename}")
  else:
    printit(f"File {filename} exists, skip saving log file")


def draw_call_stack():
  """
  Draw the calling stack of a function for debugging purpose
  """
  printit(f"{'Drawing Calling Stack':=^100s}")
  for frame_info in inspect.stack():
    printit(f"Function: {frame_info.function:<20s} | Line: {frame_info.lineno:<5d} from File: {frame_info.filename:40}")
  printit(f"{'End Drawing Calling Stack':=^100s}")


def update_config(dict_to_update:dict={}):
  for key, value in dict_to_update.items():
    if key in CONFIG:
      CONFIG[key] = value
    else:
      raise KeyError(f"Key {key} is not in the configuration file")
  global _clear, _verbose, _tempfolder, _usegpu, _debug
  _clear = CONFIG.get("clear", True)
  _verbose = CONFIG.get("verbose", False)
  _tempfolder = CONFIG.get("tempfolder", os.path.abspath("/tmp"))
  _usegpu = CONFIG.get("usegpu", False)
  _debug = CONFIG.get("debug", False)

update_config()


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))



if (not os.path.exists(_tempfolder)) or (not os.path.isdir(_tempfolder)):
  raise OSError("The temporary folder (tempfolder) does not exist; Please check the configuration file")
elif not os.access(_tempfolder, os.W_OK):
  raise OSError("The temporary folder (tempfolder) is not writable")


msg = "Summary: "
if _clear:
  msg += "Clear temporary files; "
else:
  msg += "Keep temporary files; "
if _verbose:
  msg += "Verbose mode; "
else:
  msg += "Silent mode; "
if _usegpu:
  msg += "Using GPU acceleration; "
else:
  msg += "Using CPU only; "

if _verbose:
  printit(msg)

from nearl import tests, io, features, models, utils, data
from nearl import static
