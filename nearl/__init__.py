import datetime, builtins, os, inspect
from json import load
import importlib.resources as resources

########################################################

__version__ = "0.0.1"
_runtimelog = []
configfile = resources.files("nearl").joinpath("../CONFIG.json")
if os.path.isfile(configfile):
  configfile = os.path.abspath(configfile)
  with open(configfile, "r") as f:
    CONFIG = load(f)
  _clear = CONFIG.get("clear", False)
  _verbose = CONFIG.get("verbose", False)
  _tempfolder = CONFIG.get("tempfolder", "/tmp")
  _usegpu = CONFIG.get("usegpu", False)
  _debug = CONFIG.get("debug", False)
else:
  raise FileNotFoundError(f"NEARL({__file__}): Not found the configuration file")


def logit(function):
  def add_log_info(*arg, **kwarg):
    timestamp = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
    function_stack = [i.function for i in inspect.stack()[1:-1]]
    # create log message with timestamp and function call stack for debugging purpose
    if _debug:
      log_message = f"{timestamp:15s}: {'->'.join(function_stack)} said: " + " ".join(map(str, arg))
    else:
      log_message = f"{timestamp:15s}: {'->'.join(function_stack[:2])} said: " + " ".join(map(str, arg))
    # _runtimelog.append(log_message)  # append message to _runtimelog
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
    with open(filename, "w") as file:
      for i in _runtimelog:
        file.write(i+"\n")
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
