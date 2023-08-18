import datetime, builtins, os
from sys import stdout, stderr
from json import load

import pkg_resources

# from . import test
import interpolate

configfile = pkg_resources.resource_filename("BetaPose", "myconfig.json")
if os.path.isfile(configfile):
  print("Loading configuation file", file=stdout)
  with open(configfile, "r") as f: 
    CONFIG = load(f);
else: 
  print("Warning: Not found the config file", file=stderr)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

_clear = CONFIG.get("clear", False);
_verbose = CONFIG.get("verbose", False);
_tempfolder = CONFIG.get("tempfolder", "/tmp");
_usegpu = CONFIG.get("usegpu", False);
_debug = CONFIG.get("debug", False);

if (not os.path.exists(_tempfolder)) or (not os.path.isdir(_tempfolder)):
  raise OSError("The temporary folder (tempfolder) does not exist")
elif not os.access(_tempfolder, os.W_OK):
  raise OSError("The temporary folder (tempfolder) is not writable")

########################################################
_runtimelog = []
def logit(function):
  def adddate(*arg, **kwarg):
    timestamp = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
    log_message = f"{timestamp:20s}: " + " ".join(map(str, arg))  # create log message with timestamp
    _runtimelog.append(log_message)  # append message to _runtimelog
    function(log_message, **kwarg)   # execute the decorated function
  return adddate

@logit
def printit(*arg, **kwarg):
  builtins.print(*arg, **kwarg)

def savelog(filename="", overwrite=True):
  # Save the log information to logfile
  if len(filename) == 0: 
    filename = os.path.join(_tempfolder, "runtime.log")
  if (not os.path.exists(filename)) or (overwrite == True):
    printit(f"Runtime log saved to {filename}");
    with open(filename, "w") as f:
      for i in _runtimelog:
        f.write(i+"\n")
  else:
    printit(f"File {filename} exists, skip saving log file")

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
  msg += "Using GPU acceleration; ";
else:
  msg += "Using CPU only; ";
printit(msg)
