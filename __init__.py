import datetime, builtins

import os.path
from sys import stdout, stderr
from json import load

import pkg_resources

# import pytraj as pt
# import numpy as np
# import time
# from . import utils
# from . import session_prep
from . import test

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


########################################################
_runtimelog = []
def logit(function):
  def adddate(*arg, **kwarg):
    timestamp = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
    log_message = f"{timestamp:20s}: " + " ".join(map(str, arg))  # create log message with timestamp
    _runtimelog.append(log_message)  # append message to _runtimelog
    # builtins.print(log_message, end=" ")  # print log message
    function(log_message, **kwarg)  # execute the decorated function
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


