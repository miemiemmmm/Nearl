import pkg_resources, os, sys, json, datetime, builtins

configfile = pkg_resources.resource_filename("BetaPose", "myconfig.json")
if os.path.isfile(configfile): 
  with open(configfile, "r") as f: 
    CONFIG = json.load(f)
else: 
  print("Not found the config file", file=sys.stderr)

_clear = CONFIG.get("clear", False);
_verbose = CONFIG.get("verbose", False);
_tempfolder = CONFIG.get("tempfolder", "/tmp");

from . import test
_runtimelog = []

########################################################
def logit(function):
  def adddate(*arg, **kwarg):
    timestamp = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
    log_message = f"{timestamp:20s}: " + " ".join(map(str, arg))  # create log message with timestamp
    _runtimelog.append(log_message)  # append message to _runtimelog
    builtins.print(log_message, end=" ")  # print log message
    function(*arg, **kwarg)  # execute the decorated function
  return adddate
  # def adddate(*arg, **kwarg):
  #   timestamp = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
  #   builtins.print(f"{timestamp:20s}: ", end="")
  #   function(*arg, **kwarg)
  # return adddate

@logit
def printit(*arg, **kwarg):
  builtins.print(*arg, **kwarg)

def savelog(filename, overwrite=False):
  # Save the log information to logfile
  if (not os.path.exists(filename)) or overwrite:
    with open(filename, "w") as f:
      for i in _runtimelog:
        f.write(i+"\n")

# import pytraj as pt
# import numpy as np 
# import time
# from . import utils
# from . import session_prep
