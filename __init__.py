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

########################################################
def logit(function):
  def adddate(*arg, **kwarg):
    timestamp = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
    builtins.print(f"{timestamp:20s}: ", end="")
    function(*arg, **kwarg)
  return adddate

@logit
def printit(*arg, **kwarg):
  builtins.print(*arg, **kwarg)


# import pytraj as pt
# import numpy as np 
# import time
# from . import utils
# from . import session_prep
