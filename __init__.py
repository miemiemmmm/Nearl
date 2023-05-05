import pkg_resources, os, sys, json

configfile = pkg_resources.resource_filename("BetaPose", "myconfig.json")
if os.path.isfile(configfile): 
  with open(configfile, "r") as f: 
    CONFIG = json.load(f)
else: 
  print("Not found the config file", file=sys.stderr)

from . import test

# import pytraj as pt
# import numpy as np 
# import time
# from . import utils
# from . import session_prep
