import os, sys, time, logging
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

_package_logger = logging.getLogger("nearl")
_package_logger.setLevel(logging.DEBUG)
_package_logger.propagate = False

class _ColorStreamHandler(logging.StreamHandler):
  _colors = {logging.WARNING: "\033[93m", logging.ERROR: "\033[91m"}
  def format(self, record):
    msg = super().format(record)
    color = self._colors.get(record.levelno)
    return f"{color}{msg}\033[0m" if color and self.stream.isatty() else msg

_stdout_handler = _ColorStreamHandler(sys.stdout)
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)
_stdout_handler.setFormatter(logging.Formatter("%(message)s"))
_stderr_handler = _ColorStreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)
_stderr_handler.setFormatter(logging.Formatter("%(message)s"))
_package_logger.addHandler(_stdout_handler)
_package_logger.addHandler(_stderr_handler)

_file_handler = None
def _ensure_file_handler():
  global _file_handler
  if _file_handler is None:
    logpath = os.path.join(os.path.abspath(config.tempfolder()), f"nearl.{os.getpid()}.log")
    _file_handler = logging.FileHandler(logpath)
    _file_handler.setFormatter(logging.Formatter("%(message)s"))
    _package_logger.addHandler(_file_handler)
  return _file_handler

def log(*arg, **kwarg):
  _ensure_file_handler()
  if config.debug():
    log_msg = loginfo_debug()
  elif config.verbose() or config.reportdatetime():
    log_msg = loginfo_datetime()
  else:
    log_msg = loginfo_runtime()

  message = log_msg + " ".join((str(i) for i in arg))
  lowered = message.lower()
  level = logging.WARNING if "warning" in lowered else logging.ERROR if "error" in lowered else logging.INFO

  override_stream = kwarg.pop("file", None)
  if override_stream is not None:
    color = _ColorStreamHandler._colors.get(level)
    out = f"{color}{message}\033[0m" if color and override_stream.isatty() else message
    __builtinprint(out, file=override_stream, **kwarg)
    _file_handler.handle(_package_logger.makeRecord(_package_logger.name, level, __file__, 0, message, None, None))
  else:
    _package_logger.log(level, message)


def summary(): 
  log(f"Summary: {'Clear' if config.clear() else 'Keep'} temporary files; {'Verbose' if config.verbose() else 'Silent'} mode; {'Using GPU' if config.usegpu() else 'Using CPU only'}")
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
  log(f"{'Drawing Calling Stack':=^100s}")
  for frame_info in __call_stack():
    log(f"Function: {frame_info.function:<20s} | Line: {frame_info.lineno:<5d} from File: {frame_info.filename:40}")
  log(f"{'End Drawing Calling Stack':=^100s}")


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
    if not os.path.exists("example_data.tar.gz"):
      log(f"Downloading example data to {path}")
      datafile_url = "https://miemiemmmm.b-cdn.net/shared_files/example_data.tar.gz" 
      subprocess.run(["wget", "--directory-prefix", path, datafile_url]) 
    else: 
      log("The example data (example_data.tar.gz) already exists! Skip downloading...") 
    
    # Extract the compressed file 
    if os.path.exists("example_data"):
      log("The example data folder already exists! Skip extracting...")
    else:
      log("Extracting the example data...")
      subprocess.run(["tar", "-xf", "example_data.tar.gz"], cwd=path) 
    
    # Obtain the data paths as a dictionary
    os.chdir("example_data")
    if not os.path.exists("data.py"):
      raise OSError(f"Data index file not found in the extracted folder {os.getcwd()}")
    # NOTE: Load data.py by explicit file path rather than `import data`.
    # `import data` depends on the extracted folder being on sys.path, and 
    # it collides with the data directory in Nearl. 
    import importlib.util
    data_index_path = os.path.join(os.getcwd(), "data.py")
    spec = importlib.util.spec_from_file_location("nearl_example_data_index", data_index_path)
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    paths = data_module.get_data()

    return paths

