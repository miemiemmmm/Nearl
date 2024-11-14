import sys

import_fail = False

try:
  import nearl
  import nearl.tests.test_actions
  import nearl.tests.test_features
  import nearl.tests.test_trajectory
  import nearl.tests.test_visualization
  import nearl.tests.test_training
except ImportError as e:
  import_fail = True
  error_msg = str(e) 


def main(): 
  if import_fail:
    print("ImportError: ", error_msg, file=sys.stderr)
    print("Please check if the package is installed correctly.", file=sys.stderr)
    sys.exit(1)
  
  print(f"Nearl version {nearl.__version__}")
  print("Testing the installation...")

  modules_to_test = [
    nearl.tests.test_actions,
    nearl.tests.test_features,
    nearl.tests.test_trajectory,
    nearl.tests.test_visualization,
    nearl.tests.test_training, 
  ]

  c = 0
  fname = ""
  module = ""
  try: 
    for module in modules_to_test:
      for fname in dir(module):
        if fname.startswith("test_benchmark"):
          continue
        elif fname.startswith("test_"): 
          func_obj = getattr(module, fname)
          if callable(func_obj):
            print(f"Performing test {c+1} - {fname:30s}: ", end="")
            func_obj()
            c += 1
        else:
          continue
    print(f"Installation validation successful: {c} tests passed.")
  except Exception as e:
    print(f"Installation validation failed when performing the test: {module}.{fname}", file=sys.stderr)
    print("Error: ", str(e), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
