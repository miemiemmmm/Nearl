import os
import subprocess
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install

class install_Nearl(install):
  def run(self): 
    print("#### Running the install command")
    os.chdir("src")
    subprocess.check_call(["make", "all_actions"])
    os.chdir("..")
    if os.path.isfile("./src/all_actions.so"):
      if os.path.isdir("./build/lib/nearl"):
        shutil.copy2("./src/all_actions.so", "./build/lib/nearl/all_actions.so")
      elif os.path.isdir("./nearl"):
        shutil.copy2("./src/all_actions.so", "./nearl/all_actions.so")
      else: 
        raise Exception("The build directory is not found; Please check the build process")
    else:
      raise Exception("The shared object file (all_actions.so) is not found; Please check the build process")
    install.run(self)


setup_params = dict(
  cmdclass = {"install": install_Nearl},
  packages = find_packages(),
  zip_safe = False,
)


if __name__ == "__main__":
  setup(**setup_params)



