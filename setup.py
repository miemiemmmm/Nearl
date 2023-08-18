import os, sys, subprocess, platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class CMakeExtension(Extension):
  def __init__(self, name, sourcedir='', include_dirs=None, **kwarg):
    Extension.__init__(self, name, sources=[], **kwarg)
    self.sourcedir = os.path.abspath(sourcedir)
    self.include_dirs = include_dirs or []

class CMakeBuild(build_ext):
  def run(self):
    for ext in self.extensions:
      self.build_cmake(ext)
    super().run()

  def build_cmake(self, ext):
    cwd = os.path.abspath(os.getcwd())
    build_temp = os.path.abspath(self.build_temp)
    
    cmake_args = [
      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_temp,
      '-DPYTHON_EXECUTABLE=' + sys.executable
    ]

    cfg = 'Debug' if self.debug else 'Release'
    build_args = ['--config', cfg]

    if ext.include_dirs:
      cmake_args.append('-DPYBIND11_INCLUDE_DIR=' + ';'.join(ext.include_dirs))

    if platform.system() == "Windows":
      cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), build_temp)]
      if sys.maxsize > 2 ** 32:
        cmake_args += ['-A', 'x64']

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
    subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

include_paths = [os.path.dirname(pybind11.__file__)]
print(f"The include path is {include_paths}")
setup(
  name='BetaPose',
  version='0.1',
  # packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
  packages=find_packages(),
  install_requires=[
    # list of dependencies (if any)
    'numpy',
    "pybind11", 
  ],
  package_data={
      # If any package contains *.txt or *.rst files, include them:
    "BetaPose" : ["data/PDBBind_general_v2020.csv", "data/PDBBind_refined_v2020.csv", "myconfig.json"],
  },  # Optional
  ext_modules=[
    CMakeExtension('interpolate', sourcedir='src', include_dirs=include_paths),
    CMakeExtension("testmodule", sourcedir='src', include_dirs=include_paths),
  ],
  cmdclass={'build_ext': CMakeBuild},

)
