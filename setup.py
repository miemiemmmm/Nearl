import os, sys, subprocess, platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
# import pybind11

__version__ = "0.1"

pybinddir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external/pybind11')
sys.path.append(pybinddir)
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

    print("temp build dir: ",build_temp)
    print("CMake args: ", cmake_args)
    print("Build args: ", build_args)
    subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
    subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
    subprocess.run(["ls", "-l", build_temp]); # debug
# include_paths = [os.path.join(os.path.abspath(os.getcwd()), "external/pybind11")]
# include_paths = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external/pybind11')];
include_paths = [os.path.dirname(pybind11.__file__)]
print(f"The include path is {include_paths}")

setup(
  name='BetaPose',
  version=__version__,
  author="Yang Zhang",
  author_email="y.zhang@bioc.uzh.ch",
  packages = [
    "BetaPose",
  ],
  ext_modules=[
    CMakeExtension("parent", include_dirs=include_paths),
    CMakeExtension("BetaPose.interpolate", sourcedir='src', include_dirs=include_paths),
    CMakeExtension("BetaPose.testmodule", sourcedir='src', include_dirs=include_paths),
  ],
  cmdclass={'build_ext': CMakeBuild},
  python_requires=">=3.7",
)
