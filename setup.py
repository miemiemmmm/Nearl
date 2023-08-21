import os, sys, subprocess, platform, shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
# import pybind11

class CMakeExtension(Extension):
  def __init__(self, name, sourcedir='', pybind_dir="", pgcpp="", **kwarg):
    Extension.__init__(self, name, sources=[], **kwarg)
    self.sourcedir = os.path.abspath(sourcedir)
    self.pybind_dir = pybind_dir or ""
    self.pgcpp = pgcpp or ""
    if len(self.pgcpp) > 0:
      self.compile_gpu = 1;
    else:
      self.compile_gpu = 0;

class CMakeBuild(build_ext):
  def run(self):
    for ext in self.extensions:
      self.build_cmake(ext)

  def build_cmake(self, ext):
    cwd = os.path.abspath(os.getcwd())
    build_temp = os.path.abspath(self.build_temp)
    
    cmake_args = [
      # '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_temp,
      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(self.build_lib)+"/BetaPose/",
      '-DPYTHON_EXECUTABLE=' + sys.executable
    ]

    cfg = 'Debug' if self.debug else 'Release'
    build_args = ['--config', cfg]

    if ext.pybind_dir:
      cmake_args.append('-DPYBIND11_DIR=' + ext.pybind_dir)
    if ext.compile_gpu:
      cmake_args.append('-DCOMPILE_GPU=' + str(ext.compile_gpu))
      cmake_args.append('-DPGCPP=' + ext.pgcpp)

    if platform.system() == "Windows":
      cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), build_temp)]
      if sys.maxsize > 2 ** 32:
        cmake_args += ['-A', 'x64']

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    # print("CMake args: ", cmake_args)
    # print("Build args: ", build_args)
    # print("########### The source directory is ", ext.sourcedir);
    subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp);
    subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp);

    # Copy the built files to the source directory]
    print("Importatnt paths while building: ")
    print(f"build_temp: {self.build_temp}, build_lib: {self.build_lib}, current directory: {cwd}")
    print(f"Extensions: {self.extensions}, compilter: {self.include_dirs}, libraries: {self.library_dirs}")
    print(f"Libraries: {self.libraries} ")

    # For pip install --editable mode.
    print("########### Copy the built files to the source directory");
    for i in self.extensions:
      # print(i._file_name)
      statobjname = os.path.join(self.build_lib, "BetaPose", i._file_name)
      if os.path.exists(statobjname):
        shutil.copy2( statobjname, os.path.join(cwd, "BetaPose"));
      else:
        print("################################")
        print("Not Found the so file")
        print("################################")

    subprocess.run(["ls", "-l", build_temp]);  # debug
    subprocess.run(["ls", "-l", os.path.abspath(self.build_lib)]);
    # Remove the Cache file for the following build
    os.remove(os.path.join(build_temp, "CMakeCache.txt"));
    print(dir(self));

# include_paths = [os.path.join(os.path.abspath(os.getcwd()), "external/pybind11")]
# include_paths = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external/pybind11')];

path_pgcpp = subprocess.check_output(['which', "pgc++"], text=True).strip();
path_pybind = os.path.dirname(os.path.realpath(__file__))+ "/external/pybind11";

setup(
  author="Yang Zhang" ,
  author_email="y.zhang@bioc.uzh.ch",
  python_requires=">=3.7",
  packages = [
    "BetaPose",
  ],
  ext_modules=[
    # CMakeExtension("BetaPose.parent", sourcedir = "./", include_dirs=include_paths),
    CMakeExtension("parent", sourcedir = "./", pybind_dir=path_pybind),
    CMakeExtension("interpolate", sourcedir='src', pybind_dir=path_pybind, pgcpp=path_pgcpp),
    CMakeExtension("testmodule", sourcedir='src', pybind_dir=path_pybind, pgcpp=path_pgcpp),
  ],
  cmdclass={'build_ext': CMakeBuild},

)

