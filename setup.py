import os, sys, subprocess, platform, shutil, importlib
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
  def __init__(self, name, sourcedir='', pybind_dir="", openacc_compiler="", **kwarg):
    Extension.__init__(self, name, sources=[], **kwarg)
    self.sourcedir = os.path.abspath(sourcedir)
    self.pybind_dir = pybind_dir or ""
    self.openacc_compiler = openacc_compiler or ""
    if len(self.openacc_compiler) > 0:
      self.compile_gpu = "ON"
    else:
      self.compile_gpu = "OFF"

class CMakeBuild(build_ext):
  def run(self):
    for ext in self.extensions:
      self.build_cmake(ext)

  def build_cmake(self, ext):
    cwd = os.path.abspath(os.getcwd())
    build_temp = os.path.abspath(self.build_temp)

    print("Current working directory: is ", cwd)
    print("Build temp is ", os.path.abspath(self.build_temp))
    print("Build lib is ", os.path.abspath(self.build_lib))

    cmake_args = [
      '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + os.path.abspath(self.build_temp),
      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(self.build_lib)+"/nearl/",
      # '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + os.path.abspath(self.build_lib)+"/nearl/",
      '-DPYTHON_EXECUTABLE=' + sys.executable
    ]

    cfg = 'Debug' if self.debug else 'Release'
    build_args = ['--config', cfg]

    if ext.pybind_dir:
      cmake_args.append('-DPYBIND11_DIR=' + ext.pybind_dir)

    cmake_args.append('-DCOMPILE_WITH_GPU=' + str(ext.compile_gpu))
    cmake_args.append('-DOPENACC_COMPILER=' + ext.openacc_compiler)

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    # print("CMake args: ", cmake_args)
    # print("Build args: ", build_args)
    # print("########### The source directory is ", ext.sourcedir)
    subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
    subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

    # Copy the built files to the source directory]
    print("Importatnt paths while building: ")
    print(f"build_temp: {self.build_temp}, build_lib: {self.build_lib}, current directory: {cwd}")
    print(f"Extensions: {self.extensions}, compilter: {self.include_dirs}, libraries: {self.library_dirs}")
    print(f"Libraries: {self.libraries} ")

    # For pip install --editable mode.
    print("########### Copy the built files to the source directory")
    for i in self.extensions:
      # print(i._file_name)
      statobjname = os.path.join(self.build_lib, "nearl", i._file_name)
      if os.path.exists(statobjname):
        shutil.copy2(statobjname, os.path.join(cwd, "nearl"))
      else:
        print("################################")
        print("Not Found the so file")
        print("################################")

    subprocess.run(["ls", "-l", build_temp])                            # TODO: Only for debug purpose
    subprocess.run(["ls", "-l", os.path.abspath(self.build_lib)])
    # Remove the Cache file for the following build
    os.remove(os.path.join(build_temp, "CMakeCache.txt"))
    print(dir(self))


if __name__ == "__main__":
  if importlib.util.find_spec("pybind11") is not None:
    print(f">>>>>>>> Found pybind11, Directly using it for building");
    path_pybind = importlib.util.find_spec("pybind11").submodule_search_locations[0]
  else:
    print(f">>>>>>>> pybind11 not found, using the local copy");
    path_pybind = os.path.dirname(os.path.realpath(__file__)) + "/external/pybind11"

  if "NEARL_BUILD_TYPE" in os.environ:
    if os.environ["NEARL_BUILD_TYPE"].upper() == "CPU":
      USE_GPU = False
    else:
      USE_GPU = True
  else:
    USE_GPU = False

  if USE_GPU:
    if shutil.which("nvc++") is not None:
      print("Using NVIDIA nvc++ compiler")
      path_openacc_compiler = shutil.which("nvc++")
      ext_modules = [
        # CMakeExtension("parent", sourcedir="./", pybind_dir=path_pybind, openacc_compiler=path_openacc_compiler),
        CMakeExtension("interpolate", sourcedir='src', pybind_dir=path_pybind, openacc_compiler=path_openacc_compiler),
        # CMakeExtension("testmodule", sourcedir='src', pybind_dir=path_pybind, openacc_compiler=path_openacc_compiler),
      ]

    elif shutil.which("pgc++") is not None:
      print("Using PGI compiler")
      path_openacc_compiler = shutil.which("pgc++")
      ext_modules = [
        # CMakeExtension("parent", sourcedir="./", pybind_dir=path_pybind, openacc_compiler=path_openacc_compiler),
        CMakeExtension("interpolate", sourcedir='src', pybind_dir=path_pybind, openacc_compiler=path_openacc_compiler),
        # CMakeExtension("testmodule", sourcedir='src', pybind_dir=path_pybind, openacc_compiler=path_openacc_compiler),
      ]

    else:
      print("No OpenACC/C++ compiler found; Falling back to CPU only")
      ext_modules = [
        # CMakeExtension("parent", sourcedir="./", pybind_dir=path_pybind),
        CMakeExtension("interpolate", sourcedir='src', pybind_dir=path_pybind),
        # CMakeExtension("testmodule", sourcedir='src', pybind_dir=path_pybind),
      ]
  else:
    print("Using CPU only mode")
    ext_modules = [
      # CMakeExtension("parent", sourcedir="./", pybind_dir=path_pybind),
      CMakeExtension("interpolate", sourcedir='src', pybind_dir=path_pybind),
      # CMakeExtension("testmodule", sourcedir='src', pybind_dir=path_pybind),
    ]
  setup(
    packages = [
      "nearl",
    ],
    ext_modules=ext_modules,
    cmdclass={
      'build_ext': CMakeBuild,

    },
  )
