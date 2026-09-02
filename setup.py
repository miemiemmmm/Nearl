import os
import shutil
import subprocess
import warnings

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")


# build_ext (unlike the old "install" command) runs for editable installs too.
class build_ext_nearl(_build_ext):
    def build_extension(self, ext):
        if ext.name != "nearl.all_actions":
            return super().build_extension(ext)
        if shutil.which("nvcc") is None:
            warnings.warn(
                "nvcc not found on PATH; skipping the nearl.all_actions CUDA "
                "extension. GPU-based featurization/aggregation will be "
                "unavailable until it's built (see src/Makefile).",
                stacklevel=2,
            )
            return
        subprocess.check_call(["make", "all_actions"], cwd=SRC_DIR)
        built_so = os.path.join(SRC_DIR, "all_actions.so")
        if not os.path.isfile(built_so):
            raise RuntimeError(
                f"The shared object file ({built_so}) was not produced; please check the build process"
            )
        target = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        self.copy_file(built_so, target)


setup_params = {
    "cmdclass": {"build_ext": build_ext_nearl},
    "packages": find_packages(),
    "ext_modules": [Extension("nearl.all_actions", sources=[])],
    "zip_safe": False,
}


if __name__ == "__main__":
    setup(**setup_params)
