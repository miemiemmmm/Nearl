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
        if ext.name == "nearl.all_actions":
            return self._build_cuda(ext)
        if ext.name == "nearl._voxelize_cpu":
            return self._build_cpu(ext)
        return super().build_extension(ext)

    def _install(self, ext, built_so):
        if not os.path.isfile(built_so):
            raise RuntimeError(
                f"The shared object file ({built_so}) was not produced; please check the build process"
            )
        target = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        self.copy_file(built_so, target)

    def _build_cuda(self, ext):
        if shutil.which("nvcc") is None:
            warnings.warn(
                "nvcc not found on PATH; skipping the nearl.all_actions CUDA "
                "extension. GPU-based featurization/aggregation will be "
                "unavailable until it's built (see src/Makefile). Voxelization "
                "falls back to nearl._voxelize_cpu.",
                stacklevel=2,
            )
            return
        subprocess.check_call(["make", "all_actions"], cwd=SRC_DIR)
        self._install(ext, os.path.join(SRC_DIR, "all_actions.so"))

    # No nvcc and no CUDA runtime, so this one builds everywhere.
    def _build_cpu(self, ext):
        subprocess.check_call(["make", "voxelize_cpu"], cwd=SRC_DIR)
        self._install(
            ext,
            os.path.join(
                SRC_DIR,
                os.path.basename(self.get_ext_filename(ext.name.rsplit(".", 1)[-1])),
            ),
        )


setup_params = {
    "cmdclass": {"build_ext": build_ext_nearl},
    "packages": find_packages(),
    "ext_modules": [
        Extension("nearl.all_actions", sources=[]),
        Extension("nearl._voxelize_cpu", sources=[]),
    ],
    "zip_safe": False,
}


if __name__ == "__main__":
    setup(**setup_params)
