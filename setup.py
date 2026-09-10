import os
import shutil
import subprocess
import warnings
from typing import ClassVar

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")


# build_ext (unlike the old "install" command) runs for editable installs too.
class build_ext_nearl(_build_ext):
    # name -> (make target, requires nvcc)
    _TARGETS: ClassVar[dict[str, tuple[str, bool]]] = {
        "nearl.all_actions": ("all_actions", True),
        "nearl.host_actions": ("host_actions", False),
    }

    def build_extension(self, ext):
        if ext.name not in self._TARGETS:
            return super().build_extension(ext)
        make_target, needs_cuda = self._TARGETS[ext.name]
        if needs_cuda and shutil.which("nvcc") is None:
            warnings.warn(
                "nvcc not found on PATH; skipping the nearl.all_actions CUDA "
                "extension. GPU-based featurization/aggregation will be "
                "unavailable until it's built (see src/Makefile).",
                stacklevel=2,
            )
            return
        subprocess.check_call(["make", make_target], cwd=SRC_DIR)
        built_so = os.path.join(SRC_DIR, f"{make_target}.so")
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
    "ext_modules": [
        Extension("nearl.all_actions", sources=[]),
        Extension("nearl.host_actions", sources=[]),
    ],
    "zip_safe": False,
}


if __name__ == "__main__":
    setup(**setup_params)
