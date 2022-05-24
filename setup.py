import os
import re
import subprocess
import sys

from distutils.spawn import find_executable
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py as _build_py
from distutils.command.clean import clean as _clean


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


py2to3 = find_executable("2to3")

HERE = os.path.abspath('.')
PROTO_OUT_DIR = os.path.join(HERE, "pybmix/proto/")


class build_py(_build_py):
  def run(self):
      build_dir = os.path.join(HERE, "build")
      os.makedirs(build_dir, exist_ok=True)

      cmake_args = [
            f"-DDISABLE_BENCHMARKS=TRUE",
            f"-DDISABLE_PLOTS=TRUE",
            f"-DDISABLE_EXAMPLES=TRUE",
            f"-DDISABLE_DOCS=TRUE",
            f"-DDISABLE_TESTS=TRUE",
        ]

      subprocess.check_call(["cmake", ".."] + cmake_args, cwd=build_dir)
      subprocess.check_call(["make", "generate_protos"], cwd=build_dir)
      two_to_three_command = [
            py2to3, "--output-dir={0}".format(PROTO_OUT_DIR), "-W", "-n", PROTO_OUT_DIR]
      print("********* CALLING 2to3 ***********")
      print(" ".join(two_to_three_command))
      if subprocess.call(two_to_three_command) != 0:
        sys.exit(-1)

      _build_py.run(self)

class clean(_clean):
  def run(self):
    # Delete generated files in the code tree.
    for (dirpath, dirnames, filenames) in os.walk("."):
      for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        if filepath.endswith("_pb2.py"):
          os.remove(filepath)
    # _clean is an old-style class, so super() doesn't work.
    _clean.run(self)


if __name__ == "__main__":

    setup(
        name="pybmix",
        version="0.0.1",
        author="Mario Beraha",
        author_email="berahamario@gmail.com",
        description="Python Bayesian Mixtures",
        long_description="",
        cmdclass={"build_py": build_py, "clean": clean},
        zip_safe=False,
        python_requires=">=3.6",
    )