import glob
import os
import platform
import shutil
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.command.build_py import build_py as _build_py
from distutils.command.clean import clean as _clean
from distutils.spawn import find_executable
from setuptools.command.develop import develop as _develop
from setuptools.command.egg_info import egg_info as _egg_info
from distutils.command.install import install as _install

PYBMIXCPP_PATH = os.path.join("pybmix", "core", "pybmixcpp")
BAYEXMIX_PATH = os.path.join(PYBMIXCPP_PATH , "bayesmix")
PROTO_IN_DIR = os.path.join(BAYEXMIX_PATH, "proto")
PROTO_OUT_DIR = os.path.join("pybmix", "proto/")

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}



# Find the Protocol Compiler.
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = find_executable("protoc")

py2to3 = find_executable("2to3")


def generate_proto(source, require = True):
  """Invokes the Protocol Compiler to generate a _pb2.py from the given
  .proto file.  Does nothing if the output already exists and is newer than
  the input."""

  print("Generating Proto: ", source)

  if not require and not os.path.exists(source):
    return
  
  filename = os.path.split(source)[1]
  output = os.path.join(PROTO_OUT_DIR, filename.replace(".proto", "_pb2.py"))

  if (not os.path.exists(output) or
      (os.path.exists(source) and
       os.path.getmtime(source) > os.path.getmtime(output))):
    print("Generating %s..." % output)

    if not os.path.exists(source):
      sys.stderr.write("Can't find required file: %s\n" % source)
      sys.exit(-1)

    if protoc is None:
      sys.stderr.write(
          "protoc is not installed nor found in ../src.  Please compile it "
          "or install the binary package.\n")
      sys.exit(-1)

    protoc_command = [protoc, "--proto_path={0}".format(PROTO_IN_DIR), 
                      "--python_out={0}".format(PROTO_OUT_DIR), source]
    print(" ".join(protoc_command))
    if subprocess.call(protoc_command) != 0:
      sys.exit(-1)


class clean(_clean):
  def run(self):
    # Delete generated files in the code tree.
    for (dirpath, dirnames, filenames) in os.walk("."):
      for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        if filepath.endswith("_pb2.py") or filepath.endswith(".pyc") or \
          filepath.endswith(".so") or filepath.endswith(".o"):
          os.remove(filepath)
    # _clean is an old-style class, so super() doesn't work.
    _clean.run(self)

def generate_all_protos():
    proto_files = glob.glob(os.path.join(PROTO_IN_DIR, "*.proto"))
    for file in proto_files:
        generate_proto(file)
    
    two_to_three_command = [
        py2to3, "--output-dir={0}".format(PROTO_OUT_DIR), "-W", "-n", PROTO_OUT_DIR]
    print(" ".join(two_to_three_command))
    if subprocess.call(two_to_three_command) != 0:
      sys.exit(-1)


def build_tbb():
    """Build tbb. This function is taken from
    https://github.com/stan-dev/pystan/blob/develop/setup.py"""
   
    stan_math_lib = os.path.abspath(os.path.join(os.path.dirname(
        __file__), BAYEXMIX_PATH, 'lib', 'math', 'lib'))

    make = 'make' if platform.system() != 'Windows' else 'mingw32-make'
    cmd = [make]

    tbb_root = os.path.join(stan_math_lib, 'tbb_2019_U8').replace("\\", "/")

    cmd.extend(['-C', tbb_root])
    cmd.append('tbb_build_dir={}'.format(stan_math_lib))
    cmd.append('tbb_build_prefix=tbb')
    cmd.append('tbb_root={}'.format(tbb_root))

    cmd.append('stdver=c++14')

    cmd.append('compiler=gcc')

    cwd = os.path.abspath(os.path.dirname(__file__))

    subprocess.check_call(cmd, cwd=cwd)

    tbb_debug = os.path.join(stan_math_lib, "tbb_debug")
    tbb_release = os.path.join(stan_math_lib, "tbb_release")
    tbb_dir = os.path.join(stan_math_lib, "tbb")

    if not os.path.exists(tbb_dir):
        os.makedirs(tbb_dir)

    if os.path.exists(tbb_debug):
        shutil.rmtree(tbb_debug)

    shutil.move(os.path.join(tbb_root, 'include'), tbb_dir)
    shutil.rmtree(tbb_root)

    for name in os.listdir(tbb_release):
        srcname = os.path.join(tbb_release, name)
        dstname = os.path.join(tbb_dir, name)
        shutil.move(srcname, dstname)

    if os.path.exists(tbb_release):
        shutil.rmtree(tbb_release)


class build_py(_build_py):
  def run(self):
    generate_all_protos()
    _build_py.run(self)


class install(_install):
    def run(self):
        generate_all_protos()
        _install.run(self)


class develop(_develop):
    def run(self):
        generate_all_protos()
        _develop.run(self)


class egg_info(_egg_info):
    def run(self):
        generate_all_protos()
        _egg_info.run(self)


# A CMakeExtension needs a sourcedir instead of a file list.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # This is optional - will print a nicer error if CMake is missing.
        # Since we force CMake via PEP 518 in the pyproject.toml, this should
        # never happen and this whole method can be removed in your code if you
        # want.
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            msg = "CMake missing - probably upgrade to a newer version of Pip?"
            raise RuntimeError(msg)

        # To support Python 2, we have to avoid super(), since distutils is all
        # old-style classes.
        build_ext.run(self)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DDISABLE_TESTS=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        ]
        build_args = []

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                cmake_args += ["-GNinja"]

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )

        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


if __name__ == "__main__":

    # Build tbb before setup if needed
    tbb_dir = os.path.join(BAYEXMIX_PATH, 'lib', 'math', 'lib', 'tbb')
    tbb_dir = os.path.abspath(tbb_dir)
    print("tbb_dir: ", tbb_dir)
    if not os.path.exists(tbb_dir):
        build_tbb()

    setup(
        name="pybmix",
        version="0.0.1",
        author="Mario Beraha",
        author_email="berahamario@gmail.com",
        description="Python Bayesian Mixtures",
        long_description="",
        packages=find_packages(),
        ext_modules=[CMakeExtension('pybmix.core.pybmixcpp')],
        cmdclass={
            "egg_info": egg_info,
            "build_py": build_py,
            "clean": clean,
            "build_ext": CMakeBuild,
            },
        install_requires=[
            "cmake",
            "ninja",
            "numpy",
            "scipy",
            "protobuf==3.14.0"
        ],
        zip_safe=False,
    )
