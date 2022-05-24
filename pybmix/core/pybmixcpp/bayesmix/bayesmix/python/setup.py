import os
import setuptools
import site
import sys
site.ENABLE_USER_SITE = '--user' in sys.argv[1:]

__version__ = "0.0.1"
folder = os.path.dirname(__file__)
path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.exists(path):
  with open(path) as fp:
    install_requires = [line.strip() for line in fp]


setuptools.setup(version=__version__,
                 install_requires=install_requires)
