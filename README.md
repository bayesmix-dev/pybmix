# pybmix: Bayesian Mixture Models in Python

A Python interface to [bayesmix](https://github.com/bayesmix-dev/bayesmix/)


## Installation

1. install ```cmake>=3.21.0``` from source following the official documentation
2. install ```libtbb-dev=2020.1-2``` (e.g. for Linux ```sudo apt install libtbb-dev=2020.1-2```)

3. clone the repository
```
git clone --recurse-submodules git@github.com:bayesmix-dev/pybmix.git
```

4. setup a python environment with ```python=3.9``` and install the following packages with ```pip```
```
pip3 install numpy==1.22.4 scipy==1.7.3 matplotlib==3.5.2 2to3==1.0
```

5. install ```protobuf==3.14.0``` and ```libprotobuf=3.14.0``` (e.g. ```conda install protobuf==3.14.0``` will install both)


6. add the path for ```2to3``` to the ```PATH``` environment variable, e.g.
```
export PYTHONPATH="path/to/2to3/"
```

7. finally, to build the library
```
./build_pybmix.sh build
```

Note that, the argument ```build``` substitutes ```mkdir build```, thus you can skip it in subsequent builds if only the
new changes need to be compiled.
