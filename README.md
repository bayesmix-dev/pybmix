# Bayesian Mixture Models in Python


## Prerequisites

- Protocol buffers
- cmake


## Installation

First, clone this repository with its submodules
```shell
git clone --recurse-submodule git@github.com:bayesmix-dev/pybmix.git
```
Ideally things should work by simply typing
```shell
pip3 install -e .
```
from the root folder of this repo.


# Structure

This repo contains two main directories: `pybmix` and `pybmixcpp`.

`pybmixcpp/` contains a copy of the C++ library `bayesmix` and the code for 
a small python package that is the raw interface between  `bayesmix` and Python.

`pybmix/` contains the Python package.
