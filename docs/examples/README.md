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

## To implement a hierarchy in Python

Create a ```.py``` file implementing all the necessary methods of the hierarchy,
you can find examples in  ```docs/examples```. Specifically:

- to implement a non-conjugate hierarchy you need to define the
  methods: ```is_conjugate, like_lpdf, initialize_state, initialize_hypers,
  update_hypers, draw, update_summary_statistics, sample_full_cond```. Please refer to the ```LapNIG_Hierarchy.py```
  example for details.
- to implement a conjugate hierarchy you need to define the
  methods: ```is_conjugate, like_lpdf, marg_lpdf, initialize_state,
  initialize_hypers, update_hypers, draw, compute_posterior_hypers,
  update_summary_statistics```. Please refer to the ```NNIG_Hierarchy_NGG.py``` examples for details.

## To implement a mixing in Python
Create a ```.py``` file in ```pybmix/docs/examples``` implementing all the necessary methods of the mixing,
you can find an example in  ```docs/examples```. Specifically:
- to implement a non-conditional mixing you need to define the methods:
  ```is_conditional, update_state, initialize_state, mass_existing_cluster, mass_new_cluster```.
  Please refer to the ```DP_mixing.py``` example for details.
- to implement a conditional mixing you need to define the methods:
  ```is_conditional, update_state, initialize_state, mixing_weights```.

For working examples please refer to ```test_run.py``` and ```estimate_pyhier_desnity.ipynb```.