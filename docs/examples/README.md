## Installation
1) Clone the repository
```shell
git clone --recurse-submodule git@github.com:bayesmix-dev/pybmix.git
```
2) Install the following:
- ```2to3```, ```ninja```, ```numpy```, ```scipy``` with pip or conda
- ```protobuf==3.14.0``` using conda or from source following the official documentation
- ```cmake>=3.21.0``` following the official documentation
3) Assign to the variable ```ENV_DIR``` in ```convert_proto.sh``` the local path to the current Python environment

4) Run ```./build_pybmix.sh```


## To implement a hierarchy in Python
Create a ```.py``` file implementing all the necessary methods of the hierarchy, 
you can find examples in  ```docs/examples```. Specifically: 
- to implement a non-conjugate hierarchy you need to define the methods: ```is_conjugate, like_lpdf, initialize_state, initialize_hypers,
  update_hypers, draw, update_summary_statistics, sample_full_cond```. Please refer to the ```LapNIG_Hierarchy.py``` example for details.
- to implement a conjugate hierarchy you need to define the methods: ```is_conjugate, like_lpdf, marg_lpdf, initialize_state,
   initialize_hypers, update_hypers, draw, compute_posterior_hypers,
   update_summary_statistics```. Please refer to the ```NNIG_Hierarchy_NGG.py``` examples for details.

For an example of how to run please refer to ```test_run.py```