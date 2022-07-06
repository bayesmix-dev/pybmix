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