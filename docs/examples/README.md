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

For an example of how to run please refer to ```test_run.py```, ```estimate_pyhier_desnity.ipynb```.