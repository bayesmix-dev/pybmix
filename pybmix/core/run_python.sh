#!/usr/bin/env bash

build/run_mcmc \
  --algo-params-file algo_python.asciipb \
  --hier-type Python --hier-args python.asciipb \
  --mix-type DP --mix-args pybmixcpp/bayesmix/resources/tutorial/dp_gamma.asciipb \
  --coll-name pybmixcpp/bayesmix/resources/tutorial/out/chains.recordio \
  --data-file pybmixcpp/bayesmix/resources/tutorial/data.csv \
  --grid-file pybmixcpp/bayesmix/resources/tutorial/grid.csv \
  --dens-file pybmixcpp/bayesmix/resources/tutorial/out/density.csv \
  --n-cl-file pybmixcpp/bayesmix/resources/tutorial/out/numclust.csv \
  --clus-file pybmixcpp/bayesmix/resources/tutorial/out/clustering.csv \
  --best-clus-file pybmixcpp/bayesmix/resources/tutorial/out/best_clustering.csv
