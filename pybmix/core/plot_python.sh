#!/usr/bin/env bash

build/plot_mcmc \
  --grid-file pybmixcpp/bayesmix/resources/tutorial/grid.csv \
  --dens-file pybmixcpp/bayesmix/resources/tutorial/out/density.csv \
  --dens-plot pybmixcpp/bayesmix/resources/tutorial/out/density.png \
  --n-cl-file pybmixcpp/bayesmix/resources/tutorial/out/numclust.csv \
  --n-cl-trace-plot pybmixcpp/bayesmix/resources/tutorial/out/traceplot.png  \
  --n-cl-bar-plot  pybmixcpp/bayesmix/resources/tutorial/out/nclus_barplot.png
