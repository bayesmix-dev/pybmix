#!/usr/bin/env bash

cd build/;
cmake ..;
make pybmixcpp;
make generate_protos;
make two_to_three;
