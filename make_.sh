#!/usr/bin/env bash

cd build/;
make pybmixcpp;
make generate_protos;
make two_to_three;
