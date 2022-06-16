#!/usr/bin/env bash

cd build/;
cmake .. -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON -DDISABLE_TESTS=ON -DDISABLE_EXAMPLES=ON;
make pybmixcpp;
make generate_protos;
make two_to_three;
