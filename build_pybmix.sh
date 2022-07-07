#!/usr/bin/env bash

if [ -z "$1" ]; then
  cd build/ || exit
  cmake .. -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON -DDISABLE_TESTS=ON -DDISABLE_EXAMPLES=ON
  make pybmixcpp
  make generate_protos
  make two_to_three
else
  if [[ $1 == build ]]; then
    rm -rf $1
    mkdir $1
    cd $1 || exit
    cmake .. -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON -DDISABLE_TESTS=ON -DDISABLE_EXAMPLES=ON
    make pybmixcpp
    make generate_protos
    make two_to_three
  else
    echo "wrong argument, pass build"
  fi
fi
