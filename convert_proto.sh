SCRIPT_DIR=$(pwd)/..

# 2to3 must be in the path (e.g. export PYTHONPATH="path/to/2to3/")
2to3 --output-dir=$SCRIPT_DIR/pybmix/proto -W -n $SCRIPT_DIR/pybmix/proto
