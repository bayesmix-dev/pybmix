SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..

# NEEDS TO BE SET MANUALLY
TWO_TO_THREE_PATH=/home/taguhi/miniconda3/envs/pybmix_foo/bin/2to3

$TWO_TO_THREE_PATH --output-dir=$SCRIPT_DIR/pybmix/proto -W -n $SCRIPT_DIR/pybmix/proto
