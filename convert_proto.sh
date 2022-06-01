SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..

# NEEDS TO BE SET MANUALLY
ENV_DIR=/home/taguhi/miniconda3/envs/pybmix3.9

$ENV_DIR/bin/2to3 --output-dir=$SCRIPT_DIR/pybmix/proto -W -n $SCRIPT_DIR/pybmix/proto
