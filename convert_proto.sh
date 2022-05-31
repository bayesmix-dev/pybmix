SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..

# NEEDS TO BE SET MANUALLY
ENV_DIR=/home/giovanni/miniconda3/envs/pybmix_2

$ENV_DIR/bin/2to3 --output-dir=$SCRIPT_DIR/pybmix/proto -W -n $SCRIPT_DIR/pybmix/proto
