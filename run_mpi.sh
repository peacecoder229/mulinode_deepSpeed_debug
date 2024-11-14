#!/bin/bash
export PATH=$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export CONDA_PREFIX=$CONDA_PREFIX

# Run the MPI program or command
exec "$@"
