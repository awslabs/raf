#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# You should specify a hostfile to this script with the second argument.
# For OpenMPI, the hostfile will be like:
# ```txt
# node1 slots=2
# node2 slots=2
# node3 slots=2
# node4 slots=2
# ```
# For MPICH, the hostfile will:
# ```txt
# node1:2
# node2:2
# node3:2
# node4:2
# ```

nb_gpu=$1
hostfile=$2
python_script=$3

echo "Running ${python_script} globally with ${nb_gpu} gpus."

mpirun -np $nb_gpu  --hostfile $hostfile python3 $python_script

echo "Finished running ${python_script}."
