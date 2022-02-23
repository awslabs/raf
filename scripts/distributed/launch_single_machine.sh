#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


nb_gpu=$1
python_script=$2

echo "Running ${python_script} locally with ${nb_gpu} gpus."

mpirun -np $nb_gpu python3 $python_script

echo "Finished running ${python_script}."
