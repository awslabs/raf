#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
