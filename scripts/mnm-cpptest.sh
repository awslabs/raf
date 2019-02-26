#!/bin/bash

repo=`git rev-parse --show-toplevel`

num_threads=`nproc`
j_threads=`expr $num_threads + 1`

make -C $repo/build/tests/cpp mnm-cpptest -j$j_threads || exit -1

find $repo/build/tests/cpp/ -maxdepth 1 -name test_\* | xargs -P$j_threads -I {} sh -c "{}"
