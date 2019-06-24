#!/bin/bash

repo=`git rev-parse --show-toplevel`

num_threads=`nproc`
j_threads=`expr $num_threads + 1`

for tc in cpp cuda cudnn;
do
  make -q -C $repo/build/tests/$tc mnm-${tc}"test"
  if [ $? -ne "2" ]; then
    make -C $repo/build/tests/$tc mnm-${tc}"test" -j$j_threads || exit -1
    find $repo/build/tests/$tc/ -maxdepth 1 -name test_\* | xargs -P$j_threads -I {} sh -c "{}"
  else
    echo ${tc} test target not enabled, skip!
  fi
done

