#!/bin/bash

if [ -z $1 ]; then
  repo=`git rev-parse --show-toplevel`
  find $repo/{src,tests,include,apps} \( -iname \*.cc -o -iname \*.h \) | xargs -P`nproc` clang-format -i
else
  clang-format -i $1
fi
