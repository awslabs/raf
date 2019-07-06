MNM
================

## Compile
```bash
git clone https://github.com/were/mnm --recursive
mkdir -p mnm/build && cd mnm/build
cp ../cmake/config.cmake .
cmake .. -DCMAKE_CXX_FLAGS="-Wall"  # add whatever flag you want
make -j
```
