make -C build mnm-cpptest -j9 || exit -1

for i in build/test_*; do
  ./$i || exit -1
done
