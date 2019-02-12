for folders in src tests include apps
do
  for suffix in c h hh cc cpp hpp
  do
    to_format=`find ./$folders/ -name \*.$suffix`
    if [ ! -z "$to_format" ];
    then
      find ./$folders/ -name \*.$suffix | xargs clang-format -i
    fi
  done
done
