sdl2=$(sdl2-config --cflags --static-libs)

mkdir -p ./dist/
pushd ./dist/
# clang -Wall -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 $sdl2 -o softrend ../src/main.c
clang -Wall -O2 $sdl2 -o softrend ../src/main.c
popd
