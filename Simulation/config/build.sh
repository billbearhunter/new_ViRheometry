#!/bin/zsh

cd ..

if [ -d "./ParticleSkinner3DTaichi/cpp_marching_cubes/build" ]; then
 rm -rf ./ParticleSkinner3DTaichi/cpp_marching_cubes/build
fi
mkdir ./ParticleSkinner3DTaichi/cpp_marching_cubes/build
cd ./ParticleSkinner3DTaichi/cpp_marching_cubes/build
cmake ..
make
cd ../../../

self_dir=$(cd $(dirname $0); pwd)
