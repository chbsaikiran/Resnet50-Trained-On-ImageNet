#!/bin/sh

ORT=/home/saikiran/onnx_related/onnxruntime-linux-x64-1.24.2

g++ calculate_accuracy.cpp \
-std=c++17 \
-I$ORT/include \
-L$ORT/lib \
-Wl,-rpath,$ORT/lib \
-lonnxruntime \
`pkg-config --cflags --libs opencv4` \
-O3 -o calculate_accuracy

./calculate_accuracy "$@"
