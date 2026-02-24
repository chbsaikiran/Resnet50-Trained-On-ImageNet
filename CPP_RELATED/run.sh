#!/bin/sh

ORT=/home/saikiran/onnx_related/onnxruntime-linux-x64-1.24.2

g++ video_infer_x86.cpp \
-std=c++17 \
-I$ORT/include \
-L$ORT/lib \
-Wl,-rpath,$ORT/lib \
-lonnxruntime \
`pkg-config --cflags --libs opencv4` \
-O3 -o video_infer

./video_infer