cmake_minimum_required(VERSION 3.10)
project(calc_accuracy)

include_directories(~/onnxruntime/include)

link_directories(~/onnxruntime/build/Android/Release)

include_directories(~/opencv/build_android/install/sdk/native/jni/include)

link_directories(~/opencv/build_android/lib/arm64-v8a)

add_executable(calc_accuracy arm_calculate_accuracy.cpp)
target_link_libraries(calc_accuracy
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_videoio
    onnxruntime
    log
)
