#!/bin/bash
# ======================================================================
# build_arm.sh
#
# Builds the ARM-side wrapper binary that runs on the Snapdragon
# applications processor and communicates with the DSP via FastRPC.
#
# Requirements:
#   - Android NDK installed   (set NDK_ROOT below)
#   - Hexagon SDK installed   (set HEXAGON_SDK_ROOT below)
#   - OpenCV cross-compiled for aarch64-android (set OPENCV_DIR below)
#   - qaic-generated stub from build_dsp.sh must exist
#
# The resulting binary is pushed to the device:
#   adb push arm_wrapper /data/local/tmp/
# ======================================================================

set -euo pipefail

# ---- Paths – edit these to match your environment --------------------

NDK_ROOT=${NDK_ROOT:-/opt/android-ndk}
HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon/sdk}

# Android API level and toolchain triple
API_LEVEL=30
TARGET=aarch64-linux-android

CC="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/${TARGET}${API_LEVEL}-clang++"

# OpenCV cross-compiled for Android aarch64
OPENCV_DIR=${OPENCV_DIR:-/opt/opencv-android/sdk/native}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INC_DIR="${SCRIPT_DIR}/inc"
ARM_DIR="${SCRIPT_DIR}/arm"
BUILD_DSP_DIR="${SCRIPT_DIR}/build_dsp"    # contains qaic-generated stub
BUILD_DIR="${SCRIPT_DIR}/build_arm"

mkdir -p "${BUILD_DIR}"

# ---- Verify the stub was generated -----------------------------------

STUB_C="${BUILD_DSP_DIR}/resnet_infer_stub.c"
if [ ! -f "${STUB_C}" ]; then
    echo "ERROR: ${STUB_C} not found.  Run build_dsp.sh first."
    exit 1
fi

# ---- Compile ---------------------------------------------------------

COMMON_FLAGS=(
    -O2
    -std=c++17
    -I "${INC_DIR}"
    -I "${BUILD_DSP_DIR}"
    -I "${HEXAGON_SDK_ROOT}/incs"
    -I "${HEXAGON_SDK_ROOT}/incs/stddef"
    -I "${HEXAGON_SDK_ROOT}/libs/common/rpcmem/inc"
    -I "${OPENCV_DIR}/jni/include"
)

echo ">>> Compiling arm_wrapper.cpp ..."
${CC} "${COMMON_FLAGS[@]}" -c \
    "${ARM_DIR}/arm_wrapper.cpp" \
    -o "${BUILD_DIR}/arm_wrapper.o"

echo ">>> Compiling stub ..."
${CC} "${COMMON_FLAGS[@]}" -c \
    "${STUB_C}" \
    -o "${BUILD_DIR}/resnet_infer_stub.o"

# ---- Link ------------------------------------------------------------

OPENCV_LIBS=(
    -L "${OPENCV_DIR}/libs/arm64-v8a"
    -lopencv_java4            # or link individual modules
)

FASTRPC_LIBS=(
    -L "${HEXAGON_SDK_ROOT}/libs/common/remote/ship/android_aarch64"
    -ladsprpc                 # FastRPC transport (or -lcdsprpc for CDSP)
    -L "${HEXAGON_SDK_ROOT}/libs/common/rpcmem/android_aarch64"
    -lrpcmem
)

echo ">>> Linking arm_wrapper ..."
${CC} -o "${BUILD_DIR}/arm_wrapper" \
    "${BUILD_DIR}/arm_wrapper.o" \
    "${BUILD_DIR}/resnet_infer_stub.o" \
    "${OPENCV_LIBS[@]}" \
    "${FASTRPC_LIBS[@]}" \
    -llog -lz -ldl

echo ">>> Build complete: ${BUILD_DIR}/arm_wrapper"
echo ""
echo "Deploy to device:"
echo "  adb push ${BUILD_DIR}/arm_wrapper /data/local/tmp/"
echo ""
echo "Run on device:"
echo "  adb shell 'cd /data/local/tmp && ./arm_wrapper resnet18_imagenet10.bin imagenet10_val_video.mp4'"
