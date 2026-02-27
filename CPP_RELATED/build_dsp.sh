#!/bin/bash
# ======================================================================
# build_dsp.sh
#
# Builds libresnet_infer_skel.so for the Hexagon DSP.
#
# Requirements:
#   - Hexagon SDK installed (set HEXAGON_SDK_ROOT below)
#   - QNN SDK installed      (set QNN_SDK_ROOT below)
#   - IDL compiler (qaic) on PATH
#
# The resulting .so must be pushed to the device:
#   adb push libresnet_infer_skel.so /vendor/lib/rfsa/dsp/
# ======================================================================

set -euo pipefail

# ---- Paths – edit these to match your environment --------------------

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon/sdk}
HEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT:-${HEXAGON_SDK_ROOT}/tools}
QNN_SDK_ROOT=${QNN_SDK_ROOT:-/opt/qcom/aistack/qnn}

# Hexagon architecture: v68 for SM8350+, v73 for SM8550+, etc.
HEXAGON_ARCH=${HEXAGON_ARCH:-v68}

HEXAGON_CC=${HEXAGON_TOOLS_ROOT}/bin/hexagon-clang++
HEXAGON_AR=${HEXAGON_TOOLS_ROOT}/bin/hexagon-ar

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IDL_DIR="${SCRIPT_DIR}/idl"
INC_DIR="${SCRIPT_DIR}/inc"
DSP_DIR="${SCRIPT_DIR}/dsp"
BUILD_DIR="${SCRIPT_DIR}/build_dsp"

mkdir -p "${BUILD_DIR}"

# ---- Step 1: Generate stub/skel from IDL ----------------------------

echo ">>> Running qaic on resnet_infer.idl ..."

qaic -mdll \
     -I "${HEXAGON_SDK_ROOT}/incs/stddef" \
     -o "${BUILD_DIR}" \
     "${IDL_DIR}/resnet_infer.idl"

# qaic produces:
#   build_dsp/resnet_infer.h          (generated header)
#   build_dsp/resnet_infer_stub.c     (ARM-side – used by build_arm.sh)
#   build_dsp/resnet_infer_skel.c     (DSP-side – compiled here)

echo ">>> qaic done."

# ---- Step 2: Compile DSP objects -------------------------------------

COMMON_FLAGS=(
    -m${HEXAGON_ARCH}
    -O2
    -fPIC
    -I "${INC_DIR}"
    -I "${BUILD_DIR}"
    -I "${HEXAGON_SDK_ROOT}/incs"
    -I "${HEXAGON_SDK_ROOT}/incs/stddef"
    -I "${QNN_SDK_ROOT}/include"
    -I "${QNN_SDK_ROOT}/include/QNN"
    -I "${QNN_SDK_ROOT}/include/QNN/HTP"
    -I "${QNN_SDK_ROOT}/include/QNN/System"
    -D__HEXAGON__
)

echo ">>> Compiling skel ..."
${HEXAGON_CC} "${COMMON_FLAGS[@]}" -c \
    "${BUILD_DIR}/resnet_infer_skel.c" \
    -o "${BUILD_DIR}/resnet_infer_skel.o"

echo ">>> Compiling implementation ..."
${HEXAGON_CC} "${COMMON_FLAGS[@]}" -std=c++17 -c \
    "${DSP_DIR}/resnet_infer_imp.cpp" \
    -o "${BUILD_DIR}/resnet_infer_imp.o"

# ---- Step 3: Link into shared library --------------------------------

echo ">>> Linking libresnet_infer_skel.so ..."
${HEXAGON_CC} -m${HEXAGON_ARCH} -shared -fPIC \
    -o "${BUILD_DIR}/libresnet_infer_skel.so" \
    "${BUILD_DIR}/resnet_infer_skel.o" \
    "${BUILD_DIR}/resnet_infer_imp.o" \
    -L "${HEXAGON_SDK_ROOT}/libs/hexagon/${HEXAGON_ARCH}" \
    -L "${QNN_SDK_ROOT}/lib/hexagon-${HEXAGON_ARCH}/unsigned" \
    -lQnnHtp \
    -lc \
    -lm

echo ">>> Build complete: ${BUILD_DIR}/libresnet_infer_skel.so"
echo ""
echo "Push to device:"
echo "  adb push ${BUILD_DIR}/libresnet_infer_skel.so /vendor/lib/rfsa/dsp/"
