
# Android NDK Setup & Native Build Guide (Linux)

This guide explains how to install and configure the **Android NDK (Native Development Kit)** on Linux, build native C/C++ code for Android architectures, and prepare dependencies such as OpenCV and ONNX Runtime.

---

## 📌 1. What is Android NDK?

The **Android NDK (Native Development Kit)** allows you to:

* Write performance-critical code in **C/C++**
* Cross-compile for Android architectures:

  * `armeabi-v7a` (ARMv7)
  * `arm64-v8a` (ARMv8-A) ← Most modern devices
  * `x86`
  * `x86_64`

### 📦 Output Types

* **Shared libraries (`.so`)**

  * Used inside Android applications
* **Standalone native executables**

  * Can be pushed and run using `adb shell`

### 🛠 NDK Provides

* Clang cross-compiler
* Android sysroot (headers + libc)
* Linker
* Build tools:

  * CMake
  * ndk-build

---

# 🚀 2. Installing Android SDK & NDK (Command Line Only)

If you do not want to install full Android Studio:

### Step 1: Download Command Line Tools

Download from:

```
https://developer.android.com/studio
```

Extract to:

```bash
~/Android/cmdline-tools/
```

---

### Step 2: Setup Environment Variables

Add to your shell (e.g., `~/.bashrc`):

```bash
export ANDROID_HOME=$HOME/Android
export PATH=$ANDROID_HOME/cmdline-tools/bin:$PATH
```

Reload:

```bash
source ~/.bashrc
```

---

### Step 3: Install Required SDK Components

```bash
sdkmanager --install "platform-tools"
sdkmanager --install "platforms;android-34"
sdkmanager --install "ndk;26.1.10909125"
```

Check NDK location:

```bash
~/Android/ndk/26.1.10909125
```

---

# 🔧 3. Understanding the NDK Toolchain

Inside NDK:

```
toolchains/llvm/prebuilt/linux-x86_64/bin/
```

Important compiler example:

```
aarch64-linux-android21-clang
```

### Meaning:

| Component   | Description          |
| ----------- | -------------------- |
| `aarch64`   | ARM 64-bit           |
| `android21` | Minimum API level 21 |
| `clang`     | Compiler             |

---

# 🏗 4. Building a Native Executable with CMake

## Example `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(hello)

add_executable(hello hello.c)
```

---

## Build Steps

```bash
mkdir build
cd build

cmake .. \
 -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=21

make
```

This will generate an Android-compatible executable.

---

# 🖥 5. Installing Required Tools on Linux

## Install OpenJDK 17

```bash
sudo apt update
sudo apt install openjdk-17-jdk
```

---

## Install Latest CMake

Download from:

```
https://cmake.org/download/
```

Then:

```bash
chmod +x cmake-3.29.6-linux-x86_64.sh
sudo ./cmake-3.29.6-linux-x86_64.sh --skip-license --prefix=/usr/local
```

---

# 🌍 6. Recommended `.bashrc` Configuration

Edit:

```bash
vim ~/.bashrc
```

Add at the end:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

export ANDROID_HOME=$HOME/Android
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH

export PATH=/usr/local/bin:$PATH
```

Reload:

```bash
source ~/.bashrc
```

---

# 📚 7. Building OpenCV and ONNX Runtime for Android

Clone required repositories:

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/microsoft/onnxruntime.git
```

### ⚠ Important

When building:

* ✅ Build as **Shared Libraries**
* ✅ Use **Release mode**
* ❌ Do NOT build in Debug (wastes build time and increases size)

Example build type:

```bash
-DCMAKE_BUILD_TYPE=Release
-DBUILD_SHARED_LIBS=ON
```

This significantly reduces build time and binary size.

---

# 📌 Summary

You now have:

* Android SDK installed
* NDK configured
* Cross-compilation working
* CMake properly installed
* Environment variables configured
* OpenCV & ONNX Runtime ready for Android builds

---

If you are deploying to a physical device:

```bash
adb push <binary> /data/local/tmp
adb shell
cd /data/local/tmp
chmod +x <binary>
./<binary>
```

