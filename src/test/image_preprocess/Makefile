# =============================================================================
# Qwen NPU Library Makefile
# =============================================================================
#
# Author: Zhenyu Xu (zxu3@clemson.edu)
# Created: 2024
#
# This Makefile is used to build the Qwen NPU library components.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#   make help   - Show this help message
#
# =============================================================================


BUILD_DIR := ../../build/image_preprocess
LIB_DIR := $(BUILD_DIR)/../lib
BIN_DIR := $(BUILD_DIR)/bin

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(BIN_DIR)/test_image_preprocess.exe:
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../test/image_preprocess"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target image_processor"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BIN_DIR)/test_image_preprocess.exe
		cp ../../build/lib/*.dll $(BIN_DIR)
		cp ../../external_libs/*.dll $(BIN_DIR)
	cd $(BIN_DIR) && ${PWSH} -Command ".\image_processor.exe ..\..\..\iceCream.jpg"

.PHONY: all clean test directories

