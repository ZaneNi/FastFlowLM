# =============================================================================
# Q4NX Library Makefile
# =============================================================================
#
# Author: Zhenyu Xu (zxu3@clemson.edu)
# Created: 2024
#
# This Makefile is used to build the Q4NX library components.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#   make help   - Show this help message
#
# =============================================================================

# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

BUILD_DIR := ../../build/q4nx/test
LIB_DIR := $(BUILD_DIR)/../lib
BIN_DIR := $(BUILD_DIR)/bin


ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

TARGET = $(LIB_DIR)/libq4_npu_eXpress.so

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall
CXX_FLAGS += -I../../include
CXX_FLAGS += -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP
# Flags for test executable
TEST_FLAGS := $(CXX_FLAGS)
TEST_FLAGS += -L$(LIB_DIR)
TEST_FLAGS += -Wl,-rpath,$(LIB_DIR)

LDFLAGS += -lm
LDFLAGS += -L/opt/xilinx/xrt/lib
LDFLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib
LDFLAGS += -lxrt_coreutil
LDFLAGS += -lboost_program_options -lboost_filesystem

SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard ../../include/*.hpp)
HEADERS += $(wildcard ./*.hpp)

TEST_DEPS := $(test.cpp:.cpp=.d)
-include $(TEST_DEPS)

all: directories $(BIN_DIR)/test_q4nx

directories:
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/test_q4nx: test.cpp
	$(CXX) $(TEST_FLAGS) -o $@ $< -lq4_npu_eXpress $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BIN_DIR)/test_q4nx
	cd $(BIN_DIR) && export LD_LIBRARY_PATH=../../lib && ./test_q4nx --model ../../../../../../models/Llama-3.2-1B-q4nx

.PHONY: all clean test directories

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe


all: directories test


test: directories $(BIN_DIR)/test_q4nx.exe
	cp $(LIB_DIR)/*.dll $(BIN_DIR)
	cd $(BIN_DIR) && ${PWSH} -Command ".\test_q4nx.exe -m ../../../../../../models/Llama-3.2-1B-q4nx"

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/test_q4nx.exe:
	echo "Building test_q4nx.exe"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/q4nx"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_q4nx_target"

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean test directories

endif