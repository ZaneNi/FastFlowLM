# =============================================================================
# LM Head Test Makefile
# =============================================================================
#
# Author: FastFlowLM Team
# Created: 2025
#
# This Makefile is used to build the LM Head test component.
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

BUILD_DIR := ../../build/lm_head/test
LIB_DIR := $(BUILD_DIR)/../lib
BIN_DIR := $(BUILD_DIR)/bin
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

TARGET = $(LIB_DIR)/liblm_head.so

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -DUSEAVX2=1
CXX_FLAGS += -mavx2 -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
CXX_FLAGS += -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP


# Flags for test executable
TEST_FLAGS := $(CXX_FLAGS)
TEST_FLAGS += -L$(LIB_DIR)
TEST_FLAGS += -Wl,-rpath,$(LIB_DIR)

LDFLAGS += -L/opt/xilinx/xrt/lib
LDFLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib
LDFLAGS += -lxrt_coreutil
LDFLAGS += -lboost_program_options -lboost_filesystem
LDFLAGS += -L$(LIB_DIR) -Wl,-rpath,$(LIB_DIR)
LDFLAGS += -L../../build/lib
LDFLAGS += -L../../build/tokenizer/lib
LDFLAGS += -ltokenizers_cpp -ltokenizers_c -l sentencepiece
LDFLAGS += -lqwen_npu -lllama_npu -ldequant -lgemm -llm_head -lq4_npu_eXpress -lnpu_utils



TEST_SOURCES += test.cpp
TEST_SOURCES += ../../common/chat/chat_bot.cpp
TEST_SOURCES += ../../common/npu_utils/npu_utils.cpp
TEST_SOURCES += ../../common/npu_utils/npu_instr_utils.cpp
TEST_SOURCES += ../../common/tokenizer/tokenizer.cpp
TEST_SOURCES += ../../common/modules/sampler.cpp


HEADERS += ../../include/modules/lm_head.hpp
HEADERS += ../../include/modules/embedding.hpp
HEADERS += ../../include/modules/gemm.hpp
HEADERS += ../../include/modules/dequant.hpp
HEADERS += ../../include/modules/sampler.hpp
HEADERS += ../../include/lm_config.hpp

TEST_DEPS := $(test.cpp:.cpp=.d)

all: directories $(BIN_DIR)/test_lm_head

directories:
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/test_lm_head: $(TEST_SOURCES) $(TEST_DEPS)
	$(CXX) $(TEST_FLAGS) -o $@ $^ -llm_head -lqwen_npu $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BIN_DIR)/test_lm_head
	make -C ../../detail/lm_head
	cp $(LIB_DIR)/liblm_head.so $(BIN_DIR)
	cd $(BIN_DIR) && export LD_LIBRARY_PATH=../../../../build/lib && ./test_lm_head --model qwen3.2 --model_type QWEN3_4B

-include $(TEST_DEPS)
.PHONY: all clean test directories

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe


TARGET = $(BUILD_DIR)/lm_head.dll

all: directories test

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(BIN_DIR)/test_lm_head.exe:
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/lm_head"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_lm_head_target"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BIN_DIR)/test_lm_head.exe
	cp ../../build/lib/*.dll $(BIN_DIR)
	cd $(BIN_DIR) && ${PWSH} -Command ".\test_lm_head.exe -m qwen3 --model_type QWEN3_4B"

.PHONY: all clean test directories

endif 