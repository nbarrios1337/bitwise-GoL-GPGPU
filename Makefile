# ~~~ Initialization ~~~
init:
	mkdir -p bin output

all: init

DEBUG=false

ifeq ($(DEBUG), true)
SCRIPT=debug.sh
else
SCRIPT=slurm.sh
endif

NVCXX := nvcc
CXXFLAGS := -Wall -Wextra -std=c++11
NVFLAGS := -Werror=all-warnings -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70

CLANG_EXISTS := $(shell command -v clang++ 2> /dev/null)

# See https://llvm.org/docs/CompileCudaWithLLVM.html
ifdef CLANG_EXISTS
CXX := clang++
BONUSFLAGS := --cuda-gpu-arch=sm_61 --cuda-gpu-arch=sm_70 -L/usr/lib/cuda -lcudart_static -ldl -lrt -pthread --cuda-path=/usr/lib/cuda
endif

ifeq ($(DEBUG), true)
CXXFLAGS += -DDEBUG
NVFLAGS += -g -G
endif

INCLUDE_DIR=includes/

SPACE := $(subst ,, )
COMMA := ,

# ~~~ Local vs CCR ~~~
# The CCR variable remains undefined if
# the `sbatch` command is not found
CCR := $(shell command -v sbatch 2> /dev/null)

# ~~~ Specific Requirements ~~~
bin/test_bitwise: src/cpp/bitwise.cpp

bin/test_compute: src/nv/compute.cu

# ~~~ Clangd Specific Rules ~~~
ifdef CLANG_EXISTS
BOL: src/nv/BOL.cu
	$(CXX) $^ -o bin/nv_BOL -I$(INCLUDE_DIR) $(CXXFLAGS) $(BONUSFLAGS)

compile_commands.json:
	bear $(MAKE) BOL

clangd: compile_commands.json
endif

# ~~ Compilation Rules ~~~
bin/nv_%: src/nv/%.cu
	$(NVCXX) $^ -o $@ -I$(INCLUDE_DIR) $(NVFLAGS) -Xcompiler $(subst $(SPACE),$(COMMA),$(CXXFLAGS))

# nvcc can compile C++ code, some fun was had
# cannot have the targets defined together w/ the same rule
# see last paragraph of:
# https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html

bin/cpp_%: src/cpp/%.cpp
	$(NVCXX) $^ -o $@ -I$(INCLUDE_DIR) $(NVFLAGS) -Xcompiler $(subst $(SPACE),$(COMMA),$(CXXFLAGS))

# ~~~ Tests ~~~
bin/test_%: tests/%.cpp
	$(NVCXX) $^ -o $@ -I$(INCLUDE_DIR) $(NVFLAGS) -Xcompiler $(subst $(SPACE),$(COMMA),$(CXXFLAGS))

# ~~~ Execution Rules ~~~
cpp_%: bin/cpp_%
	./$^

test_%: bin/test_%
	./$^

ifndef CCR
nv_%: bin/nv_%
	./$^
else
nv_%: bin/nv_%
	sbatch	--job-name=`basename $^`			\
			--output=output/`basename $^`-%j.out	\
			--error=output/`basename $^`-%j.err	\
			$(SCRIPT) $^
endif

clean:
	rm -vf bin/* output/*

cancel:
	scancel --user $$USER

# Necessary to keep the binaries after using the
# bin/nv_% rule as an intermediate rule
.PRECIOUS: bin/nv_% bin/cpp_%
