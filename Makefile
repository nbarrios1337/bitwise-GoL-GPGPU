# ~~~ Initialization ~~~
init:
	mkdir -p bin output

all: init

debug=false

ifeq ($(debug), true)
SCRIPT=debug.sh
else
SCRIPT=slurm.sh
endif

# ~~~ Local vs CCR ~~~
# The CCR variable remains undefined if
# the `sbatch` command is not found
CCR := $(shell command -v sbatch 2> /dev/null)

# ~~~ CUDA Rules ~~~
NVCXX := nvcc
CXXFLAGS := -Wall -Wextra -std=c++11

bin/nv_%: src/nv/%.cu
	$(NVCXX) $^ -o $@ --forward-unknown-to-host-compiler $(CXXFLAGS)

# ~~~ C++ Rules ~~~
# nvcc can compile C++ code, some fun was had
# cannot have the targets defined together w/ the same rule
# see last paragraph of:
# https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html

# CXX := g++

bin/cpp_%: src/cpp/%.cpp
	$(NVCXX) $^ -o $@ --forward-unknown-to-host-compiler $(CXXFLAGS)

# ~~~ Generic Rules ~~~
cpp_%: bin/cpp_%
	./$^

# ~~~ SLURM Rules ~~~
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

# Remove files based on what git ignores
clean:
	cat .gitignore | xargs -I _ find -name _ | xargs rm -vf

cancel:
	scancel --user $$USER

# Necessary to keep the binaries after using the
# bin/nv_% rule as an intermediate rule
.PRECIOUS: bin/nv_% bin/cpp_%
