# ~~~ Initialization ~~~
init:
	mkdir -p bin output

all: init

# ~~~ CUDA Rules ~~~
NVCXX := nvcc
CXXFLAGS := -Wall -Wextra -std=c++11

bin/nv_%: src/nv/%.cu
	$(NVCXX) $^ -o $@ --forward-unknown-to-host-compiler $(CXXFLAGS)

# ~~~ C++ Rules ~~~
# nvcc can compile C++ code, some fun was had
# cannot have the targets defined together w/ the same rule
# see last paragraph of 
# https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html

# CXX := g++

bin/cpp_%: src/cpp/%.cpp
	$(NVCXX) $^ -o $@ --forward-unknown-to-host-compiler $(CXXFLAGS)

# ~~~ Generic Rules ~~~
cpp_%: bin/cpp_% ;

# ~~~ SLURM Rules ~~~
nv_%: bin/nv_%
	sbatch	--job-name=`basename $^`			\
			--output=output/`basename $^`-%j.out	\
			--error=output/`basename $^`-%j.err	\
			slurm.sh $^

# Remove files based on what git ignores
clean:
	cat .gitignore | xargs -I _ find -name _ | xargs rm -vf

# Necessary to keep the binaries after using the
# bin/nv_% rule as an intermediate rule
.PRECIOUS: bin/nv_% bin/cpp_%