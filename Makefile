# ~~~ C++ Rules ~~~

CXX := g++
CPPFLAGS := -Wall -Wextra -std=c++11

cpp_%: src/cpp/%.cpp
	$(CXX) $^ $(CPPFLAGS) -o $@

# ~~~ CUDA Rules ~~~
NVCXX := nvcc
CUDAFLAGS := -Werror all-warnings -std=c++11

nv_%: src/nv/%.cu
	$(NVCXX) $^ $(CUDAFLAGS) -o $@

# ~~~ Generic Rules ~~~

# Remove files based on what git ignores
clean:
	cat .gitignore | xargs -I _ find -name _ | xargs rm -v