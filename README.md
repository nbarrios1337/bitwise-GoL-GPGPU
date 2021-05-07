# Accelerating Game of Life via Bitwise Operations on NVIDIA GPGPUs
A Game of Life on CUDA implementation, done for [CSE 603 - Parallel and Distributed Processing](https://cse.buffalo.edu/~jzola/PDP/) at the University at Buffalo. We aim to use integer bitwise operations to speed up traditional CUDA implementations of the Game of Life.

## Background
This project arose out of an interest in high performance computing and the CUDA programming model. With some help from [Prof. Zola](https://cse.buffalo.edu/~jzola/), we cemented the plan to do a bitwise implementation based on research done by N. Tsuda, and Fujita et. al (See [References](#references)).
## Getting Started
As with any CUDA project, you'll need an installation of [NVIDIA's CUDA toolkit](https://developer.nvidia.com/cuda-zone). The project itself was developed on Linux, but we welcome pull requests to have the code work on other platforms

Additionally, there is some setup to use the [University at Buffalo Center for Computational Research](http://www.buffalo.edu/ccr.html)'s resources, but there's no installation necessary for such functionality, and the project will still work without access to CCR.

## Using the Makefile
There is a one=time rule to set up the necessary directories:
```bash
make init
```

There are rules defined for C++-specific and CUDA-specific compilation, however they all use the same recipe as the CUDA compiler driver (`nvcc`) can handle plain C++ as well. i.e. `make cpp_%` handles files found under [src/cpp/](src/cpp), and `make nv_%` handles files found under [src/nv/](src/cpp). This general paradigm breaks down with specific exectuable requirements, as some of the C++ code calls wrappers around CUDA kernels.

Running any `make` rule with `DEBUG=true` triggers debug compilation flags like `-g`, `printf`s that only compile when `DEBUG` is defined, as well as switching to the debug cluster on CCR. For example,
```bash
make nv_BOL DEBUG=true
```
switches the BOL program to simply print out what each CUDA thread block would access, without doing any meaningful work.

When it comes to benchmarking and profiling, those rules extend the `nv_` and `cpp_` naming scheme. For example, to profile the program that was compiled and run by the `make nv_BOL` rule, we'd do `make bench_nv_BOL` and `make profile_nv_BOL`.

## Contributing
We're more than happy to accept issues and pull requests! Response time might be delayed as another coursework take priority later on.

## References
1. Fujita, Toru, et al. “Efficient GPU Implementations for the Conways Game of Life.” 2015 Third International Symposium on Computing and Networking (CANDAR), Dec. 2015, [doi:10.1109/candar.2015.11](https://ieeexplore.ieee.org/abstract/document/7424264).
2. Simpson, Adam. “CUDA Game of Life.” Oak Ridge Leadership Computing Facility, Oak Ridge National Lab, 13 Dec. 2017, [www.olcf.ornl.gov/tutorials/cuda-game-of-life/](www.olcf.ornl.gov/tutorials/cuda-game-of-life/).
3. Tsuda, Nobuhide. "Accelerate life games by bit operations (bit board) (in Japanese)", 23 Dec. 2012, [vivi.dyndns.org/tech/games/LifeGame.html](vivi.dyndns.org/tech/games/LifeGame.html).

## License
Copyright &copy; 2021 Nicolas Barrios, Muhanned Ibrahim
The code in this project is licensed under MIT license, unless otherwise stated.