# cse603s21-CUDA-GoL
Game of Life project using CUDA, done for CSE 603 PDP (https://cse.buffalo.edu/~jzola/PDP/)



## Using the Makefile
In order to submit a batch job for a *.cu file:
1. Place the src file in the [sr/nv/](src/nv/) directory
2. Make sure to run `make` or `make init` to ensure you have all directories set up
3. Run `make nv_<file>` where `<file>` is the name of the source file without the file extension.

## Environment Setup
Modules: (use `module load`)
- cuda/11.0
