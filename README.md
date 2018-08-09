# C++ Parallel mandelbrot implementation

This repository contains 2 implementation of mandelbrot's fractals optimize with AVX2 instructions
and TBB parallel for for multiple threads.

# Build

In order to build the project you will to do:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release
make
```

This will generate 2 binaries:
- view
- bench

# View
View take a size in parameter and render the corresponding image. As third argument you can put a
name
```
./view 720 [filename]
```

# Bench
The bench executable is a benchmark which uses google benchmark lib. It give an
overview of the performance of the implementation.
