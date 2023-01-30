# InterOptCuda
The cuda kernel used in the InterOpt R package for calculating NormFinder and Genorm in parallel for lots of combinations.  
This code is a modified version of the cuda kernel used in [NormiRazor](https://git.btm.umed.pl/ZBiMT/normirazor/-/blob/baa81a8ad7146419758c9cdd1308f4234bd83a1b/python/main.cu)  
The following modifications are applied to the original code:

- Aggregation weights for combinations
- Instead of all combinations, the exact combinations are defined in the input
- In each sample NormFinder only uses elements with CT<35 for calculating sample mean
- Genorm gets a list of stable genes. All genes and combinations are only comapred to those stable genes

## Installation
This kernel is only needed if you want to calculate NormFinder and Genorm stability measures in the [InterOpt](https://github.com/asalimih/InterOpt) package. You should build the InterOptCuda.cu file using `nvcc` and put the executable `InterOptCuda` into your PATH:

```
nvcc -O2 InterOptCuda.cu -o InterOptCuda -std=c++11
```

for more information please refer to:  
- [How to Add a Binary (or Executable, or Program) to Your PATH on macOS, Linux, or Windows](https://zwbetz.com/how-to-add-a-binary-to-your-path-on-macos-linux-windows/)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
