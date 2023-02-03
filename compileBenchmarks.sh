#!/bin/bash

# Add this to a folder in your home (~), and then, to your .bashrc, like:
# export PATH=$PATH:$HOME/name_of_your_folder

# Careful with the location of OpenCL, it may change with the version of ROCm
# You can find the location of OpenCL with (be at your root /):
# find . -name "libOpenCL.so" > ~/opencl.txt | grep -v "Permission denied"

cc -O2 -Wall -o $1.out $1.c -L/opt/rocm-5.2.3/opencl/lib -lOpenCL