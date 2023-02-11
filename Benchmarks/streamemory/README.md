# StreaMemory (Stream+Memory) Benchmark

There are 6 main kernels with different implementations:
1. CopyKernel
2. ScaleKernel
3. AddKernel
4. TriadKernel
5. Elementwise
6. ElementwiseCopy

The first **four** are part of the Stream Benchmark, the last **two** from the Memory Accesses Benchmark.

---

Regarding the implementation, the first four (**1, 2, 3 and 4**) are implemented using two different data types (Floats and Doubles):
- copyKernelD -> Copy kernel with Doubles
- copyKernelF -> Copy kernel with Floats

- scaleKernelD -> Scale kernel with Doubles
- scaleKernelF -> Scale kernel with Floats

- addKernelD -> Add kernel with Doubles
- addKernelF -> Add kernel with Floats

- triadKernelD -> Triad kernel with Doubles
- triadKernelF -> Triad kernel with Floats

---

The last ones (**5 & 6**) are implemented using two different data types (Floats and Doubles), and also, using or not using a strided memory accessing:
- ElementwiseDS -> Elementwise with Doubles + Stride.
- ElementwiseFS -> Elementwise with Floats + Stride.
- ElementwiseD -> Elementwise with Doubles.
- ElementwiseF -> Elementwise with Floats.

- ElementwiseCopyDS -> Elementwise Copy with Doubles + Stride.
- ElementwiseCopyFS -> Elementwise Copy with Floats + Stride.
- ElementwiseCopyD -> Elementwise Copy with Doubles.
- ElementwiseCopyF -> Elementwise Copy with Floats.

---

Along with these implementations there are two kernels for data initialization:
- initialiseDoubleArraysKernel -> Initializes a vector of doubles.
- initialiseFloatArraysKernel -> Initializes a vector of floats.

The data will always be on the device, and never copied to the host (we want to measure the bandwidth, we dont care about the result!).

---

If you want to implement a new one:
* simply add it to the kernels.cl ,
* then, link it to main() using the OpenCL API, clCreateKernel(),
* attach the kernel variables clSetKernelArg(),
* and finally, add it to the main execution block.

It should be pretty straight forward, there are special fields, i.e. if you want to use an stride, seventh parameter RunTest().