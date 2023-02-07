// enable extension for OpenCL 1.1 and lower
#if __OPENCL_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Initialize Float arrays
__kernel void initialiseDoubleArraysKernel(__global double * restrict A,
                                           __global double * restrict B,
                                           __global double * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = 1.0;
	B[tid] = 2.0;
	C[tid] = 0.0;
}

// Initialize Float arrays
__kernel void initialiseFloatArraysKernel(__global float * restrict A,
                                          __global float * restrict B,
                                          __global float * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = 1.0;
	B[tid] = 2.0;
	C[tid] = 0.0;
}

// Elementwise with Double Precision
__kernel void elementwiseDoubleStride(__global const double *A,
                                      __global const double *B,
                                      __global double *C,
                                      ulong stride,
                                      ulong vector_length)
{
  __private unsigned long tid = (get_local_size(0) * get_group_id(0)) + get_local_id(0);

  for (; tid < vector_length; tid += stride) {
    C[tid] = A[tid] * B[tid];
  }
}

// Elementwise with Single Precision
__kernel void elementwiseFloatStride(__global const float *A,
                                     __global const float *B,
                                     __global float *C,
                                     ulong stride,
                                     ulong vector_length)
{
  __private unsigned long tid = (get_local_size(0) * get_group_id(0)) + get_local_id(0);

  for (; tid < vector_length; tid += stride) {
    C[tid] = A[tid] * B[tid];
  }
}

// Elementwise with double precision without Stride
__kernel void elementwiseDouble(__global const double *A, 
                                __global const double *B,
                                __global double *C)
{
  __private size_t tid = get_global_id(0);

  C[tid] = A[tid] * B[tid];
}

// Elementwise with Single Precision without Stride
__kernel void elementwiseFloat(__global const float *A,
                               __global const float *B,
                               __global float *C)
{
  __private size_t tid = get_global_id(0);

  C[tid] = A[tid] * B[tid];
}

// Elementwise copy with Double Precision
__kernel void elementwiseCopyDoubleStride(__global const double *A,
                                          __global double *C,
                                          ulong stride,
                                          ulong vector_length)
{
  __private unsigned long tid = (get_local_size(0) * get_group_id(0)) + get_local_id(0);

  for (; tid < vector_length; tid += stride) {
    C[tid] = A[tid];
  }
}

// Elementwise copy with Single Precision
__kernel void elementwiseCopyFloatStride(__global const float *A,
                                         __global float *C,
                                         ulong stride,
                                         ulong vector_length)
{
  __private unsigned long tid = (get_local_size(0) * get_group_id(0)) + get_local_id(0);

  for (; tid < vector_length; tid += stride) {
    C[tid] = A[tid];
  }
}

// Elementwise copy with Double Precision without Stride
__kernel void elementwiseCopyDouble(__global const double *A,
                                    __global double *C)
{
  __private size_t tid = get_global_id(0);

  C[tid] = A[tid];
}

// Elementwise copy with Single Precision without Stride
__kernel void elementwiseCopyFloat(__global const float *A,
                                   __global float *C)
{
  __private size_t tid = get_global_id(0);

  C[tid] = A[tid];
}