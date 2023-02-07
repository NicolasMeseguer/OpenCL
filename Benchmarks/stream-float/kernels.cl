// enable extension for OpenCL 1.1 and lower
#if __OPENCL_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Initialize arrays
__kernel void initialiseArraysKernel(__global float * restrict A,
                                     __global float * restrict B,
                                     __global float * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = 1.0;
	B[tid] = 2.0;
	C[tid] = 0.0;

}

// Copy kernels
__kernel void copyKernel1(__global const float * restrict A,
                          __global float * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel2(__global const float2 * restrict A,
                          __global float2 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel4(__global const float4 * restrict A,
                          __global float4 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel8(__global const float8 * restrict A,
                          __global float8 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel16(__global const float16 * restrict A,
                          __global float16 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}



// Scale kernels
__kernel void scaleKernel1(const float scalar,
                           __global float * restrict B,
                           __global const float * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel2(const float scalar,
                           __global float2 * restrict B,
                           __global const float2 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel4(const float scalar,
                           __global float4 * restrict B,
                           __global const float4 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel8(const float scalar,
                           __global float8 * restrict B,
                           __global const float8 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel16(const float scalar,
                           __global float16 * restrict B,
                           __global const float16 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}



// Add kernels
__kernel void addKernel1(__global const float * restrict A,
                         __global const float * restrict B,
                         __global float * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel2(__global const float2 * restrict A,
                         __global const float2 * restrict B,
                         __global float2 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel4(__global const float4 * restrict A,
                         __global const float4 * restrict B,
                         __global float4 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel8(__global const float8 * restrict A,
                         __global const float8 * restrict B,
                         __global float8 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel16(__global const float16 * restrict A,
                         __global const float16 * restrict B,
                         __global float16 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}



// Triad kernels
__kernel void triadKernel1(const float scalar,
                           __global float * restrict A,
                           __global const float * restrict B,
                           __global const float * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]+scalar * C[tid];
}
__kernel void triadKernel2(const float scalar,
                           __global float2 * restrict A,
                           __global const float2 * restrict B,
                           __global const float2 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]+scalar * C[tid];
}
__kernel void triadKernel4(const float scalar,
                           __global float4 * restrict A,
                           __global const float4 * restrict B,
                           __global const float4 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]+scalar * C[tid];
}
__kernel void triadKernel8(const float scalar,
                           __global float8 * restrict A,
                           __global const float8 * restrict B,
                           __global const float8 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]+scalar * C[tid];
}
__kernel void triadKernel16(const float scalar,
                           __global float16 * restrict A,
                           __global const float16 * restrict B,
                           __global const float16 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]+scalar * C[tid];
}

// nstream kernels

// stream_dot kernels