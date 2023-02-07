#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <CL/opencl.h>

// OpenCL kernel. Each work item takes care of one element of c
const char *kernelFileName = "kernel.cl";

// Number of times to run tests
#define NTIMES 50

// Array size for tests. Needs to be big to sufficiently load device.
// Must be divisible by WGSIZE (and less than 256, the largest local workgroup size tested)
#define TRYARRAYSIZE 16777216

// Amount of workitems in a workgroup
// Recommended to be a multiple of 64
#define WGSIZE 64

// AMD MI100 Specs
#define CU 120
#define WFP 40

// Print per-local-size results during test?
//#define VERBOSE

// Function prototypes
double GetWallTime(void);

int main( int argc, char* argv[] )
{
    // Variable to store a defined size of the array
    size_t arraySize = TRYARRAYSIZE;

    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_out;
 
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_out;
    
    // OpenCL Parameters
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 
    // Size, in bytes, of each vector
    size_t bytes = TRYARRAYSIZE * sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_out = (double*)malloc(bytes);
 
    // Initialize vectors on host
    for( int i = 0; i < TRYARRAYSIZE; i++ ) {
        h_a[i] = ((double) rand() / RAND_MAX) * (5);
        h_b[i] = ((double) rand() / RAND_MAX) * (5);
        h_out[i] = 0.0;
    }
 
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = WGSIZE;
 
    // Number of total work items - CUs * WavefrontPoolSize * WorkgroupSize
    globalSize = CU * WFP * localSize;

    // Get Kernel from file
    FILE* kernelFile = fopen(kernelFileName, "rb");
	fseek(kernelFile, 0, SEEK_END);
	long fileLength = ftell(kernelFile);
	rewind(kernelFile);
	char *kernelSource = malloc(fileLength*sizeof(char));
	long read = fread(kernelSource, sizeof(char), fileLength, kernelFile);
	if (fileLength != read) printf("Error reading kernel file, line %d\n", __LINE__);
	fclose(kernelFile);
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    
    // Creates a properties list for the command queue
    cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

    // Create a command queue 
    queue = clCreateCommandQueueWithProperties(context, device_id, properties, &err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
 
    // Build the program executable 
    err = clBuildProgram(program, 0, NULL, "-I.", NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "elementwise", &err);

    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, 0, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, 0, bytes, NULL, NULL);
    d_out = clCreateBuffer(context, 0, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);

    err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
 
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, (cl_uint) 0, sizeof(bytes), &d_a);

    err  = clSetKernelArg(kernel, (cl_uint) 1, sizeof(bytes), &d_b);

    err |= clSetKernelArg(kernel, (cl_uint) 2, sizeof(bytes), &d_out);

    err |= clSetKernelArg(kernel, (cl_uint) 3, sizeof(size_t), &globalSize);

    err |= clSetKernelArg(kernel, (cl_uint) 4, sizeof(long), &arraySize);

    double time = GetWallTime();

    // Execute the kernel over the entire range of the data set
    for (int n = 0; n < NTIMES; n++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    }

    // Wait for the command queue to get serviced before reading back results (wait for all enqueued tasks to finish)
    clFinish(queue);

    time = GetWallTime() - time;
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                bytes, h_out, 0, NULL, NULL );
    
#ifdef VERBOSE
        int errFlag = 0;
        for (int i=0; i<TRYARRAYSIZE; i++) {
            if (h_out[i] != (h_a[i]*h_b[i])) {
                //printf("Error: [%d] %f != %f \n", i, h_out[i], h_in[i]);
                errFlag = 1;
                break;
            }
        }

        if(errFlag)
            printf("Test failed!\n");
        else
            printf("Test passed!\n");
#endif

    // Print measured kernel time
    printf("GB/s: %14.3lf\n", 2*NTIMES*TRYARRAYSIZE*sizeof(double)/1024.0/1024.0/1024.0/time);

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_out);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_a);
    free(h_b);
    free(h_out);
    free(kernelSource);
 
    return 0;
}

// Return ns accurate walltime
double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}