#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <CL/opencl.h>
 
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =
"__kernel                                                         \n"
"void elementwise( __global const float *in,                      \n"
"                  __global float *out,                           \n"
"                  ulong stride,                                  \n"
"                  ulong vector_length)                           \n"
"{                                                                \n"
"    __private unsigned long idx =                                \n"
"        (get_local_size(0) * get_group_id(0)) + get_local_id(0); \n"
"                                                                 \n"
"    for (; idx < vector_length; idx += stride) {                 \n"
"        out[idx] = in[idx];                                      \n"
"    }                                                            \n"
"}                                                                \n";
 
int main( int argc, char* argv[] )
{
    //srand(time(NULL));

    // Length of vectors (by default)
    int n = 524288;

    // The first argument is the length of the vectors
    if(argc > 1) {
        n = atoi(argv[1]);
        //printf("Changing the size of the vectors to: %d\n", n);
    }
 
    // Host input vectors
    float *h_in;
    // Host output vector
    float *h_out;
 
    // Device input buffers
    cl_mem d_in;
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
    size_t bytes = n*sizeof(float);
 
    // Allocate memory for each vector on host
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
 
    // Initialize vectors on host
    for( int i = 0; i < n; i++ ) {
        h_in[i] = (rand() % 5);
        h_out[i] = 0.0f;
    }
 
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - CUs * WavefrontPoolSize * WorkgroupSize
    globalSize = 120 * 40 * localSize;
 
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
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "elementwise", &err);

    // Create the input and output arrays in device memory for our calculation
    d_in = clCreateBuffer(context, 0, bytes, NULL, NULL);
    d_out = clCreateBuffer(context, 0, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0,
                                   bytes, h_in, 0, NULL, NULL);
 
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, (cl_uint) 0, sizeof(bytes), &d_in);

    err |= clSetKernelArg(kernel, (cl_uint) 1, sizeof(bytes), &d_out);

    err |= clSetKernelArg(kernel, (cl_uint) 2, sizeof(size_t), &globalSize);

    err |= clSetKernelArg(kernel, (cl_uint) 3, sizeof(long), &n);
 
    // Execute the kernel over the entire range of the data set
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, &event);
    
    // Wait for the kernel to finish
    clWaitForEvents(1, &event);

    // Wait for the command queue to get serviced before reading back results (wait for all enqueued tasks to finish)
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                bytes, h_out, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    int errFlag = 0;
    for (int i=0; i<n; i++) {
        if (h_out[i] != h_in[i]) {
            //printf("Error: [%d] %f != %f \n", i, h_out[i], h_in[i]);
            errFlag = 1;
            break;
        }
    }

    if(errFlag)
        printf("Test failed!\n");

    // Print measured kernel time
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end-time_start;
    printf("N = %d | OpenCl Execution time is: %0.9f seconds \n", n, nanoSeconds / 10000000000.0);
 
    // release OpenCL resources
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_in);
    free(h_out);
 
    return 0;
}