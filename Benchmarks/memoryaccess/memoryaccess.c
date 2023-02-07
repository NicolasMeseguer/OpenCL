#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>       // DBL_MIN

/* clCreateCommandQueue with 2.0 headers gives a warning about it being deprecated, avoid it */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/opencl.h>

// OpenCL kernel. Each work item takes care of one element of c
const char *kernelFileName = "kernels.cl";

// Number of times to run tests
#define NTIMES 50

// MAX array size for tests. Needs to be big to sufficiently load device.
// Must be divisible by WGSIZE
#define TRYARRAYSIZE (1 * 1024*1024*1024/8)

// AMD MI100 Specs, used for the stride benchmarks
#define CU 120
#define WFP 40

// For fast executions you can auto-select the device and platform and skip the scanf
#define AUTOPLATFORM 0
#define AUTODEVICE 0

// Print per-local-size results during test?
//#define VERBOSE

// Function prototypes
double GetWallTime(void);
void RunTest(cl_command_queue * queue, cl_kernel * kernel, size_t vecWidth, char * testName, int memops, int flops, size_t arraySize, int strideBool);
void SanitizeAndRoundArraySize(size_t * sizeBytes, cl_ulong maxAlloc, cl_ulong globalMemSize, size_t typeSize, size_t * arraySize, char * arrayName);

// OpenCL Stuff
int InitialiseCLEnvironment(cl_platform_id**, cl_device_id***, cl_context*, cl_command_queue*, cl_program*, cl_ulong*, cl_ulong*);
void CleanUpCLEnvironment(cl_platform_id**, cl_device_id***, cl_context*, cl_command_queue*, cl_program*);
void CheckOpenCLError(cl_int err, int line);

int main( int argc, char* argv[] )
{
	// Disable caching of binaries by nvidia implementation
	setenv("CUDA_CACHE_DISABLE", "1", 1);

	// Set up OpenCL environment
	cl_platform_id    *platform;
	cl_device_id      **device_id;
	cl_context        context;
	cl_command_queue  queue;
	cl_ulong          maxAlloc, globalMemSize;
	cl_program        program;
	cl_kernel         initDoubleArrays, initFloatArrays;
	cl_kernel         elementwiseDS,elementwiseFS;
	cl_kernel         elementwiseD,elementwiseF;
	cl_kernel         elementwiseCopyDS,elementwiseCopyFS;
	cl_kernel         elementwiseCopyD,elementwiseCopyF;
	cl_int            err;
	cl_mem            device_dA, device_dB, device_dC;
	cl_mem            device_fA, device_fB, device_fC;

	if (InitialiseCLEnvironment(&platform, &device_id, &context, &queue, &program, &maxAlloc, &globalMemSize) == EXIT_FAILURE) {
		printf("Error initialising OpenCL environment\n");
		return EXIT_FAILURE;
	}

    // Create kernels. 
	initDoubleArrays = clCreateKernel(program, "initialiseDoubleArraysKernel", &err);
	initFloatArrays = clCreateKernel(program, "initialiseFloatArraysKernel", &err);
	CheckOpenCLError(err, __LINE__);

    // There are 2 main kernels; elementwise and elementwiseCopy.
    // ├> elementwise is used to perform computation over a vector
    //      ├> 2 kernels are stride-dependant & data-dependant, elementwiseDoubleStride and elementwiseFloatStride
    //      └> 2 kernels data-dependant, elementwiseDouble and elementwiseFloat
    //
    // └> elementwiseCopy is used to copy data from one vector to another vector
    //      ├> 2 kernels data-dependant, elementwiseCopyDouble and elementwiseCopyFloat
    //      └> 2 kernels are stride-dependant & data-dependant, elementwiseCopyDoubleStride and elementwiseCopyFloatStride

    elementwiseDS = clCreateKernel(program, "elementwiseDoubleStride", &err);
	elementwiseFS = clCreateKernel(program, "elementwiseFloatStride", &err);
	CheckOpenCLError(err, __LINE__);
	elementwiseD = clCreateKernel(program, "elementwiseDouble", &err);
    elementwiseF = clCreateKernel(program, "elementwiseFloat", &err);
	CheckOpenCLError(err, __LINE__);
	elementwiseCopyDS = clCreateKernel(program, "elementwiseCopyDoubleStride", &err);
    elementwiseCopyFS = clCreateKernel(program, "elementwiseCopyFloatStride", &err);
	CheckOpenCLError(err, __LINE__);
	elementwiseCopyD = clCreateKernel(program, "elementwiseCopyDouble", &err);
    elementwiseCopyF = clCreateKernel(program, "elementwiseCopyFloat", &err);
	CheckOpenCLError(err, __LINE__);

    // If the user inputs a size, it will be used. Otherwise, the default size is used.
    size_t arraySize = TRYARRAYSIZE;
    if(argc == 2 && atoi(argv[1]) > 256 && atoi(argv[1]) % 16 == 0) {
        arraySize = (size_t) atoi(argv[1]);
#ifdef VERBOSE
        printf("Using array size of %zu\n", arraySize);
#endif
    }

    // Sanitize array size of doubles (in case it's bigger than the GPU memory)
    size_t sizeBytesDouble = arraySize * sizeof(double);
    SanitizeAndRoundArraySize(&sizeBytesDouble, maxAlloc, globalMemSize, sizeof(double), &arraySize, "doubles");

    size_t sizeBytesFloat = arraySize * sizeof(float);
    SanitizeAndRoundArraySize(&sizeBytesFloat, maxAlloc, globalMemSize, sizeof(float), &arraySize, "floats");

    // Assign double variables to the device
    device_dA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytesDouble, NULL, &err);
	device_dB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytesDouble, NULL, &err);
	device_dC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytesDouble, NULL, &err);

    // Assign float variables to the device
    device_fA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytesFloat, NULL, &err);
	device_fB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytesFloat, NULL, &err);
	device_fC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytesFloat, NULL, &err);

    // Assign variables to different kernels
    // initialiseDoubleArraysKernel
    err  = clSetKernelArg(initDoubleArrays, 0, sizeof(cl_mem), &device_dA);
	err |= clSetKernelArg(initDoubleArrays, 1, sizeof(cl_mem), &device_dB);
	err |= clSetKernelArg(initDoubleArrays, 2, sizeof(cl_mem), &device_dC);

    // initialiseFloatArraysKernel
    err |= clSetKernelArg(initFloatArrays, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(initFloatArrays, 1, sizeof(cl_mem), &device_fB);
	err |= clSetKernelArg(initFloatArrays, 2, sizeof(cl_mem), &device_fC);

    // elementwiseDoubleStride
	err |= clSetKernelArg(elementwiseDS, 0, sizeof(cl_mem), &device_dA);
	err |= clSetKernelArg(elementwiseDS, 1, sizeof(cl_mem), &device_dB);
	err |= clSetKernelArg(elementwiseDS, 2, sizeof(cl_mem), &device_dC);
	err |= clSetKernelArg(elementwiseDS, 4, sizeof(unsigned long), &arraySize);

    // elementwiseFloatStride
	err |= clSetKernelArg(elementwiseFS, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(elementwiseFS, 1, sizeof(cl_mem), &device_fB);
	err |= clSetKernelArg(elementwiseFS, 2, sizeof(cl_mem), &device_fC);
	err |= clSetKernelArg(elementwiseFS, 4, sizeof(unsigned long), &arraySize);

    // elementwiseDouble
	err |= clSetKernelArg(elementwiseD, 0, sizeof(cl_mem), &device_dA);
	err |= clSetKernelArg(elementwiseD, 1, sizeof(cl_mem), &device_dB);
	err |= clSetKernelArg(elementwiseD, 2, sizeof(cl_mem), &device_dC);

    // elementwiseFloat
	err |= clSetKernelArg(elementwiseF, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(elementwiseF, 1, sizeof(cl_mem), &device_fB);
	err |= clSetKernelArg(elementwiseF, 2, sizeof(cl_mem), &device_fC);

    // elementwiseCopyDoubleStride
	err |= clSetKernelArg(elementwiseCopyDS, 0, sizeof(cl_mem), &device_dA);
	err |= clSetKernelArg(elementwiseCopyDS, 1, sizeof(cl_mem), &device_dC);
	err |= clSetKernelArg(elementwiseCopyDS, 3, sizeof(unsigned long), &arraySize);

    // elementwiseCopyFloatStride
	err |= clSetKernelArg(elementwiseCopyFS, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(elementwiseCopyFS, 1, sizeof(cl_mem), &device_fC);
	err |= clSetKernelArg(elementwiseCopyFS, 3, sizeof(unsigned long), &arraySize);

    // elementwiseCopyDouble
	err |= clSetKernelArg(elementwiseCopyD, 0, sizeof(cl_mem), &device_dA);
	err |= clSetKernelArg(elementwiseCopyD, 1, sizeof(cl_mem), &device_dC);

    // elementwiseCopyFloat
	err |= clSetKernelArg(elementwiseCopyF, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(elementwiseCopyF, 1, sizeof(cl_mem), &device_fC);

	CheckOpenCLError(err, __LINE__);

	// Initialize arrays
	size_t initLocalSize = 64;
	size_t initGlobalSize = arraySize;
	err = clEnqueueNDRangeKernel(queue, initDoubleArrays, 1, NULL, &initGlobalSize, &initLocalSize, 0, NULL, NULL);
	err = clEnqueueNDRangeKernel(queue, initFloatArrays, 1, NULL, &initGlobalSize, &initLocalSize, 0, NULL, NULL);
	clFinish(queue);

    // Fourth argument is the number of memory operations per output array item. Used in bandwidth calculation.
	// Fifth argument is the number of flops per output array item. Used in flops calculation.
    // Seventh argument indicates if the kernel is a strided kernel, and copy the wgsize to the kernel each iteration.
	printf("--------------------------------------------------------------------------------------------------------\n");
	printf("Function             Best Rate GB/s   Avg time   Min time   Max time   Best Workgroup Size   Best GFLOPS\n");
	printf("--------------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &elementwiseDS,  1,  "elementwiseDS",  3, 1, arraySize, 3);
	RunTest(&queue, &elementwiseFS,  1,  "elementwiseFS",  3, 1, arraySize, 3);
	printf("--------------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &elementwiseD,  1,  "elementwiseD",  3, 1, arraySize, -1);
	RunTest(&queue, &elementwiseF,  1,  "elementwiseF",  3, 1, arraySize, -1);
	printf("--------------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &elementwiseCopyDS,  1,  "elementwiseCopyDS",  2, 0, arraySize, 2);
	RunTest(&queue, &elementwiseCopyFS,  1,  "elementwiseCopyFS",  2, 0, arraySize, 2);
	printf("--------------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &elementwiseCopyD,  1,  "elementwiseCopyD",  2, 0, arraySize, -1);
	RunTest(&queue, &elementwiseCopyF,  1,  "elementwiseCopyF",  2, 0, arraySize, -1);
	printf("--------------------------------------------------------------------------------------------------------\n");

    CleanUpCLEnvironment(&platform, &device_id, &context, &queue, &program);
	return 0;
}

void RunTest(cl_command_queue * queue, cl_kernel * kernel, size_t vecWidth, char * testName, int memops, int flops, size_t arraySize, int KernelStrideIdx)
{
	size_t localSize;
	size_t bestLocalSize;
	size_t globalSize = arraySize;
	double bestTime = DBL_MAX, worstTime = DBL_MIN, totalTime = 0.0;
	int err;

	// Test local sizes from 2 to to 256, in powers of 2
	for (localSize = 1; localSize <= 256; localSize *= 2) {

        if(KernelStrideIdx != -1) {
            // Since we are using a strided kernel, we need to pass the globalSize to the kernel each iteration.
            globalSize = CU * WFP * localSize;
            err = clSetKernelArg(*kernel, KernelStrideIdx, sizeof(unsigned long), &globalSize);
        }

		if (globalSize % localSize != 0) {
			printf("Error, localSize must divide globalSize! (%zu %% %zu = %zu)\n",
			       globalSize, localSize, globalSize%localSize);
		}

		double time = GetWallTime();

		for (int n = 0; n < NTIMES; n++) {
			err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		}
		clFinish(*queue);
		CheckOpenCLError(err, __LINE__);

		time = GetWallTime() - time;
		if (time < bestTime) {
			bestTime = time;
			bestLocalSize = localSize;
		}
		if (time > worstTime) {
			worstTime = time;
		}
		totalTime += time;

#ifdef VERBOSE
		printf("------------- localSize = %3zu, bandwidth = %7.3lf GB/s\n",
		        localSize, memops*NTIMES*arraySize*sizeof(float)/1024.0/1024.0/1024.0/(time));
#endif

	}

	printf("%18s   %14.3lf   %8.6lf   %8.6lf   %8.6lf   %19zu   %11.3lf\n",
	       testName, memops*NTIMES*arraySize*sizeof(float)/1024.0/1024.0/1024.0/bestTime, totalTime/NTIMES,
	       bestTime, worstTime, bestLocalSize, flops*NTIMES*arraySize/1.0e9/bestTime);
}

// Return ns accurate walltime
double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}

void SanitizeAndRoundArraySize(size_t * sizeBytes, cl_ulong maxAlloc, cl_ulong globalMemSize, size_t typeSize, size_t * arraySize, char * arrayName)
{
    if (*sizeBytes > maxAlloc) {
        *sizeBytes = (size_t) maxAlloc;
    }

	while (3*(*sizeBytes) > globalMemSize) {
		printf("Adjusting array size of %s from %zuMB to %zuMB\n", arrayName, *sizeBytes, (*sizeBytes)/2);
		(*sizeBytes) /= 2;
	}

    // After sanitizing, we must ensure the new arrray size is a multiple of 256, the largest local workgroup size
    if ( ((*sizeBytes)/typeSize) % 256 != 0) {
		// round down to multiple of 256
		printf("Adjusting array size from %zuMB to %zuMB\n", (*arraySize)*typeSize, (((*arraySize)/256)*256)*typeSize);
		(*sizeBytes) = (((*arraySize)/256)*256)*typeSize;
	}
}

// OpenCL functions
int InitialiseCLEnvironment(cl_platform_id **platform, cl_device_id ***device_id, cl_context *context, cl_command_queue *queue, cl_program *program, cl_ulong *maxAlloc, cl_ulong *globalMemSize)
{
	//error flag
	cl_int err;
	char infostring[1024];

	//get kernel from file
	FILE* kernelFile = fopen(kernelFileName, "rb");
	fseek(kernelFile, 0, SEEK_END);
	long fileLength = ftell(kernelFile);
	rewind(kernelFile);
	char *kernelSource = malloc(fileLength*sizeof(char));
	long read = fread(kernelSource, sizeof(char), fileLength, kernelFile);
	if (fileLength != read) printf("Error reading kernel file, line %d\n", __LINE__);
	fclose(kernelFile);

	//get platform and device information
	cl_uint numPlatforms;
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	*platform = calloc(numPlatforms, sizeof(cl_platform_id));
	*device_id = calloc(numPlatforms, sizeof(cl_device_id*));
	err |= clGetPlatformIDs(numPlatforms, *platform, NULL);
	CheckOpenCLError(err, __LINE__);
	cl_uint *numDevices;
	numDevices = calloc(numPlatforms, sizeof(cl_uint));

    // Retrieves information about the platform, and for each platform, about the devices.
	for (cl_uint i = 0; i < numPlatforms; i++) {
		clGetPlatformInfo((*platform)[i], CL_PLATFORM_VENDOR, sizeof(infostring), infostring, NULL);
		printf("\n---OpenCL: Platform Vendor %d: %s\n", i, infostring);

		err = clGetDeviceIDs((*platform)[i], CL_DEVICE_TYPE_ALL, 0, NULL, &(numDevices[i]));
		if (err == CL_DEVICE_NOT_FOUND)
			continue;
		CheckOpenCLError(err, __LINE__);
		(*device_id)[i] = malloc(numDevices[i] * sizeof(cl_device_id));
		err = clGetDeviceIDs((*platform)[i], CL_DEVICE_TYPE_ALL, numDevices[i], (*device_id)[i], NULL);
		CheckOpenCLError(err, __LINE__);
		for (cl_uint j = 0; j < numDevices[i]; j++) {
			char deviceName[200];
			cl_device_fp_config doublePrecisionSupport = 0;

			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            printf("---OpenCL:    Device found %d. %s\n", j, deviceName);

			cl_ulong maxAlloc;
			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, NULL);
#ifdef VERBOSE
            printf("---OpenCL:       CL_DEVICE_MAX_MEM_ALLOC_SIZE: %lu MB\n", maxAlloc/1024/1024);
#endif

            cl_uint cacheLineSize;
			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cacheLineSize), &cacheLineSize, NULL);
#ifdef VERBOSE
            printf("---OpenCL:       CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %u B\n", cacheLineSize);
#endif

            clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(doublePrecisionSupport), &doublePrecisionSupport, NULL);
			if ( doublePrecisionSupport == 0 )
				printf("---OpenCL:        Device %d does not support double precision!\n", j);
		}
	}

	// Get platform from user:
	cl_long chosenPlatform = -1;
	if (numPlatforms == 1) {
		chosenPlatform = 0;
		printf("Auto-selecting platform %lu.\n", chosenPlatform);
	} else while (chosenPlatform < 0) {
#ifdef AUTOPLATFORM
        chosenPlatform = AUTOPLATFORM;
        printf("Auto-selecting platform %lu.\n", chosenPlatform);
#else
		printf("\nChoose a platform: ");
		(void)!scanf("%ld", &chosenPlatform);
#endif
		if (chosenPlatform > (numPlatforms-1) || chosenPlatform < 0) {
			chosenPlatform = -1;
			printf("Invalid platform.\n");
		}
		if (numDevices[chosenPlatform] < 1) {
			chosenPlatform = -1;
			printf("Platform has no devices.\n");
		}
	}

    // Get device from user:
    cl_long chosenDevice = -1;
	if (numDevices[chosenPlatform] == 1) {
		chosenDevice = 0;
		printf("Auto-selecting device %lu.\n", chosenDevice);
	} else while (chosenDevice < 0) {
#ifdef AUTODEVICE
        chosenDevice = AUTODEVICE;
        printf("Auto-selecting device %lu.\n", chosenDevice);
#else
		printf("Choose a device: ");
		(void)!scanf("%ld", &chosenDevice);
#endif
		if (chosenDevice > (numDevices[chosenPlatform]-1) || chosenDevice < 0) {
			chosenDevice = -1;
			printf("Invalid device.\n");
		}
	}
	printf("\n");

	//store global mem size and max allocation size
	clGetDeviceInfo((*device_id)[chosenPlatform][chosenDevice], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(*globalMemSize), globalMemSize, NULL);
	clGetDeviceInfo((*device_id)[chosenPlatform][chosenDevice], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(*maxAlloc), maxAlloc, NULL);

	//create a context
	*context = clCreateContext(NULL, 1, &((*device_id)[chosenPlatform][chosenDevice]), NULL, NULL, &err);
	CheckOpenCLError(err, __LINE__);
	//create a queue
	*queue = clCreateCommandQueue(*context, (*device_id)[chosenPlatform][chosenDevice], 0, &err);
	CheckOpenCLError(err, __LINE__);

	//create the program with the source above
#ifdef VERBOSE
	printf("Creating CL Program...\n");
#endif
	*program = clCreateProgramWithSource(*context, 1, (const char**)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateProgramWithSource: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}

	//build program executable
#ifdef VERBOSE
	printf("Building CL Executable...\n");
#endif
	err = clBuildProgram(*program, 0, NULL, "-I.", NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error in clBuildProgram: %d, line %d.\n", err, __LINE__);
		char buffer[5000];
		clGetProgramBuildInfo(*program, (*device_id)[chosenPlatform][chosenDevice], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}

	free(numDevices);
	free(kernelSource);
	return EXIT_SUCCESS;
}

void CleanUpCLEnvironment(cl_platform_id **platform, cl_device_id ***device_id, cl_context *context, cl_command_queue *queue, cl_program *program)
{
	//release CL resources
	clReleaseProgram(*program);
	clReleaseCommandQueue(*queue);
	clReleaseContext(*context);

	cl_uint numPlatforms;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	for (cl_uint i = 0; i < numPlatforms; i++) {
		free((*device_id)[i]);
	}
	free(*platform);
	free(*device_id);
}

void CheckOpenCLError(cl_int err, int line)
{
	if (err != CL_SUCCESS) {
		char * errString;

		switch(err) {
			case   0: errString = "CL_SUCCESS"; break;
			case  -1: errString = "CL_DEVICE_NOT_FOUND"; break;
			case  -2: errString = "CL_DEVICE_NOT_AVAILABLE"; break;
			case  -3: errString = "CL_COMPILER_NOT_AVAILABLE"; break;
			case  -4: errString = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
			case  -5: errString = "CL_OUT_OF_RESOURCES"; break;
			case  -6: errString = "CL_OUT_OF_HOST_MEMORY"; break;
			case  -7: errString = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
			case  -8: errString = "CL_MEM_COPY_OVERLAP"; break;
			case  -9: errString = "CL_IMAGE_FORMAT_MISMATCH"; break;
			case -10: errString = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
			case -11: errString = "CL_BUILD_PROGRAM_FAILURE"; break;
			case -12: errString = "CL_MAP_FAILURE"; break;
			case -13: errString = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
			case -14: errString = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
			case -15: errString = "CL_COMPILE_PROGRAM_FAILURE"; break;
			case -16: errString = "CL_LINKER_NOT_AVAILABLE"; break;
			case -17: errString = "CL_LINK_PROGRAM_FAILURE"; break;
			case -18: errString = "CL_DEVICE_PARTITION_FAILED"; break;
			case -19: errString = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;
			case -30: errString = "CL_INVALID_VALUE"; break;
			case -31: errString = "CL_INVALID_DEVICE_TYPE"; break;
			case -32: errString = "CL_INVALID_PLATFORM"; break;
			case -33: errString = "CL_INVALID_DEVICE"; break;
			case -34: errString = "CL_INVALID_CONTEXT"; break;
			case -35: errString = "CL_INVALID_QUEUE_PROPERTIES"; break;
			case -36: errString = "CL_INVALID_COMMAND_QUEUE"; break;
			case -37: errString = "CL_INVALID_HOST_PTR"; break;
			case -38: errString = "CL_INVALID_MEM_OBJECT"; break;
			case -39: errString = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
			case -40: errString = "CL_INVALID_IMAGE_SIZE"; break;
			case -41: errString = "CL_INVALID_SAMPLER"; break;
			case -42: errString = "CL_INVALID_BINARY"; break;
			case -43: errString = "CL_INVALID_BUILD_OPTIONS"; break;
			case -44: errString = "CL_INVALID_PROGRAM"; break;
			case -45: errString = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
			case -46: errString = "CL_INVALID_KERNEL_NAME"; break;
			case -47: errString = "CL_INVALID_KERNEL_DEFINITION"; break;
			case -48: errString = "CL_INVALID_KERNEL"; break;
			case -49: errString = "CL_INVALID_ARG_INDEX"; break;
			case -50: errString = "CL_INVALID_ARG_VALUE"; break;
			case -51: errString = "CL_INVALID_ARG_SIZE"; break;
			case -52: errString = "CL_INVALID_KERNEL_ARGS"; break;
			case -53: errString = "CL_INVALID_WORK_DIMENSION"; break;
			case -54: errString = "CL_INVALID_WORK_GROUP_SIZE"; break;
			case -55: errString = "CL_INVALID_WORK_ITEM_SIZE"; break;
			case -56: errString = "CL_INVALID_GLOBAL_OFFSET"; break;
			case -57: errString = "CL_INVALID_EVENT_WAIT_LIST"; break;
			case -58: errString = "CL_INVALID_EVENT"; break;
			case -59: errString = "CL_INVALID_OPERATION"; break;
			case -60: errString = "CL_INVALID_GL_OBJECT"; break;
			case -61: errString = "CL_INVALID_BUFFER_SIZE"; break;
			case -62: errString = "CL_INVALID_MIP_LEVEL"; break;
			case -63: errString = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
			case -64: errString = "CL_INVALID_PROPERTY"; break;
			case -65: errString = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
			case -66: errString = "CL_INVALID_COMPILER_OPTIONS"; break;
			case -67: errString = "CL_INVALID_LINKER_OPTIONS"; break;
			case -68: errString = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
			case -1000: errString = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"; break;
			case -1001: errString = "CL_PLATFORM_NOT_FOUND_KHR"; break;
			case -1002: errString = "CL_INVALID_D3D10_DEVICE_KHR"; break;
			case -1003: errString = "CL_INVALID_D3D10_RESOURCE_KHR"; break;
			case -1004: errString = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR"; break;
			case -1005: errString = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR"; break;
			default: errString = "Unknown OpenCL error";
		}
		printf("OpenCL Error %d (%s), line %d\n", err, errString, line);
	}
}