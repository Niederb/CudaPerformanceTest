/*
 * This is a simple test program to measure the memcopy bandwidth of the GPU.
 * It can measure device to device copy bandwidth, host to device copy bandwidth
 * for pageable and pinned memory, and device to host copy bandwidth for pageable
 * and pinned memory.
 *
 * Usage:
 * ./bandwidthTest [option]...
 */

// CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

#include <cuda.h>

#include <memory>
#include <iostream>
#include <cassert>

static const char *sSDKsample = "CUDA Bandwidth Test";

// defines, project
#define MEMCOPY_ITERATIONS  50
#define DEFAULT_SIZE        ( 32 * ( 1 << 20 ) )    //32 M
#define DEFAULT_INCREMENT   (1 << 22)               //4 M
#define CACHE_CLEAR_SIZE    (1 << 24)               //16 M

//shmoo mode defines
#define SHMOO_MEMSIZE_MAX     (1 << 26)         //64 M
#define SHMOO_MEMSIZE_START   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_1KB   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_2KB   (1 << 11)         //2 KB
#define SHMOO_INCREMENT_10KB  (10 * (1 << 10))  //10KB
#define SHMOO_INCREMENT_100KB (100 * (1 << 10)) //100 KB
#define SHMOO_INCREMENT_1MB   (1 << 20)         //1 MB
#define SHMOO_INCREMENT_2MB   (1 << 21)         //2 MB
#define SHMOO_INCREMENT_4MB   (1 << 22)         //4 MB
#define SHMOO_LIMIT_20KB      (20 * (1 << 10))  //20 KB
#define SHMOO_LIMIT_50KB      (50 * (1 << 10))  //50 KB
#define SHMOO_LIMIT_100KB     (100 * (1 << 10)) //100 KB
#define SHMOO_LIMIT_1MB       (1 << 20)         //1 MB
#define SHMOO_LIMIT_16MB      (1 << 24)         //16 MB
#define SHMOO_LIMIT_32MB      (1 << 25)         //32 MB

//enums, project
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode  { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };

struct PerformanceMeasurement {
	PerformanceMeasurement() {
		bandwidth = 0;
		totalTime = 0;
		averageTime = 0;
		maxTime = std::numeric_limits<float>::min();
		minTime = std::numeric_limits<float>::max();
	}

	float bandwidth;
	float totalTime;
	float averageTime;
	float minTime;
	float maxTime;
};

const char *sMemoryCopyKind[] =
{
    "Device to Host",
    "Host to Device",
    "Device to Device",
    NULL
};

const char *sMemoryMode[] =
{
    "PINNED",
    "PAGEABLE",
    NULL
};

// if true, use CPU based timing for everything
static bool bDontUseGPUTiming;

int *pArgc = NULL;
char **pArgv = NULL;

int runTest(const int argc, const char **argv);
void testBandwidth(
		memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
PerformanceMeasurement testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc);
PerformanceMeasurement testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc);
PerformanceMeasurement testDeviceToDeviceTransfer(unsigned int memSize);
void printResultsReadable(unsigned int *memSizes, std::vector<PerformanceMeasurement> measurements, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc);
void printHelp(void);

int main(int argc, char **argv) {
    pArgc = &argc;
    pArgv = argv;

    // set logfile name and start logs
    printf("[%s] - Starting...\n", sSDKsample);

    int iRetVal = runTest(argc, (const char **)argv);

    if (iRetVal < 0) {
        checkCudaErrors(cudaSetDevice(0));

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
    }

    // finish
    printf("%s\n", (iRetVal==0) ? "Result = PASS" : "Result = FAIL");

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    exit((iRetVal==0) ? EXIT_SUCCESS : EXIT_FAILURE);
}

///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int runTest(const int argc, const char **argv)
{
    int startDevice = 0;
    int endDevice = 0;
    bool htod = false;
    bool dtoh = false;
    bool dtod = false;
    bool wc = false;
    char *device = NULL;
    printMode printmode = USER_READABLE;
    char *memModeStr = NULL;
    memoryMode memMode = PINNED;

    //process command line args
    if (checkCmdLineFlag(argc, argv, "help")) {
        printHelp();
        return 0;
    }

    if (checkCmdLineFlag(argc, argv, "csv")) {
        printmode = CSV;
    }

    if (getCmdLineArgumentString(argc, argv, "memory", &memModeStr)) {
        if (strcmp(memModeStr, "pageable") == 0) {
            memMode = PAGEABLE;
        } else if (strcmp(memModeStr, "pinned") == 0) {
            memMode = PINNED;
        } else {
            printf("Invalid memory mode - valid modes are pageable or pinned\n");
            printf("See --help for more information\n");
            return -1000;
        }
    } else {
        //default - pinned memory
        memMode = PINNED;
    }

    if (getCmdLineArgumentString(argc, argv, "device", &device)) {
        int deviceCount;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

        if (error_id != cudaSuccess) {
            printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            exit(EXIT_FAILURE);
        }

        if (deviceCount == 0) {
            printf("!!!!!No devices found!!!!!\n");
            return -2000;
        }

        if (strcmp(device, "all") == 0) {
            printf("\n!!!!!Cumulative Bandwidth to be computed from all the devices !!!!!!\n\n");
            startDevice = 0;
            endDevice = deviceCount-1;
        } else {
            startDevice = endDevice = atoi(device);

            if (startDevice >= deviceCount || startDevice < 0) {
                printf("\n!!!!!Invalid GPU number %d given hence default gpu %d will be used !!!!!\n", startDevice,0);
                startDevice = endDevice = 0;
            }
        }
    }

    printf("Running on...\n\n");

    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++) {
        cudaDeviceProp deviceProp;
        cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, currentDevice);

        if (error_id == cudaSuccess) {
            printf(" Device %d: %s\n", currentDevice, deviceProp.name);

            if (deviceProp.computeMode == cudaComputeModeProhibited) {
                fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
                checkCudaErrors(cudaSetDevice(currentDevice));

                // cudaDeviceReset causes the driver to clean up all state. While
                // not mandatory in normal operation, it is good practice.  It is also
                // needed to ensure correct operation when the application is being
                // profiled. Calling cudaDeviceReset causes all profile data to be
                // flushed before the application exits
                cudaDeviceReset();
                exit(EXIT_FAILURE);
            }
        } else {
            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            checkCudaErrors(cudaSetDevice(currentDevice));

            // cudaDeviceReset causes the driver to clean up all state. While
            // not mandatory in normal operation, it is good practice.  It is also
            // needed to ensure correct operation when the application is being
            // profiled. Calling cudaDeviceReset causes all profile data to be
            // flushed before the application exits
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }

    if (checkCmdLineFlag(argc, argv, "htod")) {
        htod = true;
    }

    if (checkCmdLineFlag(argc, argv, "dtoh")) {
        dtoh = true;
    }

    if (checkCmdLineFlag(argc, argv, "dtod")) {
        dtod = true;
    }

#if CUDART_VERSION >= 2020
    if (checkCmdLineFlag(argc, argv, "wc")) {
        wc = true;
    }
#endif

    if (checkCmdLineFlag(argc, argv, "cputiming")) {
        bDontUseGPUTiming = true;
    }

    if (!htod && !dtoh && !dtod) {
        //default:  All
        htod = true;
        dtoh = true;
        dtod = true;
    }

    if (htod) {
        testBandwidth(HOST_TO_DEVICE, printmode, memMode, startDevice, endDevice, wc);
    }

    if (dtoh) {
        testBandwidth(DEVICE_TO_HOST, printmode, memMode, startDevice, endDevice, wc);
    }

    if (dtod) {
        testBandwidth(DEVICE_TO_DEVICE, printmode, memMode, startDevice, endDevice, wc);
    }

    // Ensure that we reset all CUDA Devices in question
    for (int nDevice = startDevice; nDevice <= endDevice; nDevice++) {
        cudaSetDevice(nDevice);

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
//  Run a bandwidth test
///////////////////////////////////////////////////////////////////////////////
void
testBandwidth(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc)
{
	testBandwidthShmoo(kind, printmode, memMode, startDevice, endDevice, wc);
}

//////////////////////////////////////////////////////////////////////////////
// Intense shmoo mode - covers a large range of values with varying increments
//////////////////////////////////////////////////////////////////////////////
void
testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc)
{
    //count the number of copies to make
    unsigned int count = 1 + (SHMOO_LIMIT_20KB  / SHMOO_INCREMENT_1KB)
                         + ((SHMOO_LIMIT_50KB - SHMOO_LIMIT_20KB) / SHMOO_INCREMENT_2KB)
                         + ((SHMOO_LIMIT_100KB - SHMOO_LIMIT_50KB) / SHMOO_INCREMENT_10KB)
                         + ((SHMOO_LIMIT_1MB - SHMOO_LIMIT_100KB) / SHMOO_INCREMENT_100KB)
                         + ((SHMOO_LIMIT_16MB - SHMOO_LIMIT_1MB) / SHMOO_INCREMENT_1MB)
                         + ((SHMOO_LIMIT_32MB - SHMOO_LIMIT_16MB) / SHMOO_INCREMENT_2MB)
                         + ((SHMOO_MEMSIZE_MAX - SHMOO_LIMIT_32MB) / SHMOO_INCREMENT_4MB);

    unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
    std::vector<PerformanceMeasurement> measurements;

    // Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++) {
        cudaSetDevice(currentDevice);
        //Run the shmoo
        int iteration = 0;
        unsigned int memSize = 0;

        while (memSize <= SHMOO_MEMSIZE_MAX) {
            if (memSize < SHMOO_LIMIT_20KB) {
                memSize += SHMOO_INCREMENT_1KB;
            } else if (memSize < SHMOO_LIMIT_50KB) {
                memSize += SHMOO_INCREMENT_2KB;
            } else if (memSize < SHMOO_LIMIT_100KB) {
                memSize += SHMOO_INCREMENT_10KB;
            } else if (memSize < SHMOO_LIMIT_1MB) {
                memSize += SHMOO_INCREMENT_100KB;
            } else if (memSize < SHMOO_LIMIT_16MB) {
                memSize += SHMOO_INCREMENT_1MB;
            } else if (memSize < SHMOO_LIMIT_32MB) {
                memSize += SHMOO_INCREMENT_2MB;
            } else {
                memSize += SHMOO_INCREMENT_4MB;
            }

            memSizes[iteration] = memSize;
            PerformanceMeasurement measurement;
            switch (kind) {
                case DEVICE_TO_HOST:
                	measurement = testDeviceToHostTransfer(memSizes[iteration], memMode, wc);
                    break;

                case HOST_TO_DEVICE:
                	measurement = testHostToDeviceTransfer(memSizes[iteration], memMode, wc);
                    break;

                case DEVICE_TO_DEVICE:
                	measurement = testDeviceToDeviceTransfer(memSizes[iteration]);
                    break;
            }
            measurements.push_back(measurement);

            iteration++;
            printf(".");
        }
    } // Complete the bandwidth computation on all the devices

    //print results
    printf("\n");

    if (CSV == printmode) {
        //printResultsCSV(memSizes, measurements, count, kind, memMode, (1 + endDevice - startDevice), wc);
    } else {
        printResultsReadable(memSizes, measurements, count, kind, memMode, (1 + endDevice - startDevice), wc);
    }

    //clean up
    free(memSizes);
    //free(bandwidths);
}

///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
PerformanceMeasurement
testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
    StopWatchInterface *timer = NULL;
    PerformanceMeasurement m;
    float elapsedTimeInMs = 0.0f;
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;
    cudaEvent_t start, stop;

    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //allocate host memory
    if (PINNED == memMode) {
        //pinned memory mode - use special function to get OS-pinned memory
#if CUDART_VERSION >= 2020
        checkCudaErrors(cudaHostAlloc((void **)&h_idata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
        checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
#else
        checkCudaErrors(cudaMallocHost((void **)&h_idata, memSize));
        checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
    } else {
        //pageable memory mode - use malloc
        h_idata = (unsigned char *)malloc(memSize);
        h_odata = (unsigned char *)malloc(memSize);

        if (h_idata == 0 || h_odata == 0) {
            fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    }

    //initialize the memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++) {
        h_idata[i] = (unsigned char)(i & 0xff);
    }

    // allocate device memory
    unsigned char *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));

    //initialize the device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,
                               cudaMemcpyHostToDevice));

    //copy data from GPU to Host
    sdkStartTimer(&timer);

	for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
		checkCudaErrors(cudaEventRecord(start, 0));
		if (PINNED == memMode) {
			checkCudaErrors(cudaMemcpyAsync(h_odata, d_idata, memSize,
				cudaMemcpyDeviceToHost, 0));
		} else {
			checkCudaErrors(cudaMemcpy(h_odata, d_idata, memSize,
				cudaMemcpyDeviceToHost));
		}
		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
		m.totalTime += elapsedTimeInMs;
		m.maxTime = std::max(m.maxTime, elapsedTimeInMs);
		m.minTime = std::min(m.minTime, elapsedTimeInMs);
	}

    //get the total elapsed time in ms
    sdkStopTimer(&timer);

    if (PINNED != memMode || bDontUseGPUTiming) {
    	m.totalTime = sdkGetTimerValue(&timer);
    }

    //calculate bandwidth in MB/s
    m.bandwidth = ((float)(1<<10) * memSize * (float)MEMCOPY_ITERATIONS) /
                     (m.totalTime * (float)(1 << 20));
    m.averageTime = m.totalTime / MEMCOPY_ITERATIONS;

    //clean up memory
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    sdkDeleteTimer(&timer);

    if (PINNED == memMode) {
        checkCudaErrors(cudaFreeHost(h_idata));
        checkCudaErrors(cudaFreeHost(h_odata));
    } else {
        free(h_idata);
        free(h_odata);
    }
    checkCudaErrors(cudaFree(d_idata));
    return m;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a host to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
PerformanceMeasurement
testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    PerformanceMeasurement m;
    cudaEvent_t start, stop;
    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //allocate host memory
    unsigned char *h_odata = NULL;

    if (PINNED == memMode) {
#if CUDART_VERSION >= 2020
        //pinned memory mode - use special function to get OS-pinned memory
        checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
#else
        //pinned memory mode - use special function to get OS-pinned memory
        checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
    } else {
        //pageable memory mode - use malloc
        h_odata = (unsigned char *)malloc(memSize);

        if (h_odata == 0) {
            fprintf(stderr, "Not enough memory available on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    }

    unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
    unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

    if (h_cacheClear1 == 0 || h_cacheClear1 == 0) {
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    //initialize the memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++) {
        h_odata[i] = (unsigned char)(i & 0xff);
    }

    for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
        h_cacheClear1[i] = (unsigned char)(i & 0xff);
        h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
    }

    //allocate device memory
    unsigned char *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));

    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    //copy host memory to device memory

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
		checkCudaErrors(cudaEventRecord(start, 0));
		if (PINNED == memMode) {
			checkCudaErrors(cudaMemcpyAsync(d_idata, h_odata, memSize,
				cudaMemcpyHostToDevice, 0));
		} else {
			checkCudaErrors(cudaMemcpy(d_idata, h_odata, memSize,
				cudaMemcpyHostToDevice));
		}
		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
		m.totalTime += elapsedTimeInMs;
		m.maxTime = std::max(m.maxTime, elapsedTimeInMs);
		m.minTime = std::min(m.minTime, elapsedTimeInMs);
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    //total elapsed time in ms
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    if (PINNED != memMode || bDontUseGPUTiming) {
        elapsedTimeInMs = sdkGetTimerValue(&timer);
    }
    sdkResetTimer(&timer);

    //calculate bandwidth in MB/s
    m.bandwidth = ((float)(1<<10) * memSize * (float)MEMCOPY_ITERATIONS) /
                     (m.totalTime * (float)(1 << 20));
    m.averageTime = m.totalTime / MEMCOPY_ITERATIONS;

    //clean up memory
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    sdkDeleteTimer(&timer);

    if (PINNED == memMode) {
        checkCudaErrors(cudaFreeHost(h_odata));
    } else {
        free(h_odata);
    }

    free(h_cacheClear1);
    free(h_cacheClear2);
    checkCudaErrors(cudaFree(d_idata));

    return m;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a device to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
PerformanceMeasurement
testDeviceToDeviceTransfer(unsigned int memSize) {
    PerformanceMeasurement m;
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    cudaEvent_t start, stop;

    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //allocate host memory
    unsigned char *h_idata = (unsigned char *)malloc(memSize);

    if (h_idata == 0) {
        fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    //initialize the host memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++) {
        h_idata[i] = (unsigned char)(i & 0xff);
    }

    //allocate device memory
    unsigned char *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));
    unsigned char *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, memSize));

    //initialize memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,
                               cudaMemcpyHostToDevice));

    //run the memcopy
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
        checkCudaErrors(cudaEventRecord(start, 0));
        checkCudaErrors(cudaMemcpy(d_odata, d_idata, memSize,
                                           cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
        m.totalTime += elapsedTimeInMs;
        m.maxTime = std::max(m.maxTime, elapsedTimeInMs);
        m.minTime = std::min(m.minTime, elapsedTimeInMs);
    }

    checkCudaErrors(cudaEventRecord(stop, 0));

    //Since device to device memory copies are non-blocking,
    //cudaDeviceSynchronize() is required in order to get
    //proper timing.
    checkCudaErrors(cudaDeviceSynchronize());

    //get the total elapsed time in ms
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    if (bDontUseGPUTiming) {
        elapsedTimeInMs = sdkGetTimerValue(&timer);
    }

    //calculate bandwidth in MB/s
    m.bandwidth = ((float)(1<<10) * memSize * (float)MEMCOPY_ITERATIONS) /
                     (m.totalTime * (float)(1 << 20));
    m.averageTime = m.totalTime / MEMCOPY_ITERATIONS;

    //clean up memory
    sdkDeleteTimer(&timer);
    free(h_idata);
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return m;
}

/////////////////////////////////////////////////////////
//print results in an easily read format
////////////////////////////////////////////////////////
void printResultsReadable(unsigned int *memSizes, std::vector<PerformanceMeasurement> measurements, unsigned int count, memcpyKind kind, memoryMode memMode, int iNumDevs, bool wc)
{
    printf(" %s Bandwidth, %i Device(s)\n", sMemoryCopyKind[kind], iNumDevs);
    printf(" %s Memory Transfers\n", sMemoryMode[memMode]);

    if (wc) {
        printf(" Write-Combined Memory Writes are Enabled");
    }

    printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");

    for (int i = 0; i < measurements.size(); i++) {
        printf("   %u\t\t\t%s", memSizes[i], (memSizes[i] < 10000)? "\t" : "");
        printf("\t%.2f", measurements[i].totalTime);
        printf("\t%.2f", measurements[i].averageTime);
        printf("\t%.2f", measurements[i].minTime);
        printf("\t%.2f", measurements[i].maxTime);
        printf("\t%.1f\n", measurements[i].bandwidth);
    }
}

///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
    printf("Usage:  bandwidthTest [OPTION]...\n");
    printf("Test the bandwidth for device to host, host to device, and device to device transfers\n");
    printf("\n");
    printf("Example:  measure the bandwidth of device to host pinned memory copies in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments\n");
    printf("./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 --increment=1024 --dtoh\n");

    printf("\n");
    printf("Options:\n");
    printf("--help\tDisplay this help menu\n");
    printf("--csv\tPrint results as a CSV\n");
    printf("--device=[deviceno]\tSpecify the device device to be used\n");
    printf("  all - compute cumulative bandwidth on all the devices\n");
    printf("  0,1,2,...,n - Specify any particular device to be used\n");
    printf("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
    printf("  pageable - pageable memory\n");
    printf("  pinned   - non-pageable system memory\n");

    printf("--htod\tMeasure host to device transfers\n");
    printf("--dtoh\tMeasure device to host transfers\n");
    printf("--dtod\tMeasure device to device transfers\n");
#if CUDART_VERSION >= 2020
    printf("--wc\tAllocate pinned memory as write-combined\n");
#endif
    printf("--cputiming\tForce CPU-based timing always\n");
}
