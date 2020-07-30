# HOGACVOC-notebook
Hand On GPU-Accelerated Computer Vision with OpenCV 一书的阅读笔记

通用**Makefile**:

```
SOURCE = $(wildcard *.cpp)
TARGETS = $(patsubst %.cpp, %, $(SOURCE))

CC = g++
CPPFLAGS = -std=c++11 -Wall -g `pkg-config --cflags --libs opencv4`

all: $(TARGETS)

$(TARGETS):%:%.cpp
	$(CC) $< $(CPPFLAGS) -o $@

.PHONY: clean all

clean:
	rm -rf $(TARGETS)
```
### 元素个数超过线程数

```cpp
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Defining number of elements in array
#define N 600000
__global__
void gpuAdd(int* d_a, int* d_b, int* d_c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("my id is:%d\n", tid);
    while (tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[]) {

    // Declare host and device arrays
    int h_a[N], h_b[N], h_c[N];
    int* d_a;
    int* d_b;
    int* d_c;
    
    // Allocate memory on Device
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMalloc((void**)&d_b, N*sizeof(int));
    cudaMalloc((void**)&d_c, N*sizeof(int));

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_a[i] = 2 * i*i;
        h_b[i] = i;
    }

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    // kernel Call
    gpuAdd<< <512, 512>> >(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    // This ensures that kernel execution is finishes before going forward
    cudaDeviceSynchronize();
    int Correct = 1;
    printf("Vector addition on GPU \n");
    for (int i = 0; i < N; i++) {
        if (h_a[i] + h_b[i] != h_c[i]) {
            Correct = 0;
        }
    }
    if (Correct == 1) {
        printf("GPU has computed Sum Correctly\n");
    }
    else {
        printf("There is an Error in GPU Computation\n");
    }

    // Free up memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

```
### 存储器结构

![image](6FAFE98991204843B8E0D4D3D2165704)

![image](C4CE871EB949435DA678FF17A383A974)


#### Global Memory

```cpp
#include <stdio.h>
#define N 5

__global__
void gpu_global_memory(int *d_a) {
    d_a[threadIdx.x] = threadIdx.x;
}

int main(int argc, char *argv[]) {
    int h_a[N];
    int *d_a;

    cudaMalloc((void **)&d_a, sizeof(int) * N);
    cudaMemcpy((void *)d_a, (void *)h_a, sizeof(int)*N, cudaMemcpyHostToDevice);

    gpu_global_memory <<<1, N>>>(d_a);
    cudaMemcpy((void *)h_a, (void *)d_a, sizeof(int) *N, cudaMemcpyDeviceToHost);

    printf("Array in Global Memory is: \n");

    for (int i = 0; i < N; i++) {
        printf("At index: %d --> %d \n", i, h_a[i]);
    }
    return 0;
}
```

#### Local memory and registers

本地存储器（local memory）和寄存器文件（register files）对于每个线程都是唯一的。register files是每个线程可用的最快的内存。当内核变量不适合register files时，它们将使用Local memory。这称为**寄存器溢出**（register spilling）。基本上，local memory是global memory的一部分，每个线程都是唯一的。与register files相比，访问local memory的速度较慢。尽管local memory缓存在L1和L2缓存中，但寄存器溢出可能不会对程序造成不利影响。

```cpp
#include "stdio.h"
#define N 5

__global__ 
void gpu_local_memory(int d_in) {
    int t_local;
    t_local = d_in * threadIdx.x;
    printf("Value of Local variable in current thread is: %d \n", t_local);
}

int main(int argc, char **argv) {
    printf("Use of Local Memory on GPU:\n");
    gpu_local_memory << <1, N >> >(5);
    cudaDeviceSynchronize();

    return 0;
}
```

`t_local`变量对于每个线程而言都是局部的, 并存储在register files中. 当将此变量用于内核函数中的计算时, 计算将是最快的.


#### cache memory

在最新的GPU上，每个多处理器（per multiprocessor）都有一个L1缓存，而在所有多处理器之间共享一个L2缓存。全局和本地存储器都使用这些缓存。由于L1接近线程执行，因此速度非常快。如先前的内存架构图所示，L1缓存和共享内存使用相同的64 KB。两者都可以配置为使用64 KB中的多少字节。所有全局内存访问都通过L2缓存进行。纹理内存和常量内存具有各自的缓存。

#### shared memory

```cpp
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void gpu_shared_memory(float *d_a) {
    
    int i;
    int index = threadIdx.x;
    float average;
    float sum = 0.0f;
    // Defining shared memory
    __shared__ float sh_arr[10];

    sh_arr[index] = d_a[index];
    // This directive ensure all the writes to shared memory have completed

    __syncthreads();
    for (i = 0; i <= index; i++) {
        sum += sh_arr[i];
    }
    average = sum / (index + 1.0f);
    d_a[index] = average;

    // This statement is redundant and will have no effect on overall code
    // execution
    sh_arr[index] = average;
}

int main(int argc, char* argv[]) {
    
    float h_a[10];
    float *d_a;

    // Initialize host Array
    for (int i = 0; i < 10; i++) {
        h_a[i] = i;
    }

    // allocate global memory on the device
    cudaMalloc((void**)&d_a, sizeof(float)*10);

    // copy data from host memory to device memory
    cudaMemcpy((void*)d_a, (void*)h_a, sizeof(float)*10, cudaMemcpyHostToDevice);
    gpu_shared_memory<<<1, 10>>>(d_a);

    // copy the modified array back to the host
    cudaMemcpy((void*)h_a, (void*)d_a, sizeof(float)*10, cudaMemcpyDeviceToHost);
    printf("Use of Shared Memory on GPU: \n");

    for (int i = 0; i < 10; i++) {
        printf("The running average after %d element is %f\n", i, h_a[i]);
    }

    return 0;
}
```

#### 原子操作

```cpp
#include <stdio.h>

#define NUM_THREADS 10000000
#define SIZE  10

#define BLOCK_WIDTH 100

__global__ 
void gpu_increment_without_atomic(int *d_a) {
    
    // Calculate thread id for current thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread increments elements wrapping at SIZE variable
    tid = tid % SIZE;
    // printf("My id is %d\n", tid);
    d_a[tid] += 1;
}

__global__
void gpu_increment_atomic(int *d_a) {

    // Calculate thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread increments elements which wraps at SIZE
    tid = tid % SIZE;
    atomicAdd(&d_a[tid], 1);
}

int main(int argc, char **argv) {

    printf("%d total threads in %d blocks writing into %d array elements\n",
    	NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);
    
    // declare and allocate host memory
    int h_a[SIZE];
    const int ARRAY_BYTES = SIZE * sizeof(int);
    
    // declare and allocate GPU memory
    int * d_a;
    cudaMalloc((void **)&d_a, ARRAY_BYTES);
    //Initialize GPU memory to zero
    cudaMemset((void *)d_a, 0, ARRAY_BYTES);
    
    // gpu_increment_without_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_a);
    gpu_increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_a);
    
    // copy back the array to host memory
    cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    printf("Number of times a particular Array index has been incremented without atomic add is: \n");
    for (int i = 0; i < SIZE; i++) {
        printf("index: %d --> %d times\n ", i, h_a[i]);
    }
    
    cudaFree(d_a);
    return 0;
}
```

#### Constant memory

```cpp
#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Defining two constants
__constant__ int constant_f;
__constant__ int constant_g;
#define N 5

// kernel function for using constant memory
__global__
void gpu_constant_memory(float *d_in, float *d_out) {
    // Getting thread index for current kernel
    int tid = threadIdx.x;
    d_out[tid] = constant_f * d_in[tid] + constant_g;

}

int main(int argc, char *argv[]) {

    // Defining Arrays for host
    float h_in[N], h_out[N];

    // Defining Pointers for device
    float *d_in, *d_out;
    int h_f = 2;
    int h_g = 20;

    // allocate the memory on the cpu
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    // Intializing Array
    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    // Copy Array from host to device
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    // Copy constants to constant memory
    cudaMemcpyToSymbol(constant_f, &h_f, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_g, &h_g, sizeof(int), 0, cudaMemcpyHostToDevice);

    // Calling kernel with one block and N threads per block
    gpu_constant_memory<<<1, N>>>(d_in, d_out);

    // Coping result back to host from device memory
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Printing result on console
    printf("Use of Constant memory on GPU \n");
    for (int i = 0; i < N; i++) {
        printf("The expression for index %f is %f\n", h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

`cudaMemcpyToSymbol`第一个参数是目的地址，使用`__constant__`关键字定义；第二个参数是主机地址；第三个参数是传输大小，第四个参数是内存偏移，默认为0，第五个参数是数据传输方向，默认为host到device。

constant memory是只读存储器。

#### Texture memory

```cpp
#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 10
#define N 10

// Define texture reference for 1-d access
texture <float, 1, cudaReadModeElementType> textureRef;

__global__ void gpu_texture_memory(int n, float *d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float tmp = tex1D(textureRef, float(idx));
	    d_out[idx] = tmp;
    }
}

int main(int argc, char *argv[]) {

    // Calculate number of block to launch
    int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);
    float *d_out;
    // allocate space on the device for the results
    cudaMalloc((void**)&d_out, sizeof(float) * N);
    // allocate space on the host for the results
    float *h_out = (float*)malloc(sizeof(float) * N);
    float h_in[N];
    
    for (int i = 0; i < N; i++) {
        h_in[i] = float(i);
    }

    // Define CUDA Array
    cudaArray *cu_Array;
    cudaMallocArray(&cu_Array, &textureRef.channelDesc, N, 1);

    cudaMemcpyToArray(cu_Array, 0, 0, h_in, sizeof(float)*N, cudaMemcpyHostToDevice);

    // bind a texture to the CUDA array
    cudaBindTextureToArray(textureRef, cu_Array);

    gpu_texture_memory<<<num_blocks, NUM_THREADS>>>(N, d_out);

    // copy result to host
    cudaMemcpy(h_out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    printf("Use of Texture memory on GPU: \n");
    // Print the result
    for (int i = 0; i < N; i++) {
        printf("Average between two nearest element is: %f\n", h_out[i]);
    }

    free(h_out);
    cudaFree(d_out);
    cudaFreeArray(cu_Array);
    cudaUnbindTexture(textureRef);

    return 0;
}
```

#### dot product

```cpp
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N (102400+1080*720*3+921123)
#define threadsPerBlock 512

using namespace std;

__global__ void gpu_dot(float *d_a, float *d_b, float *d_c) {
    // Define Shared Memory
    __shared__ float partial_sum[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = threadIdx.x;

    float sum = 0;
    while (tid < N) {
        sum += d_a[tid] * d_b[tid];
	tid += blockDim.x * gridDim.x;
    }

    // set the partial sum in shared memory
    partial_sum[index] = sum;

    // synchronize threads in this block
    __syncthreads();

    // Calculate Partial sum for a current block using data in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (index < i) {
	    partial_sum[index] += partial_sum[index + i];
	}
	__syncthreads();
	i /= 2;
    }

    // Store result of partial sum for a block in global memory
    if (index == 0) {
        d_c[blockIdx.x] = partial_sum[0];
    }
}

#define cpu_sum(x) (x*(x+1))

int main(int argc, char *argv[]) {
    
    float *h_a, *h_b, h_c, *partial_sum;
    float *d_a, *d_b, *d_partial_sum;

    // Calucate number of blocks and number of threads
    int block_calc = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (32 < block_calc ? 32 : block_calc);

    // cout << "block_calc is: " << block_calc << endl;
    // cout << "blocksPerGrid is: " << blocksPerGrid << endl;

    // allocate memory on the cpu side
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    partial_sum = (float*)malloc(blocksPerGrid * sizeof(float));

    // allocate the memory on the gpu
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_partial_sum, blocksPerGrid * sizeof(float));

    // fill in the host memory with data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
	h_b[i] = 2;
    }

    // copy the arrays to the device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    cout << "blocksPerGrid is: " << blocksPerGrid << endl;
    cout << "threadsPerBlock is: " << threadsPerBlock << endl;
    
    gpu_dot <<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_partial_sum);

    // copy the array back to the host
    cudaMemcpy(partial_sum, d_partial_sum, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate final dot product
    h_c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        h_c += partial_sum[i];
    }

    cout << "The computed dot product is: " << h_c << endl;

    if ((cpu_sum((float)(N-1)) - h_c) < 1) {
        printf("The dot product computed by GPU is correct\n");
    }
    else {
        printf("Error in dot product computation\n");
    }

    // free memory on the gpu side
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_sum);
    // free memory on the cpu side
    free(h_a);
    free(h_b);
    free(partial_sum);

    return 0;
}
```

#### Matrix Mul

```cpp
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iomanip>

// This defines size of a small square box or thread dimensions in one block
#define TILE_SIZE 2

using namespace std;

// Matrix multiplication using non shared kernel
__global__ void gpu_Matrix_Mul_nonshared(float *d_a, float *d_b, float *d_c, const int size) {
    int row, col;
    col = TILE_SIZE * blockIdx.x + threadIdx.x;
    row = TILE_SIZE * blockIdx.y + threadIdx.y;

    for (int k = 0; k < size; k++) {
        d_c[row*size + col] += d_a[row*size + k] * d_b[k*size + col];
    }
}

// shared
__global__ void gpu_Matrix_Mul_shared(float *d_a, float *d_b, float *d_c, const int size) {
    int row, col;

    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

    // calculate thread id
    col = TILE_SIZE * blockIdx.x + threadIdx.x;
    row = TILE_SIZE * blockIdx.y + threadIdx.y;

    for (int i = 0; i < size / TILE_SIZE; i++) {
        shared_a[threadIdx.y][threadIdx.x] = d_a[row * size + (i*TILE_SIZE + threadIdx.x)];
	shared_b[threadIdx.y][threadIdx.x] = d_b[(i*TILE_SIZE + threadIdx.y) * size + col];
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            d_c[row*size+col] += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
        }
        __syncthreads(); // for synchronizing the threads
    }
}

int main(int argc, char *argv[]) {
    // Define size of the matrix
    const int size = 4;
    // Define host and device arrays
    float h_a[size][size], h_b[size][size], h_result[size][size];
    float *d_a, *d_b, *d_result; // device array
    // input in host array
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
	    h_a[i][j] = i;
	    h_b[i][j] = j;
	}
    }

    cudaMalloc((void**)&d_a, size*size*sizeof(int));
    cudaMalloc((void**)&d_b, size*size*sizeof(int));
    cudaMalloc((void**)&d_result, size*size*sizeof(int));

    // copy host array to device array
    cudaMemcpy(d_a, h_a, size*size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size*size*sizeof(int), cudaMemcpyHostToDevice);
    
    // calling kernel
    dim3 dimGrid(size / TILE_SIZE, size / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // gpu_Matrix_Mul_nonshared<<<dimGrid, dimBlock>>>(d_a, d_b, d_result, size);
    gpu_Matrix_Mul_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_result, size);

    cudaMemcpy(h_result, d_result, size*size*sizeof(int), cudaMemcpyDeviceToHost);

    // Matrix A is:
    cout << "Matrix A is: " << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
	    cout << setw(3) << h_a[i][j] << " ";
	}
	cout << endl;
    }

    // Matrix B is:
    cout << "Matrix B is: " << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
	    cout << setw(3) << h_b[i][j] << " ";
	}
	cout << endl;
    }

    // Matrix A * B is:
    cout << "Matrix A*B is: " << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
	    cout << setw(3) << h_result[i][j] << " ";
	}
	cout << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
```

#### CUDA Event

```cpp
#include <stdio.h>
#include <iostream>

// Defining Number of elements in Array
#define N 5
// Defining vector addition function for CPU
__global__
void gpuAdd(int *d_a, int *d_b, int *d_c) {
    // Getting block index of current kernel
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

int main(int argc, char *argv[]) {

    //Defining host arrays
    int h_a[N], h_b[N], h_c[N];
    
    //Defining device pointers
    int *d_a, *d_b, *d_c;
    // allocate the memory
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));
    //Initializing Arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 2*i*i;
        h_b[i] = i ;
    }
    
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);
    // All GPU code for which performance needs to be measured allocate the memory
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Copy input arrays from host to device memory
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    gpuAdd<<<512, 512>>>(d_a, d_b, d_c);

    // Copy result back to host memory from device memory
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("Time to add %d numbers: %3.1f ms\n", N, elapsedTime);

    return 0;

}
```

#### Nvidia Visual Profiler

```cpp
nvprof -o profile.out -s ./cudaEvent
```

#### Error handling in CUDA

```cpp
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void gpuAdd(int *d_a, int *d_b, int *d_c) {
    *d_c = *d_a + *d_b;
}

int main(int argc, char *argv[]) {

    // Defining host variables
    int h_a, h_b, h_c;
    // Defining Device Pointers
    int *d_a, *d_b, *d_c;
    // Initializing host variables
    h_a = 1;
    h_b = 4;

    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&d_a, sizeof(int));
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "cudaMalloc failed!");
	goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_b, sizeof(int));
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "cudaMalloc failed!");
	goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_c, sizeof(int));
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "cudaMalloc failed!");
	goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "cudaMalloc failed!");
	goto Error;
    }

    cudaStatus = cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    gpuAdd<<<1, 1>>>(d_a, d_b, d_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaSuccess != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
    }

    printf("Passing Parameter by Reference Output: %d + %d = %d\n", h_a, h_b, h_c);

Error:
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
```

#### Debugging tools

```
cuda-gdb
```

#### Performance improvement of CUDA programs

- Using an optimum number of blocks and threads
- Maximizing arithmetic efficiency
- Using coalesced or strided memory access
- Avoiding thread divergence

```cpp
// Thread divergence by way of branching
tid = ThreadId
if (tid%2 == 0) {
    Some Branch code;
}
else {
    Some other code;
}

// Thread divergence by way of looping
Pre-loop code
for (i=0; i<tid; i++) {
    Some loop code;
}
Post loop code;
```
- Using page-locked host memory

> In every example until this point, we have used the `malloc` function to allocate memory on the host, which allocates standard **pageable memory** on the host. CUDA provides another API called `cudaHostAlloc()`, which allocates **page-locked host memory** or what is sometimes referred to as **pinned memory**. It guarantees that the operating system will never page this memory out of this disk and that it will remain in physical memory. So, any application can access the physical address of the buffer. This proerty helps the GPU copy data to and from the host via **Direct Memory Access(DMA)** without CPU intervention. This helps improve the performance of memory transfer operations. But page-locked memory should be used with proper care because this memory is not swapped out of disk; your system may run out of memory. It may effect the performance of other applications running on the system. You can use this API to allocate memory that is used to transfer data to a device, using the Memcpy operation. The syntax of using this API is shown as follows:

- Allocate Memory:

```cpp
cudaHostAlloc((void**)&h_a, sizeof(*h_a), cudaHostAllocDefault);
```

- Free Memory:

```cpp
cudaFreeHost(h_a);
```

#### CUDA streams

> We have seen that the GPU provides a great performance improvement in **data parallelism** when a single instruction operates on multiple data items. We have not seen **task parallelism** where more than one kernel function, which are independent of each other, operate in parallel. For example, one function may be computing pixel values while another function is downloading something from the internet. We know that the CPU provides a very flexible method for this kind of task parallelism. The GPU also provides this capability, but it is not as flexible as the CPU. This task parallelism is achieved by using CUDA streams, which we will see in detail in this section.

> **A CUDA stream is nothing but a queue of GPU operations that execute in a specific order.** ++These functions include kernel functions, memory copy operations, and CUDA event operations. The order in which they are added to the queue will determine the order of their execution.++ Each CUDA stream can be considered a single task, so we can start multiple streams to do multiple tasks in parallel.

#### Using multiple CUDA streams

```cpp
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
// Defining number of elements in Array
#define N 500

using namespace std;

// Defining kernel function for vector addition
__global__
void gpuAdd(int *d_a, int *d_b, int *d_c) {
    // Getting Thread index of current kernel
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
	tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char *argv[]) {

    // Defining host arrays
    int *h_a, *h_b, *h_c;
    // Defining device pointers for stream 0
    int *d_a0, *d_b0, *d_c0;
    // Defining device pointers for stream 1
    int *d_a1, *d_b1, *d_c1;
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);

    // Allocate memory for host pointers
    cudaHostAlloc((void**)&h_a, 2*N*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, 2*N*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c, 2*N*sizeof(int), cudaHostAllocDefault);

    // Allocate memory for device pointers
    cudaMalloc((void**)&d_a0, N*sizeof(int));
    cudaMalloc((void**)&d_b0, N*sizeof(int));
    cudaMalloc((void**)&d_c0, N*sizeof(int));
    cudaMalloc((void**)&d_a1, N*sizeof(int));
    cudaMalloc((void**)&d_b1, N*sizeof(int));
    cudaMalloc((void**)&d_c1, N*sizeof(int));

    // Initializing Arrays
    for (int i = 0; i < N * 2; i++) {
        h_a[i] = 2 * i * i;
	h_b[i] = i;
    }

    // Asynchronous Memory Copy Operation for both streams
    cudaMemcpyAsync(d_a0,   h_a, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_a1, h_a+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b0,   h_b, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_b1, h_b+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);

    // Kernel Call
    gpuAdd<<<512, 512, 0, stream0>>>(d_a0, d_b0, d_c0);
    gpuAdd<<<512, 512, 0, stream1>>>(d_a1, d_b1, d_c1);

    // Copy result back to host memory from device memory
    cudaMemcpyAsync(h_c,   d_c0, N*sizeof(int), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(h_c+N, d_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1);

    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("Time to add %d numbers: %3.1f ms\n", 2*N, elapsedTime);

    int Correct = 1;
    printf("Vector addition on GPU \n");
    // Printing result on console
    for (int i = 0; i < 2*N; i++) {
	cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
        if ((h_a[i] + h_b[i] != h_c[i])) {
	    Correct = 0;
	}
    }

    if (1 == Correct) {
        printf("GPU has computed Sum Correctly\n");
    }

    else {
        printf("There is an Error in GPU Computation\n");
    }

    // Free up memory
    cudaFree(d_a0);
    cudaFree(d_b0);
    cudaFree(d_c0);
    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_b1);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
```

#### Acceleration of sorting algorithms using CUDA

```cpp
#include "device_launch_parameters.h"
#include <stdio.h>

#define arraySize 5
#define threadPerBlock 5
// Kernel Function for Rank sort
__global__
void addKernel(int *d_a, int *d_b) {
    int count = 0;
    int tid = threadIdx.x;
    int ttid = blockIdx.x * threadPerBlock + tid;
    int val = d_a[ttid];
    __shared__ int cache[threadPerBlock];
    for (int i = tid; i < arraySize; i += threadPerBlock) {
        cache[tid] = d_a[i];
	__syncthreads();
	for (int j = 0; j < threadPerBlock; ++j) {
	    if (val > cache[j]) {
	        count++;
	    }
	}
	__syncthreads();
    }
    d_b[count] = val;
}

int main(int argc, char *argv[]) {

    int h_a[arraySize] = {5, 9, 3, 4, 8};
    int h_b[arraySize];
    int *d_a, *d_b;

    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, arraySize * sizeof(int));

    // Copy input vector from host memory to GPU buffers.
    cudaMemcpy(d_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<arraySize/threadPerBlock, threadPerBlock>>>(d_a, d_b);

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(h_b, d_b, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("The Enumeration sorted Array is: \n");
    for (int i = 0; i < arraySize; i++) {
        printf("%d\n", h_b[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
```

#### Histogram calucation on the GPU with CUDA

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1000
#define NUM_BIN 16

__global__
void histogram_without_atomic(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];
    if (tid < SIZE) {
        d_b[item]++;
    }
}

__global__
void histogram_atomic(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];
    if (tid < SIZE) {
        atomicAdd(&(d_b[item]), 1);
    }
}

__global__ void histogram_shared_memory(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;
    __shared__ int cache[256];

    cache[threadIdx.x] = 0;
    __syncthreads();

    while (tid < SIZE) {
        atomicAdd(&(cache[d_a[tid]]), 1);
	tid += offset;
    }
    __syncthreads();
    atomicAdd(&(d_b[threadIdx.x]), cache[threadIdx.x]);
}

int main(int argc, char *argv[]) {

    int h_a[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = i % NUM_BIN;
    }
    int h_b[NUM_BIN];
    for (int i = 0; i < NUM_BIN; i++) {
        h_b[i] = 0;
    }

    // declare GPU memory pointers
    int *d_a;
    int *d_b;

    // allocate GPU memory
    cudaMalloc((void**)&d_a, SIZE*sizeof(int));
    cudaMalloc((void**)&d_b, NUM_BIN*sizeof(int));

    // transfer the arrays to the GPU
    cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

    // launch the kernel

    // histogram_without_atomic<<<((SIZE+NUM_BIN-1) / NUM_BIN), NUM_BIN>>>(d_b, d_a);
    // histogram_atomic<<<((SIZE+NUM_BIN-1) / NUM_BIN), NUM_BIN>>>(d_b, d_a);
    histogram_shared_memory<<<((SIZE+NUM_BIN-1) / NUM_BIN), NUM_BIN>>>(d_b, d_a);

    // copy back the sum from GPU
    cudaMemcpy(h_b, d_b, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Histogram using 16 bin without shared Memory is: \n");
    for (int i = 0; i < NUM_BIN; i++) {
        printf("bin %d: count %d\n", i, h_b[i]);
    }

    // free GPU memory allocation
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
```

### Getting Started with OpenCV with CUDA Support

#### Reading and displaying an image

```cpp
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    // Read the image
	Mat img = imread("cameraman.tif", 0); // 灰度图
	// Mat img = imread("cameraman.tif", 1); // 彩色图

	// Check for failure in reading an Image
	if (img.empty()) {
	    cout << "Could not open an image" << endl;
		return -1;
	}

	// Name of the window
	String win_name = "My First Opencv Program";

	// Create a window
	namedWindow(win_name);

	// Show our image inside the created window.
	imshow(win_name, img);

	// Wait for any keystroke in the window
	waitKey(0);

	// destory the created window
	destroyWindow(win_name);
    
    return 0;
}
```

#### Create images using OpenCV

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    // Create blank black grayscale Image with size 256x256
	Mat img(256, 256, CV_8UC1, Scalar(0));
	String win_name = "Blank Image";
	namedWindow(win_name);
	imshow(win_name, img);
	waitKey(0);
	destroyWindow(win_name);
    return 0;
}
```

**Drawing shapes on the blank image**

- Drawing a line

```cpp
line(img, Point(0, 0), Point(511, 511), Scalar(0, 255, 0), 7);
```

- Drawing a rectangle

```cpp
rectangle(img, Point(384, 0), Point(510, 128), Scalar(255, 255, 0), 5);
```

- Drawing a circle

```cpp
circle(img, Point(447, 63), 63, Scalar(0, 0, 255), -1);
```

- Drawing an ellipse

```cpp
ellipse(img, Point(256, 256), Point(100, 100), 0, 0, 180, 255, -1);
```

- Writing text on an image

```cpp
putText(img, "OpenCV!", Point(10, 500), FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 255, 255), 5, 8);
```

**example**

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {

    Mat img(512, 512, CV_8UC3, Scalar(0, 0, 0));
	line(img, Point(0, 0), Point(511, 511), Scalar(0, 255, 0), 7);
	rectangle(img, Point(384, 0), Point(510, 128), Scalar(255, 255, 0), 5);
	circle(img, Point(447, 63), 63, Scalar(0, 0, 255), -1);
	ellipse(img, Point(256, 256), Point(100, 100), 0, 0, 180, 255, -1);
	putText(img, "OpenCV!", Point(10, 500), FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 255, 255), 5, 8);
	String win_name = "Shapes on blank Image";
	namedWindow(win_name);
	imshow(win_name, img);
	waitKey(0);
	destroyWindow(win_name);

    return 0;
}
```

#### Saving an image to a file

```cpp
bool flag = imwrite("images/save_image.jpg", img);
```

#### Working with video stored on a computer

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    // open the video file from PC
	VideoCapture cap("images/rhinos.avi");
	double frames_per_second = cap.get(CAP_PROP_FPS);
	double frames_width = cap.get(CAP_PROP_FRAME_WIDTH);
	double frames_height = cap.get(CAP_PROP_FRAME_HEIGHT);

    cout << "Frames per seconds of the video is : " << frames_per_second;
    cout << "Frames width of the video is : " << frames_width;
    cout << "Frames height of the video is : " << frames_height;

	// if not success, exit program
	if (false == cap.isOpened()) {
	    cout << "Cannot open the video file" << endl;
		return -1;
	}

	cout << "Press Q to Quit" << endl;

	String win_name = "First Video";
	namedWindow(win_name);
	while (true) {
	    Mat frame;
		// read a frame
		bool flag = cap.read(frame);

		// Breaking the while loop at the end of the video
		if (false == flag) {
		    break;
		}

		// display the frame
		imshow(win_name, frame);

		// Wait for 100 ms and key 'q' for exit
		if ('q' == waitKey(100)) {
		    break;
		}
	}

	destroyWindow(win_name);

    return 0;
}
```

#### Working with videos from a webcam

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    // open the Webcam
	VideoCapture cap(0);
	// if not success, exit program
	if (false == cap.isOpened()) {
	    cout << "Cannot open Webcam" << endl;
		return -1;
	}

	// get the frames rate of the video from webcam
	double frames_per_second = cap.get(CAP_PROP_FPS);
    cout << "Frames per seconds : " << frames_per_second << endl;
    cout<<"Press Q to Quit" <<endl;
    String win_name = "Webcam Video";
    namedWindow(win_name); //create a window

	while (true) {
        Mat frame;
        bool flag = cap.read(frame); // read a new frame from video
		if (flag == false) {
		    break;
		}
        //show the frame in the created window
        imshow(win_name, frame);
        if (waitKey(1) == 'q') {
		    break;
		}
	}

    return 0;
}
```

#### Saving video to a disk

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    VideoCapture capture(0); // 读入视频
	VideoWriter writer("test.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, Size(640, 480));

	while (capture.isOpened()) {
	    Mat frame;
		capture >> frame; // 读取当前帧
		writer << frame;  // 将读取的帧保存
		imshow("video", frame);
		if (27 == waitKey(10)) { // 如果按下ESC退出
		    break;
		}
	}
}
```

### Basic computer vision applications using the OpenCV CUDA module

#### Introduction to the OpenCV CUDA module

```sh
cmake -D OPENCV_EXTRA_MODULES_PATH=/tensorrt_workspace/opencv_contrib-4.1.2/modules -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/ -D OPENCV_GENERATE_PKGCONFIG=ON -D WITH_GTK=ON -D WITH_CUDA=ON -D BUILD_opencv_cudaoptflow=OFF ..

ldconfig -v
```

#### Arithmetic and logical operation

- Addition of two images

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    // Read Two Images
	cv::Mat h_img1 = cv::imread("cameraman.tif");
	cv::Mat h_img2 = cv::imread("cameraman.tif");
	// Create Memory for storing Image on device
	cv::cuda::GpuMat d_result1, d_img1, d_img2;
    cv::Mat h_result1;

	// Upload Images to device
	d_img1.upload(h_img1);
	d_img2.upload(h_img2);
	
	cv::cuda::add(d_img1, d_img2, d_result1);
	// Download Result back to host
	d_result1.download(h_result1);
	cv::imshow("Image1 ", h_img1);
	cv::imshow("Image2 ", h_img2);
	cv::imshow("Result addition ", h_result1);
	cv::imwrite("result_add.png", h_result1);
	cv::waitKey();
    
    return 0;	

}
```

- Subtracting two images

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    // Read Two Images
	cv::Mat h_img1 = cv::imread("cameraman.tif");
	cv::Mat h_img2 = cv::imread("cameraman.tif");
	// Create Memory for storing Image on device
	cv::cuda::GpuMat d_result1, d_img1, d_img2;
    cv::Mat h_result1;

	// Upload Images to device
	d_img1.upload(h_img1);
	d_img2.upload(h_img2);
	
	// d_result1 = d_img1 - d_img2
	cv::cuda::subtract(d_img1, d_img2, d_result1);
	// Download Result back to host
	d_result1.download(h_result1);
	cv::imshow("Image1 ", h_img1);
	cv::imshow("Image2 ", h_img2);
	cv::imshow("Result addition ", h_result1);
	cv::imwrite("result_add.png", h_result1);
	cv::waitKey();
    
    return 0;	

}
```

- Image blending

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    // Read Two Images
	cv::Mat h_img1 = cv::imread("cameraman.tif");
	cv::Mat h_img2 = cv::imread("cameraman.tif");
	// Create Memory for storing Image on device
	cv::cuda::GpuMat d_result1, d_img1, d_img2;
    cv::Mat h_result1;

	// Upload Images to device
	d_img1.upload(h_img1);
	d_img2.upload(h_img2);

	// result = α * img1 + β * img2 + γ
	cv::cuda::addWeighted(d_img1, 0.7, d_img2, 0.3, 0, d_result1);
	// Download Result back to host
	d_result1.download(h_result1);
	cv::imshow("Image1 ", h_img1);
	cv::imshow("Image2 ", h_img2);
	cv::imshow("Result addition ", h_result1);
	cv::imwrite("result_add.png", h_result1);
	cv::waitKey();
    
    return 0;	

}
```

- Image inversion

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::Mat h_img1 = cv::imread("cameraman.tif");
	// Create Device variables
	cv::cuda::GpuMat d_result1, d_img1;
	cv::Mat h_result1;
	// Upload Image to device
	d_img1.upload(h_img1);

	cv::cuda::bitwise_not(d_img1, d_result1);

	// Download result back to host
	d_result1.download(h_result1);
	cv::imshow("Result inversion ", h_result1);
	cv::imwrite("result_inversion.png", h_result1);
	cv::waitKey();

	return 0;

}
```

#### Change the color space of an image

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("cameraman.tif");
	// Define device variables
	cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_img1;
	// Upload Image to device
	d_img1.upload(h_img1);

	// Convert image to different color spaces
	cv::cuda::cvtColor(d_img1, d_result1, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(d_img1, d_result2, cv::COLOR_BGR2RGB);
	cv::cuda::cvtColor(d_img1, d_result3, cv::COLOR_BGR2HSV);
	cv::cuda::cvtColor(d_img1, d_result4, cv::COLOR_BGR2YCrCb);

	cv::Mat h_result1, h_result2, h_result3, h_result4;
	// Download results back to host
	d_result1.download(h_result1);
	d_result2.download(h_result2);
	d_result3.download(h_result3);
	d_result4.download(h_result4);

	cv::imshow("Result in Gray", h_result1);
	cv::imshow("Result in RGB", h_result2);
	cv::imshow("Result in HSV", h_result3);
	cv::imshow("Result in YCrCb", h_result4);

	cv::waitKey();
    return 0;
}
```

#### Image thresholding

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	// Define device variables
	cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_result5, d_img1;
	// Upload image on device
	d_img1.upload(h_img1);

	// Perform different thresholding techniques on device
	cv::cuda::threshold(d_img1, d_result1, 128.0, 255.0, cv::THRESH_BINARY);
	cv::cuda::threshold(d_img1, d_result2, 128.0, 255.0, cv::THRESH_BINARY_INV);
    cv::cuda::threshold(d_img1, d_result3, 128.0, 255.0, cv::THRESH_TRUNC);
    cv::cuda::threshold(d_img1, d_result4, 128.0, 255.0, cv::THRESH_TOZERO);
    cv::cuda::threshold(d_img1, d_result5, 128.0, 255.0, cv::THRESH_TOZERO_INV);

	cv::Mat h_result1,h_result2,h_result3,h_result4,h_result5;
    
	//Copy results back to host
    d_result1.download(h_result1);
    d_result2.download(h_result2);
    d_result3.download(h_result3);
    d_result4.download(h_result4);
    d_result5.download(h_result5);

	cv::imwrite("Threshhold_binary.jpg", h_result1);
	cv::imwrite("Threshhold_binary_inverse.jpg", h_result2);
	cv::imwrite("Threshhold_truncated.jpg", h_result3);
	cv::imwrite("Threshhold_truncated2zero.jpg", h_result4);
	cv::imwrite("Threshhold_truncated2zero_inverse.jpg", h_result5);
    cv::imshow("Result Threshhold binary ", h_result1);
    cv::imshow("Result Threshhold binary inverse ", h_result2);
    cv::imshow("Result Threshhold truncated ", h_result3);
    cv::imshow("Result Threshhold truncated to zero ", h_result4);
    cv::imshow("Result Threshhold truncated to zero inverse ", h_result5);
    cv::waitKey();
    
	return 0;
}
```

#### Performance comparison of OpenCV applications with and without CUDA support

- without GPU

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	
    cv::Mat src = cv::imread("baboon.jpg", 0);
	cv::Mat result_host1, result_host2, result_host3, result_host4, result_host5;

	// Get initial time in milliseconds
	int64 work_begin = cv::getTickCount();
	cv::threshold(src, result_host1, 128.0, 255.0, cv::THRESH_BINARY);
	cv::threshold(src, result_host2, 128.0, 255.0, cv::THRESH_BINARY_INV);
	cv::threshold(src, result_host3, 128.0, 255.0, cv::THRESH_TRUNC);
	cv::threshold(src, result_host4, 128.0, 255.0, cv::THRESH_TOZERO);
    cv::threshold(src, result_host5, 128.0, 255.0, cv::THRESH_TOZERO_INV);

	// Get time after work has finished
	int64 delta = cv::getTickCount() - work_begin;

	// Frequency of timer
	double freq = cv::getTickFrequency();
	double work_fps = freq / delta;
    std::cout << "Performance of Thresholding on CPU: " << std::endl;
    std::cout << "Time: " << (1/work_fps) << std::endl;
    std::cout << "FPS: " << work_fps << std::endl;
    
	return 0;
}
```

- with GPU

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_result5, d_img1;
	// Measure initial time ticks
	int64 work_begin = cv::getTickCount();
	d_img1.upload(h_img1);
	cv::cuda::threshold(d_img1, d_result1, 128.0, 255.0, cv::THRESH_BINARY);
	cv::cuda::threshold(d_img1, d_result2, 128.0, 255.0, cv::THRESH_BINARY_INV);
	cv::cuda::threshold(d_img1, d_result3, 128.0, 255.0, cv::THRESH_TRUNC);
	cv::cuda::threshold(d_img1, d_result4, 128.0, 255.0, cv::THRESH_TOZERO);
	cv::cuda::threshold(d_img1, d_result5, 128.0, 255.0, cv::THRESH_TOZERO_INV);

	cv::Mat h_result1, h_result2, h_result3, h_result4, h_result5;
	d_result1.download(h_result1);
	d_result2.download(h_result2);
	d_result3.download(h_result3);
	d_result4.download(h_result4);
	d_result5.download(h_result5);
	// Measure difference in time ticks
	int64 delta = cv::getTickCount() - work_begin;
	double freq = cv::getTickFrequency();
	// Measure frames per second
	double work_fps = freq / delta;
	std::cout << "Performance of Thresholding on GPU: " << std::endl;
	std::cout << "Time: " << (1/work_fps) << std::endl;
	std::cout << "FPS: " << work_fps << std::endl;

	return 0;
}
```

### Basic Computer Vision Operations Using OpenCV and CUDA

#### Accessing the individual pixel intensities of an image

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    // Gray Scale Image
	cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::Scalar intensity = h_img1.at<uchar>(cv::Point(100, 50));
	std::cout << "Pixel Intensity of gray scale Image at (100, 50) is: " << intensity.val[0] << endl;

	// Color Image
	cv::Mat h_img2 = cv::imread("baboon.jpg", 1);
	cv::Vec3b intensity1 = h_img1.at<cv::Vec3b>(cv::Point(100, 50));
	std::cout << "Pixel Intensity of color Image at (100, 50) is: " << intensity1 << std::endl;

    return 0;
}
```

#### Histogram calculation and equalization in OpenCV

```cpp
void cv::cuda::calcHist(InputArray src, OutputArray hist);
```

**Histgram equalization**

- gray images

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_result1;
	d_img1.upload(h_img1);
	cv::cuda::equalizeHist(d_img1, d_result1);
	cv::Mat h_result1;
	d_result1.download(h_result1);
	
	cv::imwrite("hist_equal.jpg", h_img1);
	// cv::imshow("Original Image ", h_img1);
	// cv::imshow("Histogram Equalized Image", h_result1);
	// cv::waitKey();
    return 0;
}
```

- color images

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg");
	cv::Mat h_img2, h_result1;
	cvtColor(h_img1, h_img2, cv::COLOR_BGR2HSV);
	// Split the image into 3 channels; H, S and V channels respectively and store it in a vector
	std::vector< cv::Mat > vec_channels;
	cv::split(h_img2, vec_channels);

	// Equalize the histogram of only the V channel
    cv::equalizeHist(vec_channels[2], vec_channels[2]);

	// Merge 3 channels in the vector to form the color image in HSV color space.
	cv::merge(vec_channels, h_img2);

	// Convert the histogram equalized image from HSV to BGR color space again
	cv::cvtColor(h_img2, h_result1, cv::COLOR_HSV2BGR);
	cv::imwrite("Histogram_equal_img.jpg", h_result1);
	// cv::imshow("Original Image ", h_img1);
	// cv::imshow("Histogram Equalized Image", h_result1);
	// cv::waitKey();

    return 0;
}
```

#### Geometric transformation on images

- Image resizing

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_result1, d_result2;
	d_img1.upload(h_img1);

	int width = d_img1.cols;
	int height = d_img1.size().height;
	cv::cuda::resize(d_img1, d_result1, cv::Size(200, 200), cv::INTER_CUBIC);
	cv::cuda::resize(d_img1, d_result2, cv::Size(0.5*width, 0.5*height), cv::INTER_LINEAR);
	cv::Mat h_result1, h_result2;
	d_result1.download(h_result1);
	d_result2.download(h_result2);

	cv::imwrite("Resized.jpg", h_result1);
	cv::imwrite("Resized2.jpg", h_result2);
	/*
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Resized Image", h_result1);
	cv::imshow("Resized Image 2", h_result2);

	cv::waitKey();
	*/
    return 0;
}
```

- Image translation and rotation

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_result1, d_result2;
	d_img1.upload(h_img1);
	int cols = d_img1.cols;
	int rows = d_img1.size().height;
	cout << cols << " " << rows << endl;
	// Translation
	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, 70, 0, 1, 50);
	cv::cuda::warpAffine(d_img1, d_result1, trans_mat, d_img1.size());
	// Rotation
	cv::Point2f pt(d_img1.cols/2., d_img1.rows/2.);
	cv::Mat rot_mat = cv::getRotationMatrix2D(pt, 45, 1.0);
	cv::cuda::warpAffine(d_img1, d_result2, rot_mat, cv::Size(d_img1.cols, d_img1.rows));
	cv::Mat h_result1, h_result2;
	d_result1.download(h_result1);
	d_result2.download(h_result2);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Translated Image", h_result1);
	cv::imshow("Rotated Image", h_result2);
	cv::waitKey();

    return 0;
}
```

#### Filtering operations on images

##### Low pass filtering on an image

- Averaging or box filters

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_result3x3, d_result5x5, d_result7x7;
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filter3x3, filter5x5, filter7x7;

	filter3x3 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
	filter3x3->apply(d_img1, d_result3x3);
	filter5x5 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5));
	filter5x5->apply(d_img1, d_result5x5);
	filter7x7 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(7, 7));
	filter7x7->apply(d_img1, d_result7x7);

	cv::Mat h_result3x3, h_result5x5, h_result7x7;
	d_result3x3.download(h_result3x3);
	d_result5x5.download(h_result5x5);
	d_result7x7.download(h_result7x7);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Blurred with kernel size 3x3", h_result3x3);
	cv::imshow("Blurred with kernel size 5x5", h_result5x5);
	cv::imshow("Blurred with kernel size 7x7", h_result7x7);
	cv::waitKey();

    return 0;
}
```

- Gaussian filters

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_result3x3, d_result5x5, d_result7x7;

	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filter3x3, filter5x5, filter7x7;
	filter3x3 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3), 1);
	filter3x3->apply(d_img1, d_result3x3);
	filter5x5 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1);
	filter5x5->apply(d_img1, d_result5x5);
	filter7x7 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(7, 7), 1);
	filter7x7->apply(d_img1, d_result7x7);

	cv::Mat h_result3x3, h_result5x5, h_result7x7;
	d_result3x3.download(h_result3x3);
	d_result5x5.download(h_result5x5);
	d_result7x7.download(h_result7x7);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Blurred with kernel size 3x3", h_result3x3);
	cv::imshow("Blurred with kernel size 5x5", h_result5x5);
	cv::imshow("Blurred with kernel size 7x7", h_result7x7);
	cv::waitKey();
    return 0;
}
```

- Median filters

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_result3x3, d_result5x5, d_result7x7;

	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filter3x3, filter5x5, filter7x7;
	filter3x3 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3), 1);
	filter3x3->apply(d_img1, d_result3x3);
	filter5x5 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1);
	filter5x5->apply(d_img1, d_result5x5);
	filter7x7 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(7, 7), 1);
	filter7x7->apply(d_img1, d_result7x7);

	cv::Mat h_result3x3, h_result5x5, h_result7x7;
	d_result3x3.download(h_result3x3);
	d_result5x5.download(h_result5x5);
	d_result7x7.download(h_result7x7);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Blurred with kernel size 3x3", h_result3x3);
	cv::imshow("Blurred with kernel size 5x5", h_result5x5);
	cv::imshow("Blurred with kernel size 7x7", h_result7x7);
	cv::waitKey();
    return 0;
}
```

##### High-pass filtering on images

- Sobel filters

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_resultx, d_resulty, d_resultxy;
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filterx, filtery, filterxy;
	filterx = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 0);
	filterx->apply(d_img1, d_resultx);
	filtery = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 0, 1);
	filtery->apply(d_img1, d_resulty);
	cv::cuda::add(d_resultx, d_resulty, d_resultxy);
	cv::Mat h_resultx, h_resulty, h_resultxy;
	d_resultx.download(h_resultx);
	d_resulty.download(h_resulty);
	d_resultxy.download(h_resultxy);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Sobel-x derivative", h_resultx);
	cv::imshow("Sobel-y derivative", h_resulty);
	cv::imshow("Sobel-xy derivative", h_resultxy);
	cv::waitKey();

    return 0;
}
```

- Scharr filters

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_resultx, d_resulty, d_resultxy;
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filterx, filtery;
	filterx = cv::cuda::createScharrFilter(CV_8UC1, CV_8UC1, 1, 0);
	filterx->apply(d_img1, d_resultx);
	filtery = cv::cuda::createScharrFilter(CV_8UC1, CV_8UC1, 0, 1);
	filtery->apply(d_img1, d_resulty);
	cv::cuda::add(d_resultx, d_resulty, d_resultxy);

	cv::Mat h_resultx, h_resulty, h_resultxy;
	d_resultx.download(h_resultx);
	d_resulty.download(h_resulty);
	d_resultxy.download(h_resultxy);

	cv::imshow("Original Image", h_img1);
	cv::imshow("Scharr-x derivative", h_resultx);
	cv::imshow("Scharr-y derivative", h_resulty);
	cv::imshow("Scharr-xy derivative", h_resultxy);
	cv::waitKey();

    return 0;
}
```

- Laplacian filters

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    cv::Mat h_img1 = cv::imread("baboon.jpg", 0);
	cv::cuda::GpuMat d_img1, d_result1, d_result3;
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filter1, filter3;
	filter1 = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, 1);
	filter1->apply(d_img1, d_result1);
	filter3 = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, 3);
	filter3->apply(d_img1, d_result3);
	cv::Mat h_result1, h_result3;
	d_result1.download(h_result1);
	d_result3.download(h_result3);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Laplacian filter 1", h_result1);
	cv::imshow("Laplacian filter 3", h_result3);
	cv::waitKey();

    return 0;
}
```

- **Erosion**: Erosion sets a center pixel to the minumum over all pixels in the neighborhood. The neighborhood is defined by the structuring element, which is a matrix of 1s and 0s. Erosion is used to enlarge holes in the object, shrink the boundary, eliminate the island, and get ride of narrow peninsulas that might exist on the image boundary.
- **Dilation**: Dilation sets a center pixel to the maximum over all pixels in the neighborhood. The dilation increases the size of a white block and reduces the size of the black region. It is used to fill holes in the object and expand the boundary of the object.
- **Opening**: Image opening is basically a combination of erosion. Both operations are performed using the same structuring elements. It is used to smooth the contours of the image, break down narrow bridges and isolate objects that are touching one another. It is used in the analysis of wear particles in engine oils, ink particles in recycled paper, and so on.
- **Closing**: Image closing is defined as dilation followed by erosion. Both operations are performed using the same structuring elements. It is used to fuse narrow breaks and eliminate small holes.

### Object Detection and Tracking Using OpenCV and CUDA

#### Object detection and tracking based on color

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    VideoCapture cap(0); // capture the video from web cam
        // if webcam is not available then exit the program
        if (!cap.isOpened()) {
            cout << "Cannot open the web cam" << endl;
                return -1;
        }

        while (true) {
            Mat frame;
                // read a new frame from webcam
                bool flag = cap.read(frame);
                if (!flag) {
                    cout << "Cannot read a frame from webcam" << endl;
                        break;
                }
            cuda::GpuMat d_frame, d_frame_hsv, d_intermediate, d_result;
            cuda::GpuMat d_frame_shsv[3];
            cuda::GpuMat d_thresc[3];
            Mat h_result;
            d_frame.upload(frame);

                // Transform image to HSV
                cuda::cvtColor(d_frame, d_frame_hsv, COLOR_BGR2HSV);

                // Split HSV 3 channels
                cuda::split(d_frame_hsv, d_frame_shsv);

                // Threshold HSV channels
                cuda::threshold(d_frame_shsv[0], d_thresc[0], 110, 130, THRESH_BINARY);
                cuda::threshold(d_frame_shsv[1], d_thresc[1], 50, 255, THRESH_BINARY);
                cuda::threshold(d_frame_shsv[2], d_thresc[2], 50, 255, THRESH_BINARY);

                // Bitwise AND the channels
                cv::cuda::bitwise_and(d_thresc[0], d_thresc[1], d_intermediate);
        cv::cuda::bitwise_and(d_intermediate, d_thresc[2], d_result);

                d_result.download(h_result);
                imshow("Thresholded Image", h_result);
                imshow("Original", frame);

                if ('q' == waitKey(1)) {
                    break;
                }
        }

    return 0;
}
```

#### Object detection and tracking based on shape

- Canny edge detection

```cpp
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char *argv[]) {
    Mat h_image = imread("baboon.jpg", 0);
        if (h_image.empty()) {
            cout << "can not open image" << endl;
                return -1;
        }
        GpuMat d_edge, d_image;
        Mat h_edge;
        d_image.upload(h_image);
        cv::Ptr<cv::cuda::CannyEdgeDetector> Canny_edge = cv::cuda::createCannyEdgeDetector(2.0, 100.0, 3, false);
        Canny_edge->detect(d_image, d_edge);

        d_edge.download(h_edge);
        imshow("source", h_image);
        imshow("detected edges", h_edge);

        waitKey(0);

    return 0;
}
```

- Straight line detection using Hough transform

```cpp
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char *argv[]) {

    Mat h_image = imread("baboon.jpg", 0);
        if (h_image.empty()) {
            cout << "can not open image" << endl;
        }

        Mat h_edge;
        cv::Canny(h_image, h_edge, 100, 200, 3);

        Mat h_imagec;
        cv::cvtColor(h_edge, h_imagec, COLOR_GRAY2BGR);
        Mat h_imageg = h_imagec.clone();
        vector<Vec4i> h_lines;

        {
            const int64 start = getTickCount();
                HoughLinesP(h_edge, h_lines, 1, CV_PI/180, 50, 60, 5);
                const double time_elapsed = (getTickCount() - start) / getTickFrequency();
        cout << "CPU Time : " << time_elapsed * 1000 << " ms" << endl;
        cout << "CPU FPS : " << (1/time_elapsed) << endl;
        }

        for (size_t i = 0; i < h_lines.size(); ++i) {
            Vec4i line_point = h_lines[i];
                line(h_imagec, Point(line_point[0], line_point[1]), Point(line_point[2], line_point[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }

        GpuMat d_edge, d_lines;
        d_edge.upload(h_edge);
        {
            const int64 start = getTickCount();
                Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI/180.0f), 50, 5);
                hough->detect(d_edge, d_lines);

                const double time_elapsed = (getTickCount() - start) / getTickFrequency();
                cout << "GPU Time: " << time_elapsed * 1000 << "ms" << endl;
                cout << "GPU FPS: " << (1/time_elapsed) << endl;
        }

        vector<Vec4i> lines_g;
        if (!d_lines.empty()) {
            lines_g.resize(d_lines.cols);
                Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_g[0]);
                d_lines.download(h_lines);
        }
        for (size_t i = 0; i < lines_g.size(); ++i) {
        Vec4i line_point = lines_g[i];
                line(h_imageg, Point(line_point[0], line_point[1]), Point(line_point[2], line_point[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }

        // imshow("source", h_image);
        // imshow("detected lines [CPU]", h_imagec);
        // imshow("detected lines [GPU]", h_imageg);
        imwrite("hough_source.png", h_image);
        imwrite("hough_cpu_line.png", h_imagec);
        imwrite("hough_gpu_line.png", h_imageg);

        waitKey(0);
    return 0;
}
```

- Circle detection

```cpp
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char *argv[]) {

    Mat h_image = imread("baboon.jpg", 0);
        if (h_image.empty()) {
            cout << "can not open image" << endl;
        }

        Mat h_edge;
        cv::Canny(h_image, h_edge, 100, 200, 3);

        Mat h_imagec;
        cv::cvtColor(h_edge, h_imagec, COLOR_GRAY2BGR);
        Mat h_imageg = h_imagec.clone();
        vector<Vec4i> h_lines;

        {
            const int64 start = getTickCount();
                HoughLinesP(h_edge, h_lines, 1, CV_PI/180, 50, 60, 5);
                const double time_elapsed = (getTickCount() - start) / getTickFrequency();
        cout << "CPU Time : " << time_elapsed * 1000 << " ms" << endl;
        cout << "CPU FPS : " << (1/time_elapsed) << endl;
        }

        for (size_t i = 0; i < h_lines.size(); ++i) {
            Vec4i line_point = h_lines[i];
                line(h_imagec, Point(line_point[0], line_point[1]), Point(line_point[2], line_point[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }

        GpuMat d_edge, d_lines;
        d_edge.upload(h_edge);
        {
            const int64 start = getTickCount();
                Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI/180.0f), 50, 5);
                hough->detect(d_edge, d_lines);

                const double time_elapsed = (getTickCount() - start) / getTickFrequency();
                cout << "GPU Time: " << time_elapsed * 1000 << "ms" << endl;
                cout << "GPU FPS: " << (1/time_elapsed) << endl;
        }

        vector<Vec4i> lines_g;
        if (!d_lines.empty()) {
            lines_g.resize(d_lines.cols);
                Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_g[0]);
                d_lines.download(h_lines);
        }
        for (size_t i = 0; i < lines_g.size(); ++i) {
        Vec4i line_point = lines_g[i];
                line(h_imageg, Point(line_point[0], line_point[1]), Point(line_point[2], line_point[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }

        // imshow("source", h_image);
        // imshow("detected lines [CPU]", h_imagec);
        // imshow("detected lines [GPU]", h_imageg);
        imwrite("hough_source.png", h_image);
        imwrite("hough_cpu_line.png", h_imagec);
        imwrite("hough_gpu_line.png", h_imageg);

        waitKey(0);
    return 0;
}
```

#### Key-point detectors and descriptors

- Features from Accelerated Segment Test(Fast) feature detector

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    Mat h_image = imread("baboon.jpg", 0);

        // Detect the keypoints using FAST Detector
        cv::Ptr<cv::cuda::FastFeatureDetector> detector = cv::cuda::FastFeatureDetector::create(100, true, 2);
        std::vector<cv::KeyPoint> keypoints;
        cv::cuda::GpuMat d_image;
        d_image.upload(h_image);
        detector->detect(d_image, keypoints);
        cv::drawKeypoints(h_image, keypoints, h_image);
        // show detected keypoints
        imshow("Final Result", h_image);
        waitKey(0);

        return 0;
}
```

- Oriented FAST and Rotated BRIEF (ORB) feature detection

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {

    Mat h_image = imread("baboon.jpg", 0);
        cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create();
        std::vector<cv::KeyPoint> keypoints;
        cv::cuda::GpuMat d_image;
        d_image.upload(h_image);
        detector->detect(d_image, keypoints);
        cv::drawKeypoints(h_image, keypoints, h_image);
        imshow("Final Result", h_image);

        waitKey(0);

    return 0;
}
```

- Speeded up robust feature detection and matching

```cpp
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

using namespace std;
using namespace cv::xfeatures2d;
using namespace cv;

int main(int argc, char *argv[]) {
    
    Mat h_object_image = imread("images/object1.jpg", 0);
        Mat h_scene_image = imread( "images/scene1.jpg", 0);

        cuda::GpuMat d_object_image;
        cuda::GpuMat d_scene_image;

        cuda::GpuMat d_keypoints_scene, d_keypoints_object;
        vector<KeyPoint> h_keypoints_scene, h_keypoints_object;
        cuda::GpuMat d_descriptors_scene, d_descriptors_object;

        d_object_image.upload(h_object_image);
        d_scene_image.upload(h_scene_image);

        cuda::SURF_CUDA surf(100);
        surf(d_object_image, cuda::GpuMat(), d_keypoints_object, d_descriptors_object);
        surf(d_scene_image, cuda::GpuMat(), d_keypoints_scene, d_descriptors_scene);

        Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher();
        vector< vector<DMatch> > d_matches;
        matcher->knnMatch(d_descriptors_object, d_descriptors_scene, d_matches, 2);

        surf.downloadKeypoints(d_keypoints_scene, h_keypoints_scene);
        surf.downloadKeypoints(d_keypoints_object, h_keypoints_object);

        std::vector< DMatch > good_matches;
        for (unsigned int k = 0; k < std::min(h_keypoints_object.size()-1, d_matches.size()); k++) {
            if ((d_matches[k][0].distance < 0.6 * (d_matches[k][1].distance)) && 
                        ((int)d_matches[k].size() <= 2 && (int)d_matches[k].size() > 0)) {
                    good_matches.push_back(d_matches[k][0]);
                }
        }
        std::cout << "size: " << good_matches.size() << endl;
        Mat h_image_result;
        drawMatches(h_object_image, h_keypoints_object, h_scene_image, h_keypoints_scene,
                                good_matches, h_image_result, Scalar::all(-1), Scalar::all(-1),
                                vector<char>(), DrawMatchesFlags::DEFAULT);

        std::vector<Point2f> object;
        std::vector<Point2f> scene;

        for (unsigned int i = 0; i < good_matches.size(); i++) {
            object.push_back(h_keypoints_object[good_matches[i].queryIdx].pt);
                scene.push_back(h_keypoints_scene[good_matches[i].trainIdx].pt);
        }

        Mat Homo = findHomography(object, scene, RANSAC);
        std::vector<Point2f> corners(4);
        std::vector<Point2f> scene_corners(4);
        corners[0] = Point(0, 0);
        corners[1] = Point(h_object_image.cols, 0);
        corners[2] = Point(h_object_image.cols, h_object_image.rows);
        corners[3] = Point(0, h_object_image.rows);
        perspectiveTransform(corners, scene_corners, Homo);
        
        line(h_image_result, scene_corners[0] + Point2f(h_object_image.cols, 0),scene_corners[1] + Point2f(h_object_image.cols, 0),     Scalar(255, 0, 0), 4);
        line(h_image_result, scene_corners[1] + Point2f(h_object_image.cols, 0),scene_corners[2] + Point2f(h_object_image.cols, 0),Scalar(255, 0, 0), 4);
        line(h_image_result, scene_corners[2] + Point2f(h_object_image.cols, 0),scene_corners[3] + Point2f(h_object_image.cols, 0),Scalar(255, 0, 0), 4);
        line(h_image_result, scene_corners[3] + Point2f(h_object_image.cols, 0),scene_corners[0] + Point2f(h_object_image.cols, 0),Scalar(255, 0, 0), 4);

        imshow("Good Matches & Object detection", h_image_result);
        waitKey(0);

    return 0;
}
```

#### Object detection using Haar cascades

- Face detection using Haar cascades

**from picture**

```cpp
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    Mat h_image;
        h_image = imread("images/lena_color_512.tif", 0);
        cout << h_image << endl;
        Ptr<cuda::CascadeClassifier> cascade = cuda::CascadeClassifier::create("haarcascade_frontalface_alt2.xml");
        cuda::GpuMat d_image;
        cuda::GpuMat d_buf;
        d_image.upload(h_image);
        cascade->detectMultiScale(d_image, d_buf);
        std::vector<Rect> detections;
        cascade->convert(d_buf, detections);
        if (detections.empty()) {
            std::cout << "No detection." << std::endl;
        }

        cvtColor(h_image, h_image, COLOR_GRAY2BGR);
        for (unsigned int i = 0; i < detections.size(); ++i) {
            rectangle(h_image, detections[i], Scalar(0, 255, 255), 5);
        }

        imshow("Result image", h_image);
        waitKey(0);

        return 0;
}
```

**from video**

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Can not open video source";
                return -1;
        }
        std::vector<cv::Rect> h_found;
        cv::Ptr<cv::cuda::CascadeClassifier> cascade = 
                cv::cuda::CascadeClassifier::create("haarcascade_frontalface_alt2.xml");
        cv::cuda::GpuMat d_frame, d_gray, d_found;
        while (true) {
            Mat frame;
                if (!cap.read(frame)) {
                    cerr << "Can not read frame from webcam";
                        return -1;
                }
                d_frame.upload(frame);
                cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);
                cascade->detectMultiScale(d_gray, d_found);
                cascade->convert(d_found, h_found);
                
                for (unsigned int i = 0; i < h_found.size(); ++i) {
                    rectangle(frame, h_found[i], Scalar(0, 255, 255), 5);
                }

                imshow("Result", frame);
                if ('q' == waitKey(1)) {
                    break;
                }
        }
    return 0;
}
```

- Eye detection using Haar cascades

```cpp
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    Mat h_image;
        h_image = imread("images/lena_color_512.tif", 0);
        Ptr<cuda::CascadeClassifier> cascade = cuda::CascadeClassifier::create("haarcascade_eye.xml");

        cuda::GpuMat d_image;
        cuda::GpuMat d_buf;
        d_image.upload(h_image);
        cascade->setScaleFactor(1.02);
        cascade->detectMultiScale(d_image, d_buf);
        std::vector<Rect> detections;
        cascade->convert(d_buf, detections);

        if (detections.empty()) {
            std::cout << "No detection." << endl;
                cvtColor(h_image, h_image, COLOR_GRAY2BGR);
                for (unsigned int i = 0; i < detections.size(); ++i) {
                    rectangle(h_image, detections[i], Scalar(0, 255, 255), 5);
                }

        }

        imshow("Result image", h_image);
        waitKey(0);
    
        return 0;
}
```

#### Object tracking using background subtraction

- Mixture of Gaussian (MOG) method

```cpp
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char *argv[]) {

    VideoCapture cap("abc.avi");
        if (!cap.isOpened()) {
            cerr << "can not open camera or video file" << endl;
                return -1;
        }
        Mat frame;
        cap.read(frame);
        GpuMat d_frame;
        d_frame.upload(frame);
        Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG();
        GpuMat d_fgmask, d_fgimage, d_bgimage;
        Mat h_fgmask, h_fgimage, h_bgimage;
        mog->apply(d_frame, d_fgmask, 0.01);
        while (true) {
            cap.read(frame);
                if (frame.empty()) {
                    break;
                }
                d_frame.upload(frame);
                int64 start = cv::getTickCount();
                mog->apply(d_frame, d_fgmask, 0.01);
                mog->getBackgroundImage(d_bgimage);
                double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
                std::cout << "FPS: " << fps << std::endl;
                d_fgimage.create(d_frame.size(), d_frame.type());
                d_fgimage.setTo(Scalar::all(0));
                d_frame.copyTo(d_fgimage, d_fgmask);
                d_fgmask.download(h_fgmask);
                d_fgimage.download(h_fgimage);

                d_bgimage.download(h_bgimage);
                imshow("image", frame);
                imshow("foreground mask", h_fgmask);
                imshow("foreground image", h_fgimage);
                imshow("mean background image", h_bgimage);
                if ('q' == waitKey(1)) {
                    break;
                }
        }

        return 0;
}
```

- GMG for background subtraction

```cpp
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char *argv[]) {

    VideoCapture cap("abc.avi");
        if (!cap.isOpened()) {
            cerr << "can not open video file" << endl;
                return -1;
        }

        Mat frame;
        cap.read(frame);
        GpuMat d_frame;
        d_frame.upload(frame);
        Ptr<BackgroundSubtractor> gmg = cuda::createBackgroundSubtractorGMG(40);
        GpuMat d_fgmask, d_fgimage, d_bgimage;
        Mat h_fgmask, h_fgimage, h_bgimage;
        gmg->apply(d_frame, d_fgmask);
        while (true) {
            cap.read(frame);
                if (frame.empty()) {
                    break;
                }
                d_frame.upload(frame);
                int64 start = cv::getTickCount();
                gmg->apply(d_frame, d_fgmask, 0.01);
                double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
                std::cout << "FPS: " << fps << std::endl;
                d_fgimage.create(d_frame.size(), d_frame.type());
        d_fgimage.setTo(Scalar::all(0));
                d_frame.copyTo(d_fgimage, d_fgmask);
                d_fgmask.download(h_fgmask);
                d_fgimage.download(h_fgimage);
                imshow("image", frame);
                imshow("foreground mask", h_fgmask);
                imshow("foreground image", h_fgimage);
                if ('q' == waitKey(30)) {
                    break;
                }
        }
    return 0;
}
```

### Working with PyCUDA

#### Writing the first program in PyCUDA

```py
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
#include <stdio.h>
__global__ void myfirst_kernel() {
    printf("Hello, PyCUDA!!!");
}
""")

function = mod.get_function("myfirst_kernel")
function(block=(1,1,1))
```

#### Accessing GPU device properties from PyCUDA program

```py
import pycuda.driver as drv
import pycuda.autoinit
drv.init()
print("%d device(s) found." % drv.Device.count());
for i in range(drv.Device.count()):
    dev = drv.Device(i)
    print("Device #%d: %s" % (i, dev.name()))
    print("Compute Capability: %d.%d" % dev.compute_capability())
    print(" Total Memory: %s GB" % (dev.total_memory()//(1024*1024*1024)))

    attributes = [(str(prop), value) for prop, value in list(dev.get_attributes().items())]
    attributes.sort()
    n = 0;
    for prop, value in attributes:
        print(" %s: %s " % (prop, value), end=" ")
        n = n + 1
        if (n % 2 == 0):
            print(" ")
```

#### Thread and block execution in PyCUDA

```py
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>
    __global__ void myfirst_kernel() {
        printf("I am in block no: %d \\n", blockIdx.x);
    }
""")

function = mod.get_function("myfirst_kernel")
function(grid=(4, 1), block=(1, 1, 1))
```

#### Basic programming concepts in PyCUDA

- Adding two numbers in PyCUDA

```py
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
    __global__ void add_num(float *d_result, float *d_a, float *d_b) {
        const int i = threadIdx.x;
        d_result[i] = d_a[i] + d_b[i];
    }        
""")

if __name__ == "__main__":
    add_num = mod.get_function("add_num")

    h_a = np.random.randn(1).astype(np.float32)
    h_b = np.random.randn(1).astype(np.float32)
    h_result = np.zeros_like(h_a)

    d_a = drv.mem_alloc(h_a.nbytes)
    d_b = drv.mem_alloc(h_b.nbytes)
    d_result = drv.mem_alloc(h_result.nbytes)

    drv.memcpy_htod(d_a, h_a)
    drv.memcpy_htod(d_b, h_b)
    
    add_num(d_result, d_a, d_b, block=(1, 1, 1), grid=(1, 1))
    drv.memcpy_dtoh(h_result, d_result)
    
    print(h_a, " + ", h_b, " = ", h_result)
```

- Simplifying the addition program using driver class

```py
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
N = 10
from pycuda.compiler import SourceModule
mod = SourceModule("""
   __global__ void add_num(float *d_result, float *d_a, float *d_b) {
       const int i = threadIdx.x;
       d_result[i] = d_a[i] + d_b[i];
   }        
""")

add_num = mod.get_function("add_num")
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
h_result = np.zeros_like(h_a)
add_num(
    drv.Out(h_result), drv.In(h_a), drv.In(h_b),
    block=(N, 1, 1), grid=(1, 1)
)
print("Addition on GPU:")
for i in range(0, N):
    print(h_a[i], "+", h_b[i], "=", h_result[i])
```

#### Measuring performance of PyCUDA programs using CUDA events

```py
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
import math
N = 1000000

from pycuda.compiler import SourceModule
mod = SourceModule("""
    __global__ void add_num(float *d_result, float *d_a, float *d_b, int N) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < N) {
            d_result[tid] = d_a[tid] + d_b[tid];
            tid += blockDim.x * gridDim.x;
        }
    }        
""")

start = drv.Event()
end = drv.Event()
add_num = mod.get_function("add_num")
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
h_result = np.zeros_like(h_a)
h_result1 = np.zeros_like(h_a)

n_blocks = math.ceil((N/1024)+1)
start.record()
add_num(drv.Out(h_result), drv.In(h_a), drv.In(h_b), np.uint32(N), block=(1024, 1, 1), grid=(n_blocks, 1))
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Addition of %d element of GPU"%N)
print("%fs" % (secs))
start = time.time()
for i in range(0,N):
    h_result1[i] = h_a[i] +h_b[i]
    end = time.time()
print("Addition of %d element of CPU"%N)
print(end-start,"s")
root@localhost:/workspace/HandOnGPUAccCV/PyCUDA_tutorial# cat 06add_number.py 
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
import math
N = 1000000

from pycuda.compiler import SourceModule
mod = SourceModule("""
    __global__ void add_num(float *d_result, float *d_a, float *d_b, int N) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < N) {
            d_result[tid] = d_a[tid] + d_b[tid];
            tid += blockDim.x * gridDim.x;
        }
    }        
""")

start = drv.Event()
end = drv.Event()
add_num = mod.get_function("add_num")
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
h_result = np.zeros_like(h_a)
h_result1 = np.zeros_like(h_a)

n_blocks = math.ceil((N/1024)+1)
start.record()
add_num(drv.Out(h_result), drv.In(h_a), drv.In(h_b), np.uint32(N), block=(1024, 1, 1), grid=(n_blocks, 1))
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Addition of %d element of GPU"%N)
print("%fs" % (secs))
start = time.time()
for i in range(0,N):
    h_result1[i] = h_a[i] +h_b[i]
    end = time.time()
print("Addition of %d element of CPU"%N)
print(end-start,"s")
```

#### Complex programs in PyCUDA

- Element-wise squaring of a matrix in PyCUDA

```py
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void square(float *d_a) {
    int idx = threadIdx.x + threadIdx.y * 5;
    d_a[idx] = d_a[idx] * d_a[idx];
}
""")

start = drv.Event()
end = drv.Event()
h_a = np.random.randint(1, 5, (5, 5))
h_a = h_a.astype(np.float32)
h_b = h_a.copy()
start.record()
d_a = drv.mem_alloc(h_a.size * h_a.dtype.itemsize)
drv.memcpy_htod(d_a, h_a)
square = mod.get_function("square")
square(d_a, block=(5, 5, 1), grid=(1, 1), shared=0)
h_result = np.empty_like(h_a)
drv.memcpy_dtoh(h_result, d_a)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU without inout")
print("%fs" % (secs))
print("original array:")
print(h_a)
print("Square with kernel:")
print(h_result)

#----Using inout functionality of driver class ---------------------------
start.record()
start.synchronize()
square(drv.InOut(h_a), block=(5, 5, 1))
end.record()
end.synchronize()

print("Square with InOut:")
print(h_a)
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU with inout")
print("%fs" % (secs))

#----Using gpuarray class ---------------------------------------#
import pycuda.gpuarray as gpuarray
start.record()
start.synchronize()
h_b = np.random.randint(1, 5, (5, 5))
d_b = gpuarray.to_gpu(h_b.astype(np.float32))
h_result = (d_b**2).get()
end.record()
end.synchronize()
print("original array:")
print(h_b)
print("Squared with gpuarray:")
print(h_result)
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU with gpuarray")
print("%fs" % (secs))
```

- Dot product using GPU Array

```py
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
import time
import pycuda.autoinit
n = 100
h_a = np.float32(np.random.randint(1, 5, (1, n)))
h_b = np.float32(np.random.randint(1, 5, (1, n)))

start = time.time()
h_result = np.sum(h_a * h_b)
#print(numpy.dot(a,b))
end=time.time()-start
print("Answer of Dot Product using numpy")
print(h_result)
print("Time taken for Dot Product using numpy")
print(end,"s")

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)

start1 = drv.Event()
end1 = drv.Event()
start1.record()

d_result = gpuarray.dot(d_a, d_b)
end1.record()
end1.synchronize()
secs = start1.time_till(end1)*1e-3
print("Answer of Dot Product on GPU")
print(d_result.get())
print("Time taken for Dot Product on GPU")
print("%fs" % (secs))
if (d_result.get() == h_result):
    print("The computed dot product is correct")
```

- Matrix multiplication

```py
import numpy as np
from pycuda import driver, gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
MATRIX_SIZE = 3

matrix_mul_kernel = """
    __global__ void Matrix_Mul_Kernel(float *d_a, float *d_b, float *d_c) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        float value = 0;

        for (int i = 0; i < %(MATRIX_SIZE)s; ++i) {
            float d_a_element = d_a[ty * %(MATRIX_SIZE)s + i];
            float d_b_element = d_b[i * %(MATRIX_SIZE)s + tx];
            value += d_a_element * d_b_element;
        }

        d_c[ty * %(MATRIX_SIZE)s + tx] = value;
    }
"""

matrix_mul = matrix_mul_kernel % {'MATRIX_SIZE': MATRIX_SIZE}
mod = SourceModule(matrix_mul)

h_a = np.random.randint(1, 5, (MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)
h_b = np.random.randint(1, 5, (MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)
d_c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

matrixmul = mod.get_function("Matrix_Mul_Kernel")

matrixmul(d_a, d_b, d_c_gpu, block=(MATRIX_SIZE, MATRIX_SIZE, 1),)

print("*" * 100)
print("Matrix A:")
print(d_a.get())

print("*" * 100)
print("Matrix B:")
print(d_b.get())

print("*" * 100)
print("Matrix C:")
print(d_c_gpu.get())

# compute on the CPU to verify GPU computation
h_c_cpu = np.dot(h_a, h_b).astype(np.float32)
print("*" * 100)
print("Matrix h_c_cpu:")
print(h_c_cpu)

if (h_c_cpu == d_c_gpu.get()).all():
    print("\n\nThe computed matrix multiplication is correct")
else:
    print("\n\nThe computed matrix multiplication is wrong")
```

#### Advanced kernel functions in PyCUDA

- Element-wise kernel in PyCUDA

```py
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand

add = ElementwiseKernel(
    "float *d_a, float *d_b, float *d_c",
    "d_c[i] = d_a[i] + d_b[i]",
    "add")

n = 10000000
d_a = curand(n)
d_b = curand(n)
d_c = gpuarray.empty_like(d_a)
start = drv.Event()
end = drv.Event()
start.record()
add(d_a, d_b, d_c)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Addition of %d element of GPU" % n)
print("%fs" % (secs))
# Check the result
if (d_a + d_b) == d_c:
    print("The sum computed on GPU is correct")
```

- Reduction kernel

```py
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
from pycuda.reduction import ReductionKernel
import pycuda.autoinit

n = 5
start = drv.Event()
end = drv.Event()
start.record()
d_a = gpuarray.arange(n, dtype=np.uint32)
d_b = gpuarray.arange(n, dtype=np.uint32)
kernel = ReductionKernel(np.uint32, neutral="0", reduce_expr="a+b", map_expr="d_a[i]*d_b[i]", arguments="int *d_a, int *d_b")
d_result = kernel(d_a, d_b).get()
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Vector A")
print(d_a)
print("Vector B")
print(d_b)
print("The computed dot product using reduction:")
print(d_result)
print("Dot Product on GPU")
print("%fs" % (secs))
```

- Scan kernel

```py
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
from pycuda.scan import InclusiveScanKernel
import pycuda.autoinit
n = 10
start = drv.Event()
end = drv.Event()
start.record()
kernel = InclusiveScanKernel(np.uint32, "a+b")
h_a = np.random.randint(1, 10, n).astype(np.int32)
d_a = gpuarray.to_gpu(h_a)
kernel(d_a)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
assert((d_a.get() == np.cumsum(h_a,axis=0)).all())
print("The input data:")
print(h_a)
print("The computed cumulative sum using Scan:")
print(d_a.get())
print("Cumulative Sum on GPU")
print("%fs" % (secs))
```

### Basic Computer Vision Applications Using PyCUDA

#### Histogram calculation in PyCUDA

- Using atomic operations

```py
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void atomic_hist(int *d_b, int *d_a, int SIZE) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];
    if (tid < SIZE) {
        atomicAdd(&(d_b[item]), 1);
    }
}
""")

atomic_hist = mod.get_function("atomic_hist")
# print(atomic_hist)
import cv2
h_img = cv2.imread("cameraman.tif", 0)
h_a = h_img.flatten()
print(len(h_a))
h_a = h_a.astype(np.int)
print(h_a)
h_result = np.zeros(256).astype(np.int)
print(h_result)
SIZE = h_img.size
NUM_BIN = 256
n_threads = int(np.ceil((SIZE+NUM_BIN-1) / NUM_BIN))

start = drv.Event()
end = drv.Event()
start.record()
atomic_hist(
    drv.Out(h_result), drv.In(h_a), np.uint32(SIZE),
    block=(n_threads, 1, 1), grid=(NUM_BIN, 1)
)

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Time for Calculating Histogram on GPU without shared memory")
print("%fs" % (secs))
plt.stem(h_result, use_line_collection=True)
plt.xlim([0, 256])
plt.title("Histogram on GPU")
start = cv2.getTickCount()
hist = cv2.calcHist([h_img], [0], None, [256], [0, 256])
end = cv2.getTickCount()
time = (end - start) / cv2.getTickFrequency()
print("Time for Calculating Histogram using OpenCV")
print("%fs" % (secs))
```

- Using shared memory

```py
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
mod1 = SourceModule("""
__global__ void atomic_hist(int *d_b, int *d_a, int SIZE) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;
    __shared__ int cache[256];
    cache[threadIdx.x] = 0;
    __syncthreads();

    while (tid < SIZE) {
        atomicAdd(&(cache[d_a[tid]]), 1);
        tid += offset;
    }

    __syncthreads();
    atomicAdd(&(d_b[threadIdx.x]), cache[threadIdx.x]);
}        
""")

atomic_hist = mod1.get_function("atomic_hist")
import cv2
h_img = cv2.imread("cameraman.tif", 0)

h_a = h_img.flatten()
h_a = h_a.astype(np.int)
h_result = np.zeros(256).astype(np.int)
SIZE = h_img.size
NUM_BIN = 256
n_threads = int(np.ceil((SIZE+NUM_BIN-1) / NUM_BIN))
start = drv.Event()
end = drv.Event()
start.record()
atomic_hist(drv.Out(h_result),
            drv.In(h_a),
            np.uint32(SIZE),
            block=(n_threads, 1, 1),
            grid=(NUM_BIN, 1),
            shared=256*4)

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Time for Calculating Histogram on GPU with shared memory")
print("%fs" % (secs))
plt.stem(h_result, use_line_collection=True)
plt.xlim([0, 256])
plt.title("Histogram on GPU")

start = cv2.getTickCount()
hist = cv2.calcHist([h_img], [0], None, [256], [0, 256])
end = cv2.getTickCount()
time = (end - start) / cv2.getTickFrequency()
print("Time for Calculating Histogram using OpenCV")
print("%fs" % (time))
```

#### Basic computer vision operations using PyCUDA

- Color space conversion in PyCUDA

**BGR to gray conversion on an image**

```py
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import pycuda.autoinit

mod = SourceModule("""
#include <stdio.h>
#define INDEX(a, b) a*256+b

__global__ void bgr2gray(float *d_result, float *b_img, float *g_img, float *r_img) {
    unsigned int idx = threadIdx.x + (blockIdx.x*(blockDim.x * blockDim.y));
    unsigned int a = idx/256;
    unsigned int b = idx%256;
    d_result[INDEX(a, b)] = 0.299 * r_img[INDEX(a, b)] + 0.587 * g_img[INDEX(a, b)] + 0.114 * b_img[INDEX(a, b)];
}
""")

h_img = cv2.imread('lena_color.tif', 1)
h_gray = cv2.cvtColor(h_img, cv2.COLOR_BGR2GRAY)
b_img = h_img[:, :, 0].reshape(65536).astype(np.float32)
g_img = h_img[:, :, 1].reshape(65536).astype(np.float32)
r_img = h_img[:, :, 2].reshape(65536).astype(np.float32)
h_result = r_img
bgr2gray = mod.get_function("bgr2gray")
bgr2gray(drv.Out(h_result), drv.In(b_img), drv.In(g_img), drv.In(r_img), block=(1024, 1, 1), grid=(64, 1, 1))

h_result = np.reshape(h_result, (256, 256)).astype(np.uint8)
# cv2.imshow("Grayscale Image",h_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("GrayImage.jpg", h_result)
```

**BGR to gray conversion on a webcam video**

```py
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import pycuda.autoinit

mod = SourceModule("""
    #include <stdio.h>
    #define INDEX(a, b) a * 256 + b

    __global__ void bgr2gray(float *d_result, float *b_img, float *g_img, float *r_img) {
        unsigned int idx = threadIdx.x + (blockIdx.x*(blockDim.x*blockDim.y));
        unsigned int a = idx/256;
        unsigned int b = idx%256;
        d_result[INDEX(a, b)] = (0.299*r_img[INDEX(a, b)]+0.587*g_img[INDEX(a, b)]+0.114*b_img[INDEX(a, b)]);
    }
""")

cap = cv2.VideoCapture(0)
bgr2gray = mod.get_function("bgr2gray")
while (True):
    # Capture frame-by-frame
    ret, h_img = cap.read()
    h_img = cv2.resize(h_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    b_img = h_img[:, :, 0].reshape(65536).astype(np.float32)
    g_img = h_img[:, :, 1].reshape(65536).astype(np.float32)
    r_img = h_img[:, :, 2].reshape(65536).astype(np.float32)
    h_result = r_img

    bgr2gray(drv.Out(h_result), drv.In(b_img), drv.In(g_img), drv.In(r_img), block=(1024, 1, 1), grid=(64, 1, 1))

    h_result = np.reshape(h_result, (256, 256)).astype(np.uint8)
    cv2.imshow("Grayscale Image", h_result)

    # Display the resulting frame
    cv2.imshow("Original frame", h_img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```

- Image addition in PyCUDA

```py
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import pycuda.autoinit

mod = SourceModule("""
__global__ void add_num(float *d_result, float *d_a, float *d_b, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        d_result[tid] = d_a[tid] + d_b[tid];
        if (d_result[tid] > 255) {
            d_result[tid] = 255;
        }
        tid += blockDim.x * gridDim.x;
    }
}
""")

img1 = cv2.imread("cameraman.tif", 0)
img2 = cv2.imread("circles.png", 0)

# print a
h_img1 = img1.reshape(65536).astype(np.float32)
h_img2 = img2.reshape(65536).astype(np.float32)
N = h_img1.size
h_result = h_img1
add_img = mod.get_function("add_num")
add_img(drv.Out(h_result), drv.In(h_img1), drv.In(h_img2), np.uint32(N), block=(1024, 1, 1), grid=(64, 1, 1))

h_result = np.reshape(h_result, (256, 256)).astype(np.uint8)
# cv2.imshow("Image after addition", h_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("add.jpg", h_result)
```

- Image inversion in PyCUDA using gpuarray

```py
import pycuda.driver as drv
import numpy as np
import cv2
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

img = cv2.imread("circles.png", 0)

# print a
h_img = img.reshape(65536).astype(np.float32)
d_img = gpuarray.to_gpu(h_img)
d_result = 255 - d_img
h_result = d_result.get()
h_result = np.reshape(h_result, (256, 256)).astype(np.uint8)
# cv2.imshow("Image after Inversion", h_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("inversion.jpg", h_result)
```

