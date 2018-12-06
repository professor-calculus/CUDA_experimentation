#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>

// Thread block size
#define BLOCK_SIZE 1024

//  Size of Array
#define SOA 1025

// Allocates an array with random integer entries.
void randomInit(int* data, int size)
{
	srand( time(0) );
	for (int i = 0; i < size; ++i)
	{
		data[i] = rand() & 10;
		//std::cout << data[i] << "\n";
	}
}

__global__ void ReductionMax2(int *input, int *results, int n)    //take thread divergence into account
{
	extern __shared__ int sdata[BLOCK_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tx = threadIdx.x;
	 //load input into __shared__ memory
	if(i < n)
	{
		sdata[tx] = input[i];
	}
	else
	{
		sdata[tx] = 0;
	}

	__syncthreads();

	// block-wide reduction
	for(unsigned int offset = 1; offset < blockDim.x; offset <<= 1)
	{
		int index = 2 * offset * tx;
		if(index < blockDim.x)
	        {
			sdata[index] += sdata[index + offset];
		}
		__syncthreads();
	}

	// finally, thread 0 writes the result
	if(threadIdx.x == 0)
	{
		// the result is per-block
		results[blockIdx.x] = sdata[0];
	}
}


// get global max element via per-block reductions
	int main(int argc, char **argv)
	{
		int num_blocks = 1 + ((SOA - 1) / BLOCK_SIZE);
		std::cout << num_blocks << " blocks" << std::endl;

		//allocate host memory for array a
		unsigned int mem_size_a = sizeof(int) * SOA;
		int* h_a = (int*)malloc(mem_size_a);

		//initialize host memory
		randomInit(h_a,SOA);

		//allocate device memory
		int* d_a;
		cudaMalloc((void**) &d_a, mem_size_a);

		randomInit(h_a,SOA);

		//copy host memory to device
		cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice);

		//allocate device memory for temporary results
		unsigned int mem_size_b = sizeof(int) * num_blocks;
		int* d_b;
		cudaMalloc((void**) &d_b, mem_size_b);
		int* h_b = (int*)malloc(mem_size_b);

		//allocate device memory for final result
		unsigned int mem_size_c = sizeof(int) * num_blocks;
		int* d_c;
		cudaMalloc((void**) &d_c, mem_size_c);

		//setup execution parameters
		dim3 block(BLOCK_SIZE);
		dim3 grid(num_blocks);

		//execute the kernel
		//first reduce per-block partial maxs
		ReductionMax2<<<grid, block>>>(d_a,d_b,SOA);

		// Copy partial sums to host
		cudaMemcpy(h_b, d_b, mem_size_b, cudaMemcpyDeviceToHost);

		//then reduce partial maxs to a final max
		ReductionMax2<<<grid, block>>>(d_b,d_c,num_blocks);

	       	// allocate host memory for the result
		int* h_c = (int*)malloc(mem_size_c);

		//copy final result from device to host
		cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost);

		int sum_partials = 0;
		for(int i=0; i<num_blocks; i++)
		{
			sum_partials += h_b[i];
		}

		std::cout << "Sum of partial sums from GPU is: " << sum_partials << std::endl;

                std::cout << "GPU sum: " << h_c[0] << "\n";

		int tot = 0;
		for(int i=0; i<SOA; i++)
		{
			tot += h_a[i];
		}
		std::cout << "Old-fashioned way: " << tot << "\n";

		//clean up memory
		free(h_a);
		free(h_c);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);

		cudaThreadExit();

}
