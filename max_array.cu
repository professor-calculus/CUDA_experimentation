#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <algorithm>

// Thread block size
#define BLOCK_SIZE 1024

//  Size of Array
// #define SOA 67107840
// #define SOA 2147483647
#define SOA 1147483647
// #define SOA 8193

// Allocates an array with random integer entries.
void randomInit(int* data, unsigned long long int size)
{
	srand( time(0) );
	for (unsigned long long int i = 0; i < size; ++i)
	{
		data[i] = rand() & INT_MAX;
		//std::cout << data[i] << "\n";
	}
}

__global__ void ReductionMax2(int *input, int *results, unsigned long long int n)    //take thread divergence into account
{
	extern __shared__ int sdata[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tx = threadIdx.x;
	 //load input into __shared__ memory
	if(i < n)
	{
		sdata[tx] = input[i];
	}
	else
	{
		sdata[tx] = INT_MIN;
	}

	__syncthreads();

	// block-wide reduction
	for(unsigned int offset = 1; offset < blockDim.x; offset <<= 1)
	{
		int index = 2 * offset * tx;
		if(index < blockDim.x)
	        {
			// sdata[index] += sdata[index + offset];
			if(sdata[index] < sdata[index + offset]) sdata[index] = sdata[index + offset];
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

// Reduce function wrapper
	void reduce(int* d_a, int* d_b)
	{
		unsigned long long int arraySize = SOA;
		unsigned long long int numBlocks = 1 + ((SOA - 1) / BLOCK_SIZE);

		int* device_intermediate;
		cudaMalloc(&device_intermediate, sizeof(int)*numBlocks);
  		cudaMemset(device_intermediate, 0, sizeof(int)*numBlocks);

		int i=1;

		do
		{
			std::cout << "GPU Iteration " << i << std::endl;
			i++;

			//setup execution parameters
                	dim3 block(BLOCK_SIZE);
	                dim3 grid(numBlocks);

        	        //execute the kernel
	                ReductionMax2<<<grid, block, BLOCK_SIZE*sizeof(int)>>>(d_a,device_intermediate,arraySize);
			arraySize = 1 + ((arraySize - 1) / BLOCK_SIZE);

			// device_in to device_intermediate
			cudaMemcpy(d_a, device_intermediate, sizeof(int)*numBlocks, cudaMemcpyDeviceToDevice);

			// Update required number of blocks
			numBlocks = 1 + ((numBlocks - 1) / BLOCK_SIZE);

			cudaFree(device_intermediate);
			cudaMalloc(&device_intermediate, sizeof(int)*numBlocks);
		}
		while(arraySize > BLOCK_SIZE);

		// Now compute the rest
		ReductionMax2<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(d_a,d_b,arraySize);
	}


// get global max element via per-block reductions
	int main(int argc, char **argv)
	{
		// Introduce program
		std::cout << "Finding max element of an array on a GPU" << std::endl;

		// show memory usage of GPU
        	size_t free_byte ;
	        size_t total_byte ;
		cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        	if ( cudaSuccess != cuda_status )
		{
            		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		        return 1;
		}

	        double free_db = (double)free_byte ;
        	double total_db = (double)total_byte ;
	        double used_db = total_db - free_db ;

        	std::cout << "GPU memory usage: used = " << used_db/1024.0/1024.0 << "MB, free = " <<
			free_db/1024.0/1024.0 << "MB, total = " << total_db/1024.0/1024.0 << " MB" << std::endl;



		// initial num of blocks
		unsigned long long int num_blocks = 1 + ((SOA - 1) / BLOCK_SIZE);
		std::cout << num_blocks << " blocks initially" << std::endl;

		//allocate host memory for array a
		unsigned long long int mem_size_a = sizeof(int) * SOA;
		if(mem_size_a > free_db)
		{
			std::cout << "Error: Not enough available GPU memory!" << std::endl;
			return 1;
		}

		std::cout << mem_size_a/1024.0/1024.0 << "MB requested" << std::endl;
		int* h_a = (int*)malloc(mem_size_a);

		//allocate device memory
		int* d_a;
		cudaMalloc((void**) &d_a, mem_size_a);

		randomInit(h_a,SOA);

		//copy host memory to device
		cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice);

		//allocate device memory for temporary results
		int mem_size_b = sizeof(int) * 1;
		int* d_b;
		cudaMalloc((void**) &d_b, mem_size_b);
		int h_b;

		// Run our kernel wrapper
		reduce(d_a, d_b);

		//copy final result from device to host
		cudaMemcpy(&h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);

                std::cout << "GPU max: " << h_b << "\n";

		int max = *std::max_element(h_a, h_a + SOA);

		std::cout << "Old-fashioned way: " << max << "\n";

		//clean up memory
		free(h_a);
		cudaFree(d_a);
		cudaFree(d_b);

		cudaThreadExit();

}
