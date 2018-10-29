//nvcc -o test histogram_eq_gpu.cu -std=c++11 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
#include <iostream>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"
#include <cuda_runtime.h>

using namespace std;


//Adrian Biller A01018940
//historgram equalizer




__global__ void create_image_histogram(char *input, int* histogram, int nx, int ny){


	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	if(idx < nx && idx < iy){
			histogram[(int)input[idx]] ++;
	}


}


__global__ void normalize_histogram( char *input,int *histogram, int *normalized_histogram, int nx, int ny){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	int accumulated = 0;
	for(int x = 0; x <= idx;  x ++){
		accumulated += histogram[x];
	}

	if(idx < 256){
		normalized_histogram[idx] = accumulated * 255 / (nx*ny);
	}
}

__global__ void contrast_image(char *input, char *output, int* normalized_histogram, int nx, int ny){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	output[idx] = normalized_histogram[ (int)input[idx] ];
}


int main(int argc, char *argv[])
{
	string imagePath;

	// checking image path
	if(argc < 2)
		imagePath = "Images/dog2.jpeg";
	else
		imagePath = argv[1];

	// read color image
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

//converting image to grayscale
	cv::Mat grayscale_input;
  cvtColor(input, grayscale_input, cv::COLOR_BGR2GRAY);

  //creating output image
  cv::Mat output(grayscale_input.rows, grayscale_input.cols, grayscale_input.type());

  //changing contrastof output image
	// image_histogram_equalizer(grayscale_input, output);

 //declaring histogram arrays
	int histogram[256] = {0};
	int normalized_histogram[256] = {0};

	//CUDA Kernel
	int dev = 0;
	cudaDeviceProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	SAFE_CALL(cudaSetDevice(dev), "Error setting device");

	int nx = grayscale_input.cols;
	int ny = grayscale_input.rows;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	char *d_input, *d_output;
	int *d_histogram, *d_normalized_histogram;
	int rBytes  = 256 * sizeof(int);
	SAFE_CALL(cudaMalloc((void **)&d_input, nBytes), "Error allocating input image");
	SAFE_CALL(cudaMalloc((void **)&d_output, nBytes), "Error allocating output image");
	SAFE_CALL(cudaMalloc((void **)&d_histogram, rBytes), "Error allocating histogram");
	SAFE_CALL(cudaMalloc((void **)&d_normalized_histogram, rBytes), "Error allocating normalized histogram");


	SAFE_CALL(cudaMemcpy(d_input, grayscale_input.ptr(), nBytes, cudaMemcpyHostToDevice), "Error copying input image");
	SAFE_CALL(cudaMemcpy(d_histogram, &histogram, rBytes, cudaMemcpyHostToDevice), "Error copying input image");
	SAFE_CALL(cudaMemcpy(d_normalized_histogram, &normalized_histogram, rBytes, cudaMemcpyHostToDevice), "Error copying input image");

	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);


	auto start_cpu =  chrono::high_resolution_clock::now();
	create_image_histogram<<<grid, block>>>(d_input, d_histogram, nx, ny);
	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel1");
	normalize_histogram<<<grid, block>>>(d_input, d_histogram, d_normalized_histogram, nx, ny);
	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel2");
	contrast_image<<<grid, block>>>(d_input, d_output, d_normalized_histogram, nx, ny);
	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel3");
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("elapsed %f ms\n", duration_ms.count());

	// SAFE_CALL kernel error
	SAFE_CALL(cudaGetLastError(), "Error with last error");


	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, nBytes, cudaMemcpyDeviceToHost), "Error copying image");


	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);


  //showing initial image vs contrast change
	cv::imshow("Input", grayscale_input);
	cv::imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
