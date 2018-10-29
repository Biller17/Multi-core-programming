#include "common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

using namespace std;

#define TILEDIM 32


//Code used from examples and modified for activity
//Adrian Biller A01018940




//inicialization of matrices
void initialData(float *ip, const int size)
{
    int i;
    for(i = 0; i < size; i++)
    {
        ip[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
    }

    return;
}

//printing arrays
void printArray(int * arr, int size)
{
  int totalSize = size * size;
  int row = 1;
  for(int x = 0; x < totalSize; x++){
    printf("%d ", arr[x]);
    if((size * row)-1 == x){
      row++;
      printf("\n");
    }
  }
}


//multiplication of matrices using cpu
void multiplyMatrixOnHost(float *A, float *B, float *C, int nx, int ny)
{
      for(int i = 0; i < nx; i++){
        for(int j = 0; j < nx ; j++){
          for(int k = 0; k < nx; k++){
            C[i*nx+j] += A[i*nx+k] * B[k*nx+j];
          }
        }
      }

    return;
}

//checking result of gpu and comparing them with cpu matrix
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}




// //matrix calculation using cpu
// __global__ void multMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
// {
//     unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
//     unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//     unsigned int idx = iy * nx + ix;
//
//
//     if (ix < nx && iy < ny){
//         for(int k = 0; k < nx; k++){
//           MatC[ix * nx + iy] += MatA[ix * nx + k] * MatB[k * nx + iy];
//         }
//     }
// }

//matrix calculation using tile method
__global__ void tiledMult(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * TILEDIM;
    unsigned int iy = threadIdx.y + blockIdx.y * TILEDIM;
    // unsigned int idx = iy * nx + ix;

    __shared__ float sharedMatA[TILEDIM][TILEDIM];
    __shared__ float sharedMatB[TILEDIM][TILEDIM];


    int ty = threadIdx.y;
    int tx = threadIdx.x;
    float sum = 0;

    //initializing tile in 0s
    __syncthreads();
    for(int i = 0; i <= TILEDIM; i ++) {
      for(int j = 0; j <= TILEDIM; j++) {
        sharedMatA[i][j] = 0;
        sharedMatA[i][j] = 0;
      }
    }
    __syncthreads();

    for(int i = (TILEDIM + nx - 1)/TILEDIM; i > 0; i--) {

      if((i * TILEDIM + tx) < nx && (iy < ny)) {
        sharedMatA[ty][tx] = MatA[(iy*ny) + (i*TILEDIM+tx)];
      }
      else{
        sharedMatA[ty][tx] = 0;
      }
      if((i * TILEDIM + ty) < ny && (ix < nx)) {
        sharedMatB[ty][tx] = MatB[(i*TILEDIM+ty) * nx + ix];
      }
      else{
        sharedMatB[ty][tx] = 0;
      }

      //syncing threads and getting final value for result matrix
      __syncthreads();
      for(int j = 0; j < TILEDIM; j++) {
        sum += sharedMatA[ty][j] * sharedMatB[j][tx];
      }
      __syncthreads();
    }

    //writing value in result matrix
    if(ix < nx && iy < ny) {
      MatC[iy*ny+ix] = sum;
    }
}



int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    // int nx = 1 << 12;
    // int ny = 1 << 12;
    int nx = 500;
    int ny = 500;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side

    initialData(h_A, nxy);
    initialData(h_B, nxy);
    // printArray(h_A, nx);
    // printf("\n");
    // printArray(h_B, nx);
    // printf("\n");

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    //add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    multiplyMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");
    SAFE_CALL(cudaMemset(d_MatC, 0, nBytes), "Error setting d_MatC to zeros");

    // invoke kernel at host side
    int dimx = TILEDIM;
    int dimy = TILEDIM;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    start_cpu =  chrono::high_resolution_clock::now();
    tiledMult<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    //
    // printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
    //        grid.y,
    //        block.x, block.y, duration_ms.count());
    //
    // // SAFE_CALL kernel error
    // SAFE_CALL(cudaGetLastError(), "Error with last error");
    //
    // // copy kernel result back to host side
    // SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // printArray(hostRef, nx);
    // printf("Host\n");
    // printArray(gpuRef, nx);
    // printf("GPU\n");
    // // check device results
    // checkResult(hostRef, gpuRef, nxy);


    // //calculating matrix multiplication using Tiling
    // start_cpu =  chrono::high_resolution_clock::now();
    // tiledMult<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    // SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    // end_cpu =  chrono::high_resolution_clock::now();
    //
    // duration_ms = end_cpu - start_cpu;

    printf("Matrix multiplication with tiling <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());


    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");


    checkResult(hostRef, gpuRef, nxy);




    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
