#define FP float

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

#define TW 32

__global__ void gpu_matrixmult_tiled(FP *a,FP *b, FP *c, int n, int p, int m) {
  // A: [n,p], B: [p, m], C: [n, m]
  // Tiled Version, block_dim = TW, tile_width
  __shared__ FP atile[TW][TW];
  __shared__ FP btile[TW][TW];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int col = tx + blockDim.x * blockIdx.x;
  int row = ty + blockDim.y * blockIdx.y;

  int index = row * m + col;
  FP cvalue = 0;

  for (int i = 0; i < (p + TW - 1)/TW; i++) {
    // Load a,b tiles from global memory to shared memory
    // Make sure that the tile is within the matrix bounds
    if (i*TW + tx < p && row < n) {
        atile[ty][tx] = a[row * p + i*TW + tx];
    } else {
        atile[ty][tx] = 0.0;
    }
    if (i*TW + ty < p && col < m) {
        btile[ty][tx] = b[(i*TW + ty) * m + col];
    } else {
        btile[ty][tx] = 0.0;
    }
    __syncthreads(); // ensure a,b tiles are loaded before computation

    for (int k = 0; k < TW; k++) {
      cvalue += atile[ty][k] * btile[k][tx];
    }
    __syncthreads(); // ensure all threads have finished before loading new tiles

  }
  
  // check if the thread is within the result matrix bounds
  if (row < n && col < m) {
    c[index] = cvalue;
  }

}


void cpu_matrixmult(FP *a,FP *b, FP *c, int n, int p, int m) {
  // A: [n,p], B: [p, m], C: [n, m]
  int index, indexa, indexb;
  FP cvalue;
  for(int col=0;col < m; col++)
    for(int row=0;row < n; row++) {
      indexb = col;
      index = row * m + col;
      cvalue = 0.;
      for (indexa = row*p; indexa < (row*p + p); indexa++, indexb+=m) 
	      cvalue += a[indexa]*b[indexb];
      c[index] -= cvalue; //NOTE: This calculates the diff between CPU and GPU computations.
    }
}

void cpu_matrixmult_kij(FP *a,FP *b, FP *c, int n, int p, int m) {
  // A: [n,p], B: [p, m], C: [n, m]
  int index, indexa, indexb;
  FP r;

  for (int k=0; k<p; k++) {

    for (int i=0; i<n; i++) {
      indexa = i * p + k;
      // r = A[i][k];
      r = a[indexa];
      for (int j=0; j<m; j++) {
        index = i * m + j;
        indexb = k * m + j;
        //C[i][j] += r * B[k][j];
        c[index] -= r * b[indexb]; // NOTE: diff between CPU and GPU computations.
      }
    }

  }
}


int main(int argc, char *argv[]) {

  int i, j; // loop counters

  int gpucount = 0; // Count of available GPUs
  int gpunum = 0; // Device number to use
  int Grid_Dim_X = 1; //Grid x dimension
  int Grid_Dim_Y = 1; // Grid y dim
  int Block_Dim = 1; //Block dimension, x and y, square

  int n, m, p; // matrix dimensions
  FP *a,*b,*c;
  FP *dev_a, *dev_b, *dev_c;
  size_t Asize, Bsize, Csize; // number of bytes in arrays

  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  cudaError_t errorcode;

  // --------------------SET PARAMETERS AND DATA -----------------------

  errorcode = cudaGetDeviceCount(&gpucount);
  if (errorcode == cudaErrorNoDevice) {
    printf("No GPUs are visible\n");
    exit(-1);
  }
  else {
     printf("Device count = %d\n",gpucount);
  }

  if (argc!=6) {
    printf("Usage: matmul <matrix dim n> <matrix dim m> <matrix dim p> <block dim> <grid dim>\n");
    exit (-1);
  }

  n = atoi(argv[1]);
  m = atoi(argv[2]);
  p = atoi(argv[3]);

  //Block_Dim = atoi(argv[4]); // Square block
  Block_Dim = TW; // sqaure block, TW=Block_Dim
  if (Block_Dim*Block_Dim > 1024) {
    printf("Error, too many threads in block\n");
    exit (-1);
  }

  Grid_Dim_X = (m + Block_Dim - 1) / Block_Dim; // rectangular grid; how many blocks in X direction
  Grid_Dim_Y = (n + Block_Dim - 1) / Block_Dim; // how many blocks in Y direction

  if (Grid_Dim_X*Block_Dim < m) {
    printf("Error, number of threads in x dimensions less than number of array elements\n");
    exit (-1);
  }
  if (Grid_Dim_Y*Block_Dim < n) {
    printf("Error, number of threads in y dimensions less than number of array elements\n");
    exit (-1);
  }

  cudaSetDevice(gpunum);
  printf("Using device %d\n",gpunum);
  printf("Matrix A Dimension = [%d, %d]\n", n, p);
  printf("Matrix B Dimension = [%d, %d]\n", p, m);

  printf("Block_Dim = (%d, %d), Grid_Dim = (%d, %d)\n", Block_Dim, Block_Dim, Grid_Dim_X, Grid_Dim_Y);

  dim3 Grid(Grid_Dim_X, Grid_Dim_Y); //Grid structure
  dim3 Block(Block_Dim, Block_Dim); //Block structure

  Asize = n * p * sizeof(FP);
  Bsize = p * m * sizeof(FP);
  Csize = n * m * sizeof(FP);

  a = (FP*) malloc(Asize); // dynamically allocated memory for arrays on host
  b = (FP*) malloc(Bsize);
  c = (FP*) malloc(Csize); // results from GPU

  srand(12345);
  for(i=0;i < n;i++)
    for(j=0;j < p;j++) {
      a[i * p + j] = (FP) rand() / (FP) RAND_MAX;
      //      a[i * p + j] = (FP) i+j; // may be helpful for debugging
    }

  for(i=0;i < p;i++)
    for(j=0;j < m;j++) {
      b[i * m + j] = (FP) rand() / (FP) RAND_MAX;
      //      b[i * n + j] = (FP) i+j; // may be helpful for debugging
    }

  // ------------- COMPUTATION DONE ON GPU ----------------------------

  cudaMalloc((void**)&dev_a, Asize); // allocate memory on device
  cudaMalloc((void**)&dev_b, Bsize);
  cudaMalloc((void**)&dev_c, Csize);
  cudaMemcpy(dev_a, a , Asize ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b , Bsize ,cudaMemcpyHostToDevice);

  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);
  // cudaEventSynchronize(start); // not needed

  gpu_matrixmult_tiled<<<Grid,Block>>>(dev_a,dev_b,dev_c, n, p, m);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  cudaMemcpy(c, dev_c, Csize ,cudaMemcpyDeviceToHost);

  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
// START OF OPTIONAL SECTION THAT CAN BE OMITTED
/*
  // ------------- COMPUTATION DONE ON HOST CPU ----------------------------
  // DEBUGGING USE ONLY (AND FOR LIMITED NUMBERS OF TIMING RUNS)

  cudaEventRecord(start, 0); // use same timing
  // cudaEventSynchronize(start); // not needed


  //cpu_matrixmult(a,b,c, n, p, m); // do calculation on host (NOTE: This computes the diff with GPU result.)
  cpu_matrixmult(a,b,c, n, p, m); // do calculation on host (NOTE: This computes the diff with GPU result.)

  cudaEventRecord(stop, 0); // instrument code to measue end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop );

  printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms); // exec. time

// ------------------- check device creates correct results -----------------

  double error, suma, sumb, sumc, ai, bi, ci;
  suma = 0.; sumb = 0; sumc = 0;
  for(i=0;i < n*n;i++) {
    ai = (double) a[i];
    bi = (double) b[i];
    ci = (double) c[i];
    suma += ai*ai;
    sumb += bi*bi;
    sumc += ci*ci;
  }
  suma = sqrt(suma);
  sumb = sqrt(sumb);
  sumc = sqrt(sumc);
  error =  sumc/(suma*sumb);
  printf("Approximate relative error between GPU and CPU: %e\n", error);
*/
// END OF OPTIONAL SECTION THAT CAN BE OMITTED
// -------------- clean up ---------------------------------------

  free(a);
  free(b);
  free(c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
