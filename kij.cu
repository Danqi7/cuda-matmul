#define FP float

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

__global__ void gpu_matrixmult(FP *a,FP *b, FP *c, int n, int p, int m) {
  // A: [n,p], B: [p, m], C: [n, m]

  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  int indexb = col;
  int index = row * m + col;
  
  if(col < m && row < n) {
    c[index] = 0.;
    for (int indexa = row*p; indexa < (row*p + p); indexa++, indexb+=m) 
      c[index] += a[indexa]*b[indexb];
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
  printf("n: %d, p: %d, m: %d\n", n, p, m);

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


int main() {
    int n = 3, p = 3, m = 3; // Example dimensions
    FP *a, *b, *c;

    a = (FP *)malloc(n * p * sizeof(FP));
    b = (FP *)malloc(p * m * sizeof(FP));
    c = (FP *)malloc(n * m * sizeof(FP));

    // if (a == NULL || b == NULL || c == NULL) {
    //     // Handle allocation failure
    // }

    // Initialize matrices a, b, and c
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

    cpu_matrixmult_kij(a, b, c, n, p, m);

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
}
