
/*************************************
 * Simple CUDA kernel for vector sum *
 *************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                              \
 cudaError_t err = call;                                                      \
 if( cudaSuccess != err ) {                                                   \
   fprintf(stderr,"CUDA error at %s:%d: %s\n",                                \
           __FILE__, __LINE__, cudaGetErrorString(err));                      \
   exit(err);                                                                 \
 } }


__global__ void compute_kernel( const unsigned int n, float *d_a, float *d_b, float *d_c ) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;

  if ( i < n) {
    d_c[i] = d_a[i] + d_b[i];
  }
}

int cu_vector_sum( const unsigned int n, const unsigned int blocksize, float *h_a, float *h_b, float *h_c ) {

  // 2. Reservar memoria en GPU
  float *d_a, *d_b, *d_c; 
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, n * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, n * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, n * sizeof(float)));

  // 3. Copiar los datos de CPU a GPU
  CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));
  
  //definir nº bloques: tamaño problema / threadsperblock
  int nblocks = (n + blocksize -1) / blocksize;

  // 4. Lanzar el kernel
  dim3 dimGrid( nblocks );
  dim3 dimBlock( blocksize );
  compute_kernel<<<nblocks, blocksize>>>(n, d_a, d_b, d_c);

  // 5.  Copiar resultados de GPU a CPU
  CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

  // 6. Liberar memoria en GPU
  CUDA_SAFE_CALL(cudaFree(d_a));
  CUDA_SAFE_CALL(cudaFree(d_b));
  CUDA_SAFE_CALL(cudaFree(d_c));
  
  return EXIT_SUCCESS;
}
 
int vector_sum( const unsigned int n, float *a, float *b, float *c ) {

  for( unsigned int i=0; i<n; i++ ) {
      c[ i ] = a[ i ] + b[ i ];
  }
  return EXIT_SUCCESS;

}

int main( int argc, char *argv[] ) {
  unsigned int n;
  unsigned int blocksize;

  /* Generating input data */
  if( argc<3 ) {
    printf("Usage: %s n blocksize \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  sscanf(argv[2],"%d",&blocksize);
  float *a = (float *) malloc( n*sizeof(float) );
  float *b = (float *) malloc( n*sizeof(float) );
  printf("%s: Generating two random vectors of size %d...\n",argv[0],n);
  for( int i=0; i<n; i++ ) {
    a[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    b[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
  }

  printf("%s: Adding vectors in CPU...\n",argv[0]);
  float *c_cpu = (float *) malloc( n*sizeof(float) );
  vector_sum( n, a, b, c_cpu );
  
  // 1. Reservar memoria en CPU
  printf("%s: Adding matrices in GPU...\n",argv[0]);
  float *c_gpu = (float *) malloc( n*sizeof(float) );
  cu_vector_sum( n, blocksize, a, b, c_gpu );

  /* Check for correctness */
  float error = 0.0f;
  for( int i=0; i<n; i++ ) {
    error += fabs( c_gpu[ i ] - c_cpu[ i ] );
  }
  printf("Error CPU/GPU = %.3e\n",error);
  
  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  
}

