
/*************************************
 * Simple CUDA kernel for vector sum *
 *************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

__global__ void compute_kernel( const unsigned int n, float a, float *d_x, float *d_y ) {

                /* AQUI se implementa el kernel */

}

int cu_saxpy( const unsigned int n, const unsigned int blocksize, const unsigned int gridsize, float a, float *h_x, float *h_y ) {

  // Allocate device memory
  float *d_x, *d_y;
  CUDA_SAFE_CALL( /* Reserva de memoria para d_x */ );
  CUDA_SAFE_CALL( /* Reserva de memoria para d_y */ );

  // Copy host memory to device 
  CUDA_SAFE_CALL( /* Copia de datos de h_x a d_x */ );
  CUDA_SAFE_CALL( /* Copia de datos de h_y a d_y */ );

  // Execute the kernel
  dim3 dimGrid( gridsize );
  dim3 dimBlock( blocksize );
  /* Llamada al kernel */

  // Copy device memory to host 
  CUDA_SAFE_CALL( /* Copia de datos de d_y a h_y */ );

  // Deallocate device memory
  CUDA_SAFE_CALL( /* Liberar espacio de memoria asignado a d_x */ );
  CUDA_SAFE_CALL( /* Liberar espacio de memoria asignado a d_y */ );

  return EXIT_SUCCESS;
}
 
int saxpy( const unsigned int n, float a, float *x, float *y ) {

  for( unsigned int i=0; i<n; i++ ) {
      y[ i ] += a * x[ i ];
  }
  return EXIT_SUCCESS;

}

int main( int argc, char *argv[] ) {
  unsigned int n;
  unsigned int blocksize;
  unsigned int gridsize;

  /* Generating input data */
  if( argc<4 ) {
    printf("Usage: %s n blocksize gridsize \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  sscanf(argv[2],"%d",&blocksize);
  sscanf(argv[3],"%d",&gridsize);
  float *x = (float *) malloc( n*sizeof(float) );
  float *y = (float *) malloc( n*sizeof(float) );
  printf("%s: Generating two random vectors of size %d...\n",argv[0],n);
  for( int i=0; i<n; i++ ) {
    x[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    y[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
  }
  float a = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;

  printf("%s: Computing saxpy in CPU...\n",argv[0]);
  float *y_cpu = (float *) malloc( n*sizeof(float) );
  memcpy ( y_cpu, y, n*sizeof(float) );
  saxpy( n, a, x, y_cpu );

  printf("%s: Computing saxpy in GPU...\n",argv[0]);
  float *y_gpu = (float *) malloc( n*sizeof(float) );
  memcpy ( y_gpu, y, n*sizeof(float) );
  cu_saxpy( n, blocksize, gridsize, a, x, y_gpu );

  /* Check for correctness */
  float error = 0.0f;
  float norm = 0;
  for( int i=0; i<n; i++ ) {
    error += fabs( y_gpu[ i ] - y_cpu[ i ] );
    norm += y_cpu[ i ] * y_cpu[ i ];
  }
  norm = sqrt( norm );
  printf("Error CPU/GPU = %.3e\n",error/norm);


  free(x);
  free(y);
  free(y_cpu);
  free(y_gpu);
  
}

