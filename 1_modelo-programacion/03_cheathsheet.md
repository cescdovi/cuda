# 3 Cheatsheets

## 3.1 Flujo mecánico para problemas CUDA


1. **Dimensionalidad**: observar dimensionalidad del problema.
    - no la define el usuario
    - 1D vector, 2D matriz, 3D volumen 

2. **Shape**: definir el shape de los datos dada la dimensionalidad
    - `N` elementos para un vector, `M×N` para una matriz, `M×N×K` para un volumen. 

3. **Definir el nº de hilos por bloque**: debe ser **múltiplo de 32** (tamaño del warp)
    - el resto de pasos se deduce a partir de este
    - por defecto: **256 threads totales** 

        | Dim | Valor | Total threads |
        |---|---|---|
        | 1D | `256` | 256 |
        | 2D | `dim3(16, 16)` | 256 |
        | 3D | `dim3(8, 8, 8)` | 512 |

4. **Bloques por grid** el nº de bloques por grid se calcula: 
    ```c
    numBlocks = (shape + threadsPerBlock - 1) / threadsPerBlock;
    ```
    - los hilos sobrantes se descartan con un `if (i < N)` dentro del kernel.

5. **Lanzar el kernel**`kernel<<<grid, block>>>(args)`

```c
__global__ void kernel(int N, ...) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {  // ← check de límites SIEMPRE
        // operar
    }
}
```

## 3.2 Kernels por dimensionalidad

Mismo patrón en los tres casos: cada hilo calcula sus coordenadas globales, comprueba límites y opera sobre un elemento. La diferencia está en cuántos índices se calculan y cómo se aplana a memoria.

#### 1D — vectores (`N` elementos)

```c
__global__ void sumar1D(int N, const float *A, const float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void multiplicar1D(int N, const float *A, const float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}
```

Lanzamiento:
```c
int threads = 256;
int blocks  = (N + threads - 1) / threads;
sumar1D<<<blocks, threads>>>(N, d_A, d_B, d_C);
```

#### 2D — matrices (`M` filas × `N` columnas, row-major)
```
Matriz 3×4:           Memoria lineal (12 elementos):
[ a  b  c  d ]        [ a b c d | e f g h | i j k l ]
[ e  f  g  h ]          fila 0     fila 1    fila 2
[ i  j  k  l ]

```
```c
__global__ void sumar2D(int M, int N, const float *A, const float *B, float *C) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // fila
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // columna
    if (j < N && i < M) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void multiplicar2D(int M, int N, const float *A, const float *B, float *C) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N && i < M) {
        int idx = i * N + j;
        C[idx] = A[idx] * B[idx];
    }
}
```

Lanzamiento:
```c
dim3 threads(16, 16);
dim3 blocks((N + 15) / 16, (M + 15) / 16);
sumar2D<<<blocks, threads>>>(M, N, d_A, d_B, d_C);
```

#### 3D — volúmenes (`M × N × K`)

```c
__global__ void sumar3D(int M, int N, int K,
                        const float *A, const float *B, float *C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (k < K && j < N && i < M) {
        int idx = (i * N + j) * K + k;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void multiplicar3D(int M, int N, int K,
                              const float *A, const float *B, float *C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (k < K && j < N && i < M) {
        int idx = (i * N + j) * K + k;
        C[idx] = A[idx] * B[idx];
    }
}
```

Lanzamiento:
```c
dim3 threads(8, 8, 8);
dim3 blocks((K + 7) / 8, (N + 7) / 8, (M + 7) / 8);
sumar3D<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
```

## 3.3 Flujo típico host ↔ device

```c
// Kernel sencillo: C[i] = A[i] * 2
__global__ void miKernel(int N, const float *A, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] * 2.0f;
    }
}

int main() {
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    // 1. Reservar memoria HOST
    float *h_A = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    inicializar(h_A, N);
    
    // 2. Reservar memoria DEVICE
    float *d_A, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_C, bytes);
    
    // 3. Copiar datos HOST → DEVICE
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    
    // 4. Configurar y lanzar KERNEL
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    miKernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_C);
    
    // 5. Comprobar errores y sincronizar (opcional, lo hace cudaMemcpy)
    cudaDeviceSynchronize();
    
    // 6. Copiar resultado DEVICE → HOST
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // 7. Liberar memoria
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
    
    return 0;
}
```

## 3.4 Compilador (nvcc)

`nvcc` es el compilador de CUDA. Separa el código en host (lo pasa a `gcc`/`clang`/`MSVC`) y device (lo compila a PTX/SASS).

#### Comandos básicos

```bash
nvcc programa.cu -o programa              # compilación básica
nvcc -O3 programa.cu -o programa          # con optimización
nvcc -G -g programa.cu -o programa        # debug (device + host)
nvcc -arch=sm_80 programa.cu              # target específico (Ampere)
```

#### Flags importantes

| Flag | Para qué |
|---|---|
| `-arch=sm_XX` | Target de Compute Capability (sm_75=Turing, sm_80=Ampere, sm_90=Hopper) |
| `-O0` / `-O3` | Nivel de optimización |
| `-G` | Información de debug en device |
| `-g` | Información de debug en host |
| `-lineinfo` | Info de líneas para profiling sin overhead de `-G` |
| `-Xcompiler "..."` | Pasa flags al compilador host |
| `-Xptxas -v` | Muestra registros y shared memory usados por kernel |
| `--use_fast_math` | Usa intrínsecas rápidas (menos precisas) |
| `-rdc=true` | Compilación separada (relocatable device code) |

#### Mapping arch → arquitectura

| `-arch=` | Arquitectura | GPU típica |
|---|---|---|
| `sm_60` | Pascal | P100 |
| `sm_70` | Volta | V100 |
| `sm_75` | Turing | T4, RTX 2080 |
| `sm_80` | Ampere | A100 |
| `sm_86` | Ampere (consumer) | RTX 3090 |
| `sm_89` | Ada Lovelace | RTX 4090 |
| `sm_90` | Hopper | H100 |

## 3.5 Especificadores y variables built-in

#### Especificadores de funciones

| Especificador | Llamada | Ejecución | Uso |
|---|---|---|---|
| `__global__` | Host | Device | Kernels (entry points), `void` obligatorio |
| `__device__` | Device | Device | Funciones auxiliares dentro de kernels |
| `__host__` | Host | Host | Función normal de CPU (default) |
| `__host__ __device__` | Ambos | Ambos | Compila para ambos lados |

#### Especificadores de variables

| Especificador | Dónde vive | Ámbito | Vida |
|---|---|---|---|
| (sin especificador, en kernel) | Registers / local | Thread | Kernel |
| `__shared__` | Shared memory | Bloque | Kernel |
| `__device__` (global) | Global memory | Grid | Aplicación |
| `__constant__` | Constant memory | Grid | Aplicación |

#### Variables built-in dentro del kernel

| Variable | Tipo | Qué representa |
|---|---|---|
| `threadIdx.{x,y,z}` | `dim3` | Posición del thread dentro del bloque |
| `blockIdx.{x,y,z}` | `dim3` | Posición del bloque dentro del grid |
| `blockDim.{x,y,z}` | `dim3` | Tamaño del bloque |
| `gridDim.{x,y,z}` | `dim3` | Tamaño del grid |
| `warpSize` | `int` | Siempre 32 |

#### Sintaxis de lanzamiento

```c
kernel<<<numBlocks, threadsPerBlock>>>(args);
kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(args);
kernel<<<numBlocks, threadsPerBlock, sharedMemBytes, stream>>>(args);
```

## 3.6 Memoria del device

#### Reservar y liberar

```c
cudaMalloc(&devPtr, bytes);
```

| Argumento | Significado |
|---|---|
| `&devPtr` | Dirección del puntero del device (se rellena con la dirección reservada) |
| `bytes` | Tamaño en bytes a reservar (`N * sizeof(tipo)`) |

```c
float* d_A;
cudaMalloc(&d_A, N * sizeof(float));      // reserva en device
cudaFree(d_A);                             // libera

cudaMallocHost(&h_A, N * sizeof(float));   // pinned host memory (transferencias rápidas)
cudaFreeHost(h_A);

cudaMallocManaged(&p, N * sizeof(float));  // unified memory (visible host+device)
```

#### Copiar datos

```c
cudaMemcpy(dst, src, bytes, direction);
```

| Dirección | Significado |
|---|---|
| `cudaMemcpyHostToDevice` | CPU → GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU |
| `cudaMemcpyDeviceToDevice` | GPU → GPU |
| `cudaMemcpyHostToHost` | CPU → CPU (raro) |

#### Inicializar memoria del device

```c
cudaMemset(d_A, 0, N * sizeof(float));     // pone bytes a 0
```

#### Copia asíncrona (con streams)

```c
cudaMemcpyAsync(dst, src, bytes, direction, stream);
```

## 3.7 Sincronización

| Función | Nivel | Cuándo usar |
|---|---|---|
| `__syncthreads()` | Bloque | Coordinar threads del mismo bloque (típico tras escribir shared memory) |
| `__syncwarp(mask)` | Warp | Sincronizar threads de un warp (raro, lo hace el hardware) |
| `cudaDeviceSynchronize()` | Host ↔ Device | Esperar a que la GPU acabe TODO el trabajo pendiente |
| `cudaStreamSynchronize(stream)` | Host ↔ Stream | Esperar solo a un stream concreto |
| Barrera implícita entre kernels | Grid | Automática al lanzar kernels secuencialmente |

#### Reglas de oro

- `__syncthreads()` **debe ser alcanzable por todos los threads del bloque** (no dentro de un `if` divergente) → si no, deadlock.
- El lanzamiento de kernel es **asíncrono**: `cudaDeviceSynchronize()` después si necesitas esperar.
- No hay sincronización entre bloques dentro de un kernel (sin `cooperative_groups`).

## 3.8 Manejo de errores

Las funciones CUDA devuelven `cudaError_t`. Los errores asíncronos (en kernels) no se ven hasta que sincronizas.

#### Patrón básico

```c
cudaError_t err = cudaMalloc(&d_A, bytes);
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

#### Macro útil para envolver llamadas

```c
#define CUDA_CHECK(call) do {                               \
    cudaError_t e = (call);                                 \
    if (e != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

// Uso:
CUDA_CHECK(cudaMalloc(&d_A, bytes));
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
```

#### Comprobar errores tras un kernel

```c
miKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // error de lanzamiento
CUDA_CHECK(cudaDeviceSynchronize());   // error de ejecución
```

#### Funciones útiles

| Función | Para qué |
|---|---|
| `cudaGetLastError()` | Devuelve y resetea el último error |
| `cudaPeekAtLastError()` | Devuelve sin resetear |
| `cudaGetErrorString(err)` | String legible del error |
| `cudaGetDeviceCount(&n)` | Cuántas GPUs hay |
| `cudaGetDeviceProperties(&prop, id)` | Info de la GPU |
| `cudaSetDevice(id)` | Selecciona GPU activa |

