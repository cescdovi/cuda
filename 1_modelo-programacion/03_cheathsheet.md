# 3 Cheatsheets

## 3.1 Flujo mecánico para problemas CUDA
| Paso | Quién decide | Cómo |
|---|---|---|
| **1. Dimensionalidad** | El **problema** | 1D vector, 2D matriz, 3D volumen |
| **2. Tamaño** | Los **datos** | N, M×N, M×N×K |
| **3. Hilos/bloque** ⭐ | **Tú** + hardware | Múltiplo de 32, default 256 totales |
| **4. Bloques/grid** | **Fórmula** | `ceil(tamaño / hilos_por_bloque)` |
| **5. Lanzamiento** | Ejecutar | `kernel<<<grid, block>>>(args)` |

**1. Dimensionalidad** — La forma natural del problema fija cuántas dimensiones tiene el grid: un vector pide 1D, una matriz 2D, un volumen 3D. No lo eliges tú, viene dado por la estructura de los datos.

**2. Tamaño** — Los datos dan los números a cubrir: `N` elementos para un vector, `M×N` para una matriz, `M×N×K` para un volumen. Es el total de trabajo que el grid tiene que abarcar.

**3. Hilos por bloque** ⭐ — Aquí sí decides tú, con dos reglas: debe ser **múltiplo de 32** (tamaño del warp) y por defecto apunta a **256 threads totales** por bloque (256 en 1D, `16×16` en 2D, `8×8×8` en 3D). Es el único paso donde tienes libertad real; el resto se deduce de este.

**4. Bloques por grid** — Una vez fijados tamaño y hilos por bloque, los bloques salen por fórmula: `ceil(tamaño / hilos_por_bloque)` en cada dimensión. Garantiza que cubres todos los datos aunque el tamaño no sea múltiplo exacto, y los hilos sobrantes se descartan con un `if (i < N)` dentro del kernel.

**5. Lanzamiento** — Solo queda escribir `kernel<<<grid, block>>>(args)` y la GPU se encarga del resto.

#### Defaults de `blockDim`

| Dim | Valor | Total threads |
|---|---|---|
| 1D | `256` | 256 |
| 2D | `dim3(16, 16)` | 256 |
| 3D | `dim3(8, 8, 8)` | 512 |

#### Fórmula del ceil (aplicada a cada dimensión)

```c
numBlocks = (tamaño + threadsPerBlock - 1) / threadsPerBlock;
```

#### Plantilla del kernel
```c
__global__ void kernel(int N, ...) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {              // ← check de límites SIEMPRE
        // operar
    }
}
```

## 3.2 Flujo típico host ↔ device

El esqueleto que repetirás en casi todos los programas CUDA:

```c
int main() {
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    // 1. Reservar memoria HOST
    float *h_A = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    inicializar(h_A, N);
    
    // 2. Reservar memoria DEVICE
    float *d_A, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);
    
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

## 3.3 Compilador (nvcc)

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

## 3.4 Especificadores y variables built-in

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

## 3.5 Memoria del device

#### Reservar y liberar

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

## 3.6 Sincronización

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

## 3.7 Manejo de errores

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

