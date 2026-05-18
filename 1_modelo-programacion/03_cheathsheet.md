# 3 Cheatsheets CUDA

## 3.1 Las 5 reglas fundamentales

```
┌─────────────────────────────────────────────────────────────┐
│  1.  warp = 32 threads                                      │
│      (siempre, en todas las GPUs NVIDIA)                    │
│                                                             │
│  2.  blockDim ≤ 1024 por restricción de HW                  │
│      (y siempre múltiplo de 32)                             │
│                                                             │
│  3.  Un bloque va a UN SM                                   │
│      (no se reparte; un SM tiene varios bloques)            │
│                                                             │
│  4.  nblocks = ceil(n / blockDim)                           │
│      (para cubrir n elementos; habrá hilos sobrantes)       │
│                                                             │
│  5.  blockDim = 256 es buen valor por defecto               │
└─────────────────────────────────────────────────────────────┘
```

## 3.2 Flujo mecánico para diseñar un kernel

### 3.2.1 Paso 1 — Identificar la dimensionalidad del problema

- La define la naturaleza del problema, no el programador
- Tipos según dimensión:
    - 1D → Vector → `N` elementos
    - 2D → Matriz → `M × N`
    - 3D → Volumen → `M × N × K`

### 3.2.2 Paso 2 — Definir el shape de los datos

- Una vez conocida la dimensionalidad, declarar el tamaño en cada eje

### 3.2.3 Paso 3 — Elegir threads por bloque (`blockDim`)

- Debe ser **múltiplo de 32**
- Por defecto, apuntar a **~256 threads totales por bloque**
- Valores típicos según dimensión:
    - 1D → `256` → 256 threads
    - 2D → `dim3(16, 16)` → 256 threads
    - 3D → `dim3(8, 8, 8)` → 512 threads
- El resto de pasos se deduce a partir de aquí

### 3.2.4 Paso 4 — Calcular el número de bloques (`gridDim`)

- Fórmula del techo (ceiling division) para garantizar que todos los elementos quedan cubiertos:
    ```c
    numBlocks = (shape + threadsPerBlock - 1) / threadsPerBlock;
    ```
- Los hilos sobrantes del último bloque se descartan con `if (i < N)` dentro del kernel
- Para 2D/3D aplicas la misma fórmula por cada eje:
    ```c
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);   // ejemplo 2D
    ```

### 3.2.5 Paso 5 — Decidir el patrón: 1 thread por elemento, o varios

- Opciones según tamaño del problema:
    - `n` manejable (< 10M) → patrón **1 thread = 1 elemento** → kernel sin bucle
    - `n` enorme (> 10M) → patrón **grid-stride loop** → kernel con bucle
- Para empezar, **siempre usa 1 thread = 1 elemento**
    - Es lo más simple

### 3.2.6 Paso 6 — Lanzar el kernel y escribirlo

- Estructura en el host:
    ```cuda
    kernel<<<gridDim, blockDim>>>(args);
    ```
- Estructura del kernel:
    ```cuda
    __global__ void kernel(int N, ...) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {           // ← check de límites SIEMPRE
            // operar
        }
    }
    ```

## 3.3 Plantilla universal (copy-paste)

- En el host:
    ```cuda
    int blockDim = 256;
    int nblocks  = (n + blockDim - 1) / blockDim;
    kernel<<<nblocks, blockDim>>>(n, d_a, d_b, d_c);
    ```
- El kernel:
    ```cuda
    __global__ void kernel(int n, float *a, float *b, float *c) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];   // o lo que sea
        }
    }
    ```
- Cubre el ~80% de kernels básicos:
    - Suma de vectores
    - Saxpy
    - Aplicar una función elemento a elemento
    - Mapear cualquier operación 1-a-1

## 3.4 Variante: grid-stride loop (para `n` enorme)

- Solo necesario cuando lanzar `n` threads sería demasiado
- Estructura del kernel:
    ```cuda
    __global__ void kernel(int n, float *a, float *b, float *c) {
        int idx    = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < n; i += stride) {
            c[i] = a[i] + b[i];
        }
    }
    ```

## 3.5 Conceptos importantes (para tener en la cabeza, no para memorizar)

### 3.5.1 Los tres niveles de concurrencia

- Pregunta clave: *"¿Cuántos threads ejecutan a la vez?"*
- Depende de qué entiendas por "a la vez":
    - Cuántos **lanzas** → lo que tú decidas en `<<<...>>>`
    - Cuántos están **vivos** en hardware → muchos miles (residentes)
    - Cuántos **ejecutan instrucción este ciclo** → unos pocos miles (ejecutando)
- Como programador, **no tienes que preocuparte de esto**
    - La GPU se encarga

### 3.5.2 Solo importan dos cosas como programador

- Cuando escribes un kernel, solo te tienen que importar **dos cosas**:
    - **`blockDim` múltiplo de 32** → para no desperdiciar threads del último warp
    - **`nblocks` lo bastante grande** → para cubrir tu problema (`ceil(n / blockDim)`)
- Todo lo demás (warps, SMs, schedulers, ocupación, residentes...) **la GPU lo gestiona sola**

## 3.6 Cuándo te empezará a importar el resto

- Solo cuando aparezca uno de estos síntomas:
    - *"Mi kernel es lento"* → coalescencia, ocupación
    - *"Da resultados raros"* → race conditions, sincronización
    - *"Quiero optimizar más"* → shared memory, registros
    - *"Compite con la CPU"* → memory bandwidth, latency hiding
- Mientras no tengas esos problemas, **el modelo simple es suficiente**

## 3.3 Kernels por dimensionalidad

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

## 3.4 Flujo típico host ↔ device

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

## 3.5 Compilador (nvcc)

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

## 3.6 Especificadores y variables built-in

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

## 3.7 Memoria del device

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

#### `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol`

- Copian datos entre el host y una **variable global del device** (`__device__` o `__constant__`), identificándola por **nombre del símbolo** en lugar de por puntero. Son el puente estándar para leer/escribir escalares o estructuras pequeñas declaradas globalmente en la GPU, sin necesidad de `cudaMalloc`.

```c
cudaMemcpyToSymbol(symbol, src, bytes);    // host → device
cudaMemcpyFromSymbol(dst, symbol, bytes);  // device → host
```

| Parámetro | Significado |
|---|---|
| `symbol` | Variable global del device (`__device__` / `__constant__`), pasada por nombre |
| `src` / `dst` | Puntero en el host (origen en `To`, destino en `From`) |
| `bytes` | Número de bytes a copiar (típicamente `sizeof(tipo)`) |
| `offset` *(opcional)* | Desplazamiento en bytes dentro del símbolo (por defecto 0) |
| `kind` *(opcional)* | Dirección de la copia (por defecto la correcta para cada función) |


## 3.8 Sincronización

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

## 3.9 Manejo de errores

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

