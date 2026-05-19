# 3 Cheatsheets

## Bloque A — Lo que consultas siempre

### 3.1 Las 6 reglas fundamentales

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
│                                                             │
│  6.  Envolver toda llamada CUDA en CUDA_CHECK               │
│      (sin esto los errores fallan en silencio)              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Flujo mecánico para diseñar un kernel

#### 3.2.1 Paso 1 — Identificar la dimensionalidad del problema

- La define la naturaleza del problema, no el programador
- Tipos según dimensión:
    - 1D → Vector → `N` elementos
    - 2D → Matriz → `M × N`
    - 3D → Volumen → `M × N × K`

#### 3.2.2 Paso 2 — Definir el shape de los datos

- Una vez conocida la dimensionalidad, declarar el tamaño en cada eje

#### 3.2.3 Paso 3 — Elegir threads por bloque (`blockDim`)

- Debe ser **múltiplo de 32**
- Por defecto, apuntar a **~256 threads totales por bloque**
- Valores típicos según dimensión:
    - 1D → `256` → 256 threads
    - 2D → `dim3(16, 16)` → 256 threads
    - 3D → `dim3(8, 8, 8)` → 512 threads
- El resto de pasos se deduce a partir de aquí

#### 3.2.4 Paso 4 — Calcular el número de bloques (`gridDim`)

- Fórmula del techo (ceiling division) para garantizar que todos los elementos quedan cubiertos:
    ```c
    numBlocks = (shape + threadsPerBlock - 1) / threadsPerBlock;
    ```
- Los hilos sobrantes del último bloque se descartan con `if (i < N)` dentro del kernel
- Para 2D/3D aplicas la misma fórmula por cada eje:
    ```c
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);   // ejemplo 2D
    ```

#### 3.2.5 Paso 5 — Decidir el patrón: 1 thread por elemento, o varios

- Opciones según tamaño del problema:
    - `n` manejable (< 10M) → patrón **1 thread = 1 elemento** → kernel sin bucle
    - `n` enorme (> 10M) → patrón **grid-stride loop** → kernel con bucle
- Para empezar, **siempre usa 1 thread = 1 elemento**
    - Es lo más simple

#### 3.2.6 Paso 6 — Lanzar el kernel y escribirlo

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

### 3.3 Plantilla universal mínima (host + kernel completo)

Programa CUDA mínimo funcional: reserva, transfiere, ejecuta, devuelve, libera. Cubre el ~80% de kernels básicos (suma de vectores, SAXPY, operaciones elemento-a-elemento).

```c
__global__ void kernel(int n, const float *a, const float *b, float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];   // o lo que sea
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // 1. 
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockDim = 256;
    int nblocks  = (n + blockDim - 1) / blockDim;
    kernel<<<nblocks, blockDim>>>(n, d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

Variante grid-stride loop (cuando `n` es enorme):

```c
__global__ void kernel(int n, const float *a, const float *b, float *c) {
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
```

### 3.4 Equivalencias mentales rápidas

| Concepto CUDA | Equivalente mental |
|---|---|
| Kernel | "Cuerpo del bucle paralelo" |
| Thread | "Una iteración del bucle" |
| Block | "Grupo de iteraciones que cooperan" |
| Grid | "Conjunto total de iteraciones" |
| `threadIdx + blockIdx * blockDim` | "El índice `i` del bucle for" |
| `__syncthreads()` | "Barrera dentro del cuerpo del bucle" |
| `cudaMemcpy` H2D/D2H | "Cruzar el bus PCIe" |

### 3.5 Conceptos clave de concurrencia

#### 3.5.1 Los tres niveles de concurrencia

- Pregunta clave: *"¿Cuántos threads ejecutan a la vez?"*
- Depende de qué entiendas por "a la vez":
    - Cuántos **lanzas** → lo que tú decidas en `<<<...>>>`
    - Cuántos están **vivos** en hardware → muchos miles (residentes)
    - Cuántos **ejecutan instrucción este ciclo** → unos pocos miles (ejecutando)
- Como programador, **no tienes que preocuparte de esto**: la GPU se encarga

#### 3.5.2 Solo importan dos cosas como programador

- Cuando escribes un kernel, solo te tienen que importar **dos cosas**:
    - **`blockDim` múltiplo de 32** → para no desperdiciar threads del último warp
    - **`nblocks` lo bastante grande** → para cubrir tu problema (`ceil(n / blockDim)`)
- Todo lo demás (warps, SMs, schedulers, ocupación, residentes...) **la GPU lo gestiona sola**

#### 3.5.3 Cuándo te empezará a importar el resto

- Solo cuando aparezca uno de estos síntomas:
    - *"Mi kernel es lento"* → coalescencia, ocupación
    - *"Da resultados raros"* → race conditions, sincronización
    - *"Quiero optimizar más"* → shared memory, registros
    - *"Compite con la CPU"* → memory bandwidth, latency hiding
- Mientras no tengas esos problemas, **el modelo simple es suficiente**

### 3.6 Índices y dimensiones

Mismo patrón en los tres casos: cada hilo calcula sus coordenadas globales, comprueba límites y opera sobre un elemento. La diferencia está en cuántos índices se calculan y cómo se aplana a memoria.

#### 3.6.1 1D — vectores (`N` elementos)

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

#### 3.6.2 2D — matrices (`M` filas × `N` columnas, row-major)
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
        int idx = i * N + j; //i_global * n_cols + j_global
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

#### 3.6.3 3D — volúmenes (`M × N × K`)

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

#### 3.6.4 Cálculo de bloques (ceil) y bounds check

- Fórmula universal: `nblocks = (n + blockDim - 1) / blockDim`
- Siempre acompañar de `if (idx < n)` dentro del kernel
- En 2D/3D, aplicar ceil **por cada eje** y check para **cada eje**

#### 3.6.5 Grid-stride loop

- Patrón cuando lanzas menos threads que elementos:
    ```c
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) { ... }
    ```
- Ventajas: kernel independiente del tamaño del problema, reutiliza threads vivos

---

## Bloque B — Sintaxis de referencia

### 3.7 Especificadores y built-ins

#### 3.7.1 Especificadores de funciones

| Especificador | Llamada | Ejecución | Uso |
|---|---|---|---|
| `__global__` | Host | Device | Kernels (entry points), `void` obligatorio |
| `__device__` | Device | Device | Funciones auxiliares dentro de kernels |
| `__host__` | Host | Host | Función normal de CPU (default) |
| `__host__ __device__` | Ambos | Ambos | Compila para ambos lados |

#### 3.7.2 Especificadores de variables

| Especificador | Dónde vive | Ámbito | Vida |
|---|---|---|---|
| (sin especificador, en kernel) | Registers / local | Thread | Kernel |
| `__shared__` | Shared memory | Bloque | Kernel |
| `__device__` (global) | Global memory | Grid | Aplicación |
| `__constant__` | Constant memory | Grid | Aplicación |

#### 3.7.3 Variables built-in dentro del kernel

| Variable | Tipo | Qué representa |
|---|---|---|
| `threadIdx.{x,y,z}` | `dim3` | Posición del thread dentro del bloque |
| `blockIdx.{x,y,z}` | `dim3` | Posición del bloque dentro del grid |
| `blockDim.{x,y,z}` | `dim3` | Tamaño del bloque |
| `gridDim.{x,y,z}` | `dim3` | Tamaño del grid |
| `warpSize` | `int` | Siempre 32 |

#### 3.7.4 Sintaxis de lanzamiento `<<<grid, block, shmem, stream>>>`

Lanza un kernel desde host. Los dos últimos parámetros son opcionales.

```c
kernel<<<numBlocks, threadsPerBlock>>>(args);
kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(args);
kernel<<<numBlocks, threadsPerBlock, sharedMemBytes, stream>>>(args);
```

| Parámetro | Significado |
|---|---|
| `numBlocks` | `int` o `dim3`: tamaño del grid (cuántos bloques) |
| `threadsPerBlock` | `int` o `dim3`: tamaño del bloque (cuántos threads) |
| `sharedMemBytes` | Bytes de shared memory dinámica (por defecto 0) |
| `stream` | Stream donde se lanza (por defecto stream 0) |

### 3.8 Jerarquía y tipos de memoria

#### 3.8.1 Visión rápida

```
┌───────────────────────────────────────────────────────┐
│  Registers     │ per-thread  │ ~1 ciclo   │ on-chip  │
│  Local         │ per-thread  │ ~400-800   │ DRAM     │
│  Shared        │ per-block   │ ~20-30     │ on-chip  │
│  L1 cache      │ per-SM      │ ~20-30     │ on-chip  │
│  L2 cache      │ todos SMs   │ ~200       │ on-chip  │
│  Global        │ per-grid    │ ~400-800   │ DRAM     │
│  Constant      │ per-grid    │ ~1 (hit)   │ DRAM+$   │
│  Texture       │ per-grid    │ variable   │ DRAM+$   │
└───────────────────────────────────────────────────────┘
```

#### 3.8.2 Registros y local memory (per-thread)

- **Registros**: declaras una variable normal dentro del kernel y vive en registros (si caben). Lo más rápido.
- **Local memory**: si una variable es demasiado grande o se accede dinámicamente (arrays con índice no-constante), el compilador la pone en *local memory* → realmente en DRAM, lenta.
- Síntoma de spill: el kernel va inesperadamente lento → mirar con `-Xptxas -v`.

#### 3.8.3 Shared memory (per-block, on-chip)

- Memoria compartida entre todos los threads del mismo bloque
- Tamaño típico: 48–228 KB por SM según arquitectura
- Dos formas de declarar:
    ```c
    __shared__ float tile[16][16];     // estática (tamaño fijo)
    extern __shared__ float buf[];     // dinámica (tamaño en <<<...>>>)
    ```
- Usar siempre con `__syncthreads()` para coordinar lecturas/escrituras

#### 3.8.4 Global memory (per-grid, DRAM)

- La VRAM de la GPU; visible para todos los threads y persistente entre kernels
- Acceso lento (~400-800 ciclos), pero cacheada en L2 (y a veces L1)
- Se reserva con `cudaMalloc` y se accede con punteros normales

#### 3.8.5 Constant memory

- Memoria de solo lectura desde el kernel, escrita desde el host
- Tamaño: 64 KB total
- Muy rápida si **todos los threads del warp leen la misma dirección** (broadcast)
- Declaración:
    ```c
    __constant__ float coeffs[256];
    cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(coeffs));
    ```

#### 3.8.6 Texture / Surface memory

- Read-only desde el kernel, con cache espacial 2D
- Útil para acceso con localidad espacial (interpolación, imágenes)
- API moderna: `cudaTextureObject_t` (los antiguos `tex1D/tex2D` están deprecados)

#### 3.8.7 L1 / L2 cache (transparentes)

- L1 por SM, L2 global a la GPU
- No se programan, pero conocer su existencia ayuda a entender por qué a veces el reuso de datos es "gratis"
- En Volta+ la L1 y la shared comparten el mismo bloque on-chip (carveout configurable)

#### 3.8.8 Unified Memory (`cudaMallocManaged`)

- Puntero único accesible desde host y device, con migración automática de páginas
- Útil para prototipado o estructuras complejas; suele ser más lento que el manejo manual

#### 3.8.9 Tabla resumen

| Tipo | Scope | Lifetime | Latencia | Tamaño típico | On/off-chip | Qualifier |
|---|---|---|---|---|---|---|
| Registros | Thread | Thread | ~1 ciclo | ~64K/SM | On-chip | (ninguno) |
| Local | Thread | Thread | ~400-800 | Limitada | Off-chip | (auto) |
| Shared | Block | Block | ~20-30 | 48-228 KB/SM | On-chip | `__shared__` |
| Global | Grid | Aplicación | ~400-800 | GB | Off-chip | `__device__` |
| Constant | Grid | Aplicación | ~1 (hit) | 64 KB | Off-chip + $ | `__constant__` |
| Texture | Grid | Aplicación | Variable | — | Off-chip + $ | (objeto) |

### 3.9 Memoria del device (API)

#### 3.9.1 Reservar y liberar

```c
cudaMalloc(&devPtr, bytes);
```

Reserva memoria en la VRAM del device. El puntero resultante solo es válido desde el device (o como argumento a APIs CUDA), nunca desreferenciable desde el host.

```c
float* d_A;
cudaMalloc(&d_A, N * sizeof(float));   // reserva
```

| Parámetro | Significado |
|---|---|
| `&devPtr` | Dirección del puntero (se rellena con la dirección reservada) |
| `bytes` | Tamaño en bytes (`N * sizeof(tipo)`) |

```c
cudaFree(devPtr);
```

Libera memoria reservada previamente con `cudaMalloc` / `cudaMallocManaged`.

```c
cudaFree(d_A);
```

| Parámetro | Significado |
|---|---|
| `devPtr` | Puntero a liberar |

```c
cudaMallocHost(&hostPtr, bytes);
```

Reserva memoria **pinned** en host (no paginable). Transferencias H↔D mucho más rápidas y permite `cudaMemcpyAsync`.

```c
float* h_A;
cudaMallocHost(&h_A, N * sizeof(float));
```

| Parámetro | Significado |
|---|---|
| `&hostPtr` | Dirección del puntero del host |
| `bytes` | Tamaño en bytes |

```c
cudaMallocManaged(&ptr, bytes);
```

Reserva memoria **unified**: un único puntero accesible desde host y device, con migración automática de páginas.

```c
float* p;
cudaMallocManaged(&p, N * sizeof(float));
// se puede usar tanto desde host como desde kernel
```

| Parámetro | Significado |
|---|---|
| `&ptr` | Dirección del puntero |
| `bytes` | Tamaño en bytes |

#### 3.9.2 Copiar datos

```c
cudaMemcpy(dst, src, bytes, direction);
```

Copia síncrona entre host y device (o dentro del device). Bloquea al host hasta completar.

```c
cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
```

| Parámetro | Significado |
|---|---|
| `dst` | Destino |
| `src` | Origen |
| `bytes` | Tamaño en bytes |
| `direction` | Dirección de la copia (ver tabla) |

| Dirección | Significado |
|---|---|
| `cudaMemcpyHostToDevice` | CPU → GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU |
| `cudaMemcpyDeviceToDevice` | GPU → GPU |
| `cudaMemcpyHostToHost` | CPU → CPU |
| `cudaMemcpyDefault` | Inferida (requiere UVA) |

#### 3.9.3 Inicializar memoria

```c
cudaMemset(devPtr, value, bytes);
```

Pone `bytes` bytes a un valor (típicamente 0). Funciona a nivel de byte, **no de elemento** (por eso solo es útil para `0` o patrones byte-uniformes).

```c
cudaMemset(d_A, 0, N * sizeof(float));   // pone a cero
```

| Parámetro | Significado |
|---|---|
| `devPtr` | Puntero device |
| `value` | Byte a escribir (0–255) |
| `bytes` | Cuántos bytes escribir |

#### 3.9.4 Copia asíncrona (streams)

```c
cudaMemcpyAsync(dst, src, bytes, direction, stream);
```

Igual que `cudaMemcpy` pero no bloquea al host. Requiere memoria pinned en el host para ser realmente asíncrona.

```c
cudaMemcpyAsync(d_A, h_A, N * sizeof(float),
                cudaMemcpyHostToDevice, stream);
```

| Parámetro | Significado |
|---|---|
| `dst`, `src`, `bytes`, `direction` | Igual que `cudaMemcpy` |
| `stream` | Stream donde se encola la operación |

#### 3.9.5 `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol`

```c
cudaMemcpyToSymbol(symbol, src, bytes);
cudaMemcpyFromSymbol(dst, symbol, bytes);
```

Copian datos entre el host y una variable global del device (`__device__` o `__constant__`), identificándola por **nombre del símbolo**. Es el puente típico para inicializar `__constant__`.

```c
__constant__ float coeffs[256];
// ...
cudaMemcpyToSymbol(coeffs, h_coeffs, 256 * sizeof(float));
```

| Parámetro | Significado |
|---|---|
| `symbol` | Variable global del device (por nombre) |
| `src` / `dst` | Puntero en el host (origen en `To`, destino en `From`) |
| `bytes` | Bytes a copiar |
| `offset` *(opcional)* | Desplazamiento dentro del símbolo |
| `kind` *(opcional)* | Dirección de la copia |

### 3.10 Sincronización

#### 3.10.1 Device-side

```c
__syncthreads();
```

Barrera dentro del kernel: todos los threads del **mismo bloque** esperan aquí antes de seguir. Indispensable tras escribir en shared memory.

```c
tile[ty][tx] = global[idx];
__syncthreads();           // todos han escrito antes de leer
float v = tile[tx][ty];
```

| Aspecto | Detalle |
|---|---|
| Ámbito | Bloque |
| Coste | Bajo (~unos pocos ciclos) |
| Peligro | Si no la alcanzan todos los threads del bloque → **deadlock** |

```c
__syncwarp(mask);
```

Barrera dentro de un warp. En Volta+ es **necesaria explícitamente** cuando los threads del warp divergen y luego deben coordinarse.

```c
__syncwarp(0xffffffff);    // sincroniza los 32 threads del warp
```

| Parámetro | Significado |
|---|---|
| `mask` | Máscara de 32 bits: qué threads del warp participan (`0xffffffff` = todos) |

#### 3.10.2 Host-side

```c
cudaDeviceSynchronize();
```

Bloquea al host hasta que **toda** la cola de trabajo del device esté vacía. Útil para depurar y antes de medir tiempos.

```c
kernel<<<grid, block>>>(args);
cudaDeviceSynchronize();   // ahora seguro que el kernel ha terminado
```

| Aspecto | Detalle |
|---|---|
| Devuelve | `cudaError_t` (útil para detectar errores asíncronos) |
| Coste | Alto (bloquea host); evitar en hot path |

```c
cudaStreamSynchronize(stream);
```

Bloquea al host hasta que el stream concreto haya completado todo su trabajo. Más fino que `cudaDeviceSynchronize`.

```c
cudaStreamSynchronize(stream);
```

| Parámetro | Significado |
|---|---|
| `stream` | Stream a esperar |

#### 3.10.3 Reglas de oro

- `__syncthreads()` debe ser alcanzable por **todos** los threads del bloque (nunca dentro de un `if` divergente) → si no, deadlock
- Los lanzamientos de kernel son **asíncronos**: si necesitas esperar, sincroniza explícitamente
- Entre kernels lanzados al mismo stream hay **barrera implícita**: el segundo no empieza hasta que el primero acaba
- No hay sincronización entre bloques dentro de un kernel (sin `cooperative_groups`)

---

## Bloque C — Patrones de optimización

### 3.11 Shared memory: patrón tiling canónico

Idea: cada bloque carga un "tile" de datos de global a shared, sincroniza, opera reutilizando shared, sincroniza, escribe a global. Reduce drásticamente los accesos a global memory.

```c
__global__ void kernel(const float *A, float *B, int N) {
    __shared__ float tile[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE + tx;
    int gy = blockIdx.y * TILE + ty;

    // 1. Cargar de global → shared
    if (gx < N && gy < N) tile[ty][tx] = A[gy * N + gx];
    __syncthreads();

    // 2. Operar reutilizando shared
    // ... aquí cada thread puede leer cualquier elemento de tile ...

    __syncthreads();

    // 3. Escribir resultado a global
    if (gx < N && gy < N) B[gy * N + gx] = tile[ty][tx];
}
```

Pasos clave: cargar → sync → operar → sync → escribir.

### 3.12 Reduction en un bloque

Patrón clásico: cada thread carga un elemento, reducción en árbol dentro de shared memory.

```c
__global__ void reduceSum(const float *in, float *out, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}
```

Notas:
- Se llama dos veces si quieres reducir todo el vector (segundo kernel reduce los parciales)
- Versiones óptimas usan warp-level primitives (`__shfl_down_sync`) → para más adelante

### 3.13 Coalescing: reglas de oro

| Regla | Detalle |
|---|---|
| Threads consecutivos del warp deben leer **direcciones consecutivas** | Acceso "stride 1" |
| Tamaño del acceso alineado a 32/64/128 bytes | El hardware sirve 1 transacción en vez de 32 |
| 2D row-major → indexar como `A[i * N + j]` y poner `j` en `threadIdx.x` | Mantiene coalescing |
| Stride > 1 → mata el ancho de banda | Reordenar datos o transponer |

Anti-patrón típico:
```c
// MAL: thread t lee A[t * stride], stride grande → no coalesced
val = A[threadIdx.x * stride];

// BIEN: thread t lee A[t]
val = A[threadIdx.x];
```

### 3.14 Bank conflicts y padding

- La shared memory está dividida en **32 bancos**; cada banco sirve una palabra (4 bytes) por ciclo
- Si dos threads del warp acceden a **distintas palabras del mismo banco** → conflicto serializado
- Si acceden a la **misma palabra del mismo banco** → broadcast (no es conflicto)

Antipatrón clásico (matriz 32×32 en shared, acceso por columna):
```c
__shared__ float tile[32][32];   // columna entera cae en el mismo banco
float v = tile[threadIdx.x][0];  // ← 32-way bank conflict
```

Fix: padding de una columna extra:
```c
__shared__ float tile[32][33];   // +1 rompe el alineamiento
float v = tile[threadIdx.x][0];  // ← sin conflictos
```

### 3.15 Streams y solapamiento compute/transfer

Idea: ejecutar `cudaMemcpyAsync` y `kernel<<<...>>>` en streams distintos para solapar transferencias y cómputo.

```c
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, s1);
kernel<<<grid, block, 0, s2>>>(d_B);   // se solapa con la copia
cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, s1);

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);
```

Requisitos:
- Memoria host **pinned** (`cudaMallocHost`)
- Streams distintos (el stream 0 serializa todo)

---

## Bloque D — Compilación, profiling, debug

### 3.16 `nvcc`: flags por caso de uso

`nvcc` separa el código en host (lo pasa a `gcc`/`clang`/`MSVC`) y device (lo compila a PTX/SASS).

```bash
# Release (producción)
nvcc -O3 -arch=sm_80 prog.cu -o prog

# Debug (paso a paso en cuda-gdb)
nvcc -G -g -arch=sm_80 prog.cu -o prog

# Profile (Nsight Compute, sin overhead grande)
nvcc -O3 -lineinfo -arch=sm_80 prog.cu -o prog

# Ver uso de registros y shared por kernel
nvcc -O3 -arch=sm_80 -Xptxas -v prog.cu -o prog
```

| Flag | Para qué |
|---|---|
| `-arch=sm_XX` | Target de Compute Capability |
| `-O0` / `-O3` | Nivel de optimización |
| `-G` | Debug info en device (desactiva opts → mucho más lento) |
| `-g` | Debug info en host |
| `-lineinfo` | Mapea líneas de código sin desactivar opts (ideal para profiling) |
| `-Xcompiler "..."` | Pasa flags al compilador host |
| `-Xptxas -v` | Muestra registros y shared memory usados |
| `--use_fast_math` | Intrínsecas rápidas (menos precisas) |
| `-rdc=true` | Compilación separada (relocatable device code) |

### 3.17 Arch mapping

| `-arch=` | Arquitectura | GPU típica |
|---|---|---|
| `sm_60` | Pascal | P100 |
| `sm_70` | Volta | V100 |
| `sm_75` | Turing | T4, RTX 2080 |
| `sm_80` | Ampere | A100 |
| `sm_86` | Ampere (consumer) | RTX 3090 |
| `sm_89` | Ada Lovelace | RTX 4090 |
| `sm_90` | Hopper | H100 |
| `sm_100` | Blackwell | B100/B200 |

### 3.18 Medir tiempos con `cudaEvent`

```c
cudaEventCreate(&event);
cudaEventRecord(event, stream);
cudaEventSynchronize(event);
cudaEventElapsedTime(&ms, start, stop);
```

Mide tiempos de ejecución en GPU con precisión de microsegundos. Es la forma correcta de medir kernels (no uses temporizadores de CPU).

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

| Función | Significado |
|---|---|
| `cudaEventCreate(&e)` | Crea el evento |
| `cudaEventRecord(e, stream)` | Marca el evento en el stream (0 por defecto) |
| `cudaEventSynchronize(e)` | Espera a que el evento se haya registrado |
| `cudaEventElapsedTime(&ms, s, e)` | Tiempo en milisegundos entre dos eventos |
| `cudaEventDestroy(e)` | Libera el evento |

### 3.19 Consultar propiedades del device

```c
cudaGetDeviceProperties(&prop, deviceId);
```

Rellena una estructura con las propiedades del device: nombre, compute capability, memoria, número de SMs, tamaños máximos, etc.

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("GPU: %s\n", prop.name);
printf("  CC: %d.%d\n", prop.major, prop.minor);
printf("  SMs: %d\n", prop.multiProcessorCount);
printf("  Mem: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
printf("  Shared/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
```

| Parámetro | Significado |
|---|---|
| `&prop` | Estructura `cudaDeviceProp` a rellenar |
| `deviceId` | Id de la GPU (0, 1, …) |

| Función | Para qué |
|---|---|
| `cudaGetDeviceCount(&n)` | Cuántas GPUs hay |
| `cudaSetDevice(id)` | Selecciona GPU activa |
| `cudaGetDevice(&id)` | Devuelve la GPU activa actual |

---

## Bloque E — Errores y troubleshooting

### 3.21 Macro `CUDA_CHECK`

```c
#define CUDA_CHECK(call) do {                               \
    cudaError_t e = (call);                                 \
    if (e != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)
```

Uso:
```c
CUDA_CHECK(cudaMalloc(&d_A, bytes));
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
```

Las funciones CUDA devuelven `cudaError_t`. Los errores **asíncronos** (en kernels) no se ven hasta que sincronizas.

### 3.22 Comprobar errores tras un kernel

```c
miKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // error de lanzamiento (sync)
CUDA_CHECK(cudaDeviceSynchronize());   // error de ejecución (async)
```

| Función | Para qué |
|---|---|
| `cudaGetLastError()` | Devuelve y resetea el último error |
| `cudaPeekAtLastError()` | Devuelve sin resetear |
| `cudaGetErrorString(err)` | String legible del error |

### 3.23 Bestiario de errores frecuentes

| Síntoma | Causa probable | Fix |
|---|---|---|
| Kernel "no hace nada" | Olvidado el `cudaMemcpy` device→host | Verificar copias |
| `invalid configuration argument` | `blockDim > 1024` o grid mal calculado | Revisar dimensiones |
| `invalid device pointer` | Pasar puntero host a kernel (o al revés) | Comprobar de qué lado vive |
| Resultados raros y no deterministas | Race condition (escrituras sin sync) | Usar `__syncthreads` o repensar |
| `illegal memory access` | Falta `if (i < N)` o índice mal calculado | Añadir bounds check |
| Programa cuelga | `__syncthreads` dentro de `if` divergente | Sacar barrera fuera del `if` |
| Errores que aparecen "después" | Error asíncrono no detectado a tiempo | Sincronizar y chequear |
| Todo lentísimo | `cudaMemcpy` en bucle, o `-G` activado | Sacar copias del loop, compilar `-O3` |
| Resultados correctos en CPU pero no en GPU | Acceso fuera de rango sobreescribe | Compilar con `compute-sanitizer` |

### 3.24 Antipatrones de rendimiento

| Antipatrón | Por qué duele | Alternativa |
|---|---|---|
| Stride no-1 en lecturas globales | Mata coalescing → ancho de banda destrozado | Reordenar datos / transponer |
| `cudaMalloc/cudaFree` en hot path | Es muy lento; serializa con el device | Reservar una vez, reutilizar |
| Lanzar kernels muy pequeños en bucle | Latencia de lanzamiento domina | Fusionar kernels o usar grid-stride |
| Compilar con `-G` para benchmarks | Desactiva optimizaciones | Usar `-lineinfo` para profile |
| Transferencias H↔D dentro del loop | PCIe es lento | Mover datos una sola vez |
| Shared memory con stride 32 | 32-way bank conflict | Padding `[N][N+1]` |
| Branches divergentes en hot path | Serializa el warp | Reestructurar para que threads del mismo warp vayan por la misma rama |

---
