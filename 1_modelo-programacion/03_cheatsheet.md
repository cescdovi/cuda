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

    // reservar memoria en DEVICE
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //copiar datos HOST -> DEVICE
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // ejecutar kernel
    int blockDim = 256;
    int nblocks  = (n + blockDim - 1) / blockDim;
    kernel<<<nblocks, blockDim>>>(n, d_a, d_b, d_c);

    // copiar DEVICE --> HOST
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // liberar memoria
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

> **Convención de ejes (pantalla, no matricial):** `.x` = columna (horizontal), `.y` = fila (vertical), `.z` = profundidad. Origen arriba-izquierda, `.y` crece hacia abajo. Por eso al indexar row-major es `A[fila * W + columna]` → `A[y * W + x]`. Ver [02_modelo-programacion.md](02_modelo-programacion.md#L193) §2.3.3 para el detalle.

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

La API de memoria de CUDA cubre tres bloques: **reservar/liberar** memoria en device, host pinned o unified; **mover datos** entre host y device (síncrono o asíncrono); e **inicializar** regiones de memoria. Todas las funciones devuelven `cudaError_t` y deben envolverse en `CUDA_CHECK`.

> **Convención de nombres**: por claridad se suele prefijar `d_` los punteros device (`d_A`) y `h_` los punteros host (`h_A`). El compilador no lo impone, pero **mezclarlos es la fuente #1 de errores `invalid device pointer`** (un puntero host pasado a un kernel, o desreferenciar un puntero device desde la CPU).

#### 3.9.1 Reservar y liberar

##### `cudaMalloc` — reservar en VRAM

```c
cudaMalloc(&devPtr, bytes);
```

Reserva un bloque contiguo de **memoria global** (VRAM) en el device activo. El puntero resultante vive en el espacio de direcciones del device:

- Es válido como argumento a kernels y a otras APIs CUDA (`cudaMemcpy`, `cudaFree`...).
- **NO** se puede desreferenciar desde el host: hacerlo causa segfault, no error CUDA.
- La memoria **no está inicializada** (contiene basura): si necesitas ceros, usa `cudaMemset` o `cudaMemcpy` después.
- La reserva es **síncrona** y relativamente cara (~µs): nunca llamar a `cudaMalloc` dentro del *hot path*; reservar una vez y reutilizar.

```c
float *d_A;
CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));   // reserva N floats en VRAM
// ❌ printf("%f", d_A[0]);   // segfault: d_A no es desreferenciable desde host
// ✅ cudaMemcpy(h_A, d_A, ...) para leer
```

| Parámetro | Significado |
|---|---|
| `&devPtr` | Dirección del puntero (se rellena con la dirección reservada) |
| `bytes` | Tamaño en bytes (`N * sizeof(tipo)`) |

##### `cudaFree` — liberar memoria del device

```c
cudaFree(devPtr);
```

Libera memoria reservada previamente con `cudaMalloc` o `cudaMallocManaged`:

- Es **síncrono**: bloquea al host hasta liberar.
- Pasar `nullptr` es seguro (no-op), igual que `free` en C.
- **No** liberar memoria reservada con `cudaMallocHost` (para eso está `cudaFreeHost`).
- Doble free → error CUDA, no UB.

```c
CUDA_CHECK(cudaFree(d_A));
d_A = nullptr;                  // buena práctica: invalidar el puntero
```

| Parámetro | Significado |
|---|---|
| `devPtr` | Puntero a liberar |

##### `cudaMallocHost` — memoria pinned en host

```c
cudaMallocHost(&hostPtr, bytes);
```

Reserva memoria en el host marcada como **pinned** (page-locked): el SO no puede paginarla a disco. Tiene dos efectos clave:

- Las transferencias `cudaMemcpy` H↔D son **2-3× más rápidas** (el driver puede usar DMA directo sin buffer intermedio).
- Es **requisito** para que `cudaMemcpyAsync` sea realmente asíncrono (con memoria paginable, `cudaMemcpyAsync` se degrada a copia síncrona).

Tiene también un coste: reduce la memoria paginable disponible al SO. Reservar gigabytes pinned puede degradar todo el sistema. Liberar con `cudaFreeHost`, no con `free`.

```c
float *h_A;
CUDA_CHECK(cudaMallocHost(&h_A, N * sizeof(float)));   // pinned: rápido + async-ready
// ... usar h_A como un puntero normal del host ...
CUDA_CHECK(cudaFreeHost(h_A));
```

| Parámetro | Significado |
|---|---|
| `&hostPtr` | Dirección del puntero del host |
| `bytes` | Tamaño en bytes |

##### `cudaMallocManaged` — Unified Memory

```c
cudaMallocManaged(&ptr, bytes);
```

Reserva memoria **unified**: un único puntero accesible desde host y device, con migración automática de páginas gestionada por el driver. Cuando el host accede, las páginas se migran a RAM; cuando el device accede, se migran a VRAM.

- **Ventaja**: simplifica el código drásticamente (no `cudaMemcpy` explícitos, ideal para prototipado o estructuras complejas con punteros).
- **Coste**: cada page fault dispara una migración (latencia alta). En kernels optimizados suele ser **más lento** que `cudaMalloc + cudaMemcpy` manual.
- **Requiere sincronización**: tras un lanzamiento de kernel, llamar a `cudaDeviceSynchronize()` antes de leer desde host para evitar race conditions con la migración.

```c
float *p;
CUDA_CHECK(cudaMallocManaged(&p, N * sizeof(float)));

for (int i = 0; i < N; i++) p[i] = i;    // host escribe → páginas en RAM
kernel<<<grid, block>>>(p, N);            // kernel accede → migración automática a VRAM
CUDA_CHECK(cudaDeviceSynchronize());     // imprescindible antes de leer desde host
printf("%f\n", p[0]);                    // host lee → migración de vuelta a RAM

CUDA_CHECK(cudaFree(p));
```

| Parámetro | Significado |
|---|---|
| `&ptr` | Dirección del puntero |
| `bytes` | Tamaño en bytes |

##### Tabla comparativa de reservas

| Función | Dónde vive | Acceso host | Acceso device | Velocidad H↔D | Cuándo usar |
|---|---|---|---|---|---|
| `cudaMalloc` | VRAM | ❌ | ✓ directo | — (requiere copia) | Caso por defecto: máximo control |
| `cudaMallocHost` | RAM pinned | ✓ directo | ❌ (requiere copia) | 🚀 Muy rápida | Buffers de transferencia, streams |
| `cudaMallocManaged` | Unified (migra) | ✓ con migración | ✓ con migración | Automática (con page faults) | Prototipado, estructuras con punteros |

#### 3.9.2 Copiar datos

```c
cudaMemcpy(dst, src, bytes, direction);
```

Copia **síncrona** de datos entre regiones de memoria. Bloquea al host hasta que la copia termine. Es la primitiva de transferencia más usada:

- La dirección de la copia debe declararse explícitamente con un enum `cudaMemcpyKind` (ver tabla inferior).
- Las copias H↔D viajan por el **bus PCIe** (~16-32 GB/s en PCIe 4.0): mucho más lento que el ancho de banda de VRAM (~1-3 TB/s). Por eso la regla es **transferir lo mínimo, una sola vez, fuera del hot path**.
- Las copias D→D son intra-VRAM y por tanto rapidísimas.
- Si `dst` y `src` se solapan → comportamiento indefinido (igual que `memcpy` de C).

```c
// H → D: subir input a la GPU antes de lanzar el kernel
CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

kernel<<<grid, block>>>(d_A, d_B, N);

// D → H: bajar resultado a la CPU
CUDA_CHECK(cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost));
```

| Parámetro | Significado |
|---|---|
| `dst` | Puntero al destino |
| `src` | Puntero al origen |
| `bytes` | Tamaño en bytes |
| `direction` | Dirección de la copia (ver tabla) |

| Dirección | Significado | Velocidad típica |
|---|---|---|
| `cudaMemcpyHostToDevice` | CPU → GPU (vía PCIe) | ~16-32 GB/s |
| `cudaMemcpyDeviceToHost` | GPU → CPU (vía PCIe) | ~16-32 GB/s |
| `cudaMemcpyDeviceToDevice` | GPU → GPU (intra-VRAM) | ~1-3 TB/s |
| `cudaMemcpyHostToHost` | CPU → CPU (equivalente a `memcpy`) | RAM bandwidth |
| `cudaMemcpyDefault` | Inferida desde los punteros (requiere UVA) | Según ruta real |

> **Regla de oro**: cada `cudaMemcpy` H↔D es muy caro. Si tu kernel tarda 0.1 ms y tu copia 10 ms, la GPU está al 1 % de utilización. Reorganiza el pipeline para amortizar las copias entre muchos kernels.

#### 3.9.3 Inicializar memoria

```c
cudaMemset(devPtr, value, bytes);
```

Pone los primeros `bytes` bytes de `devPtr` al valor `value`. Operación equivalente a `memset` de C, pero en VRAM. **Atención al gotcha clásico**:

- `value` se interpreta como **byte** (0-255), no como elemento del tipo de dato.
- Por tanto solo es útil para patrones byte-uniformes: `0`, `0xFF` (todos los bits a 1), o constantes con todos los bytes iguales.
- `cudaMemset(d_A, 1, N * sizeof(float))` **NO** pone los floats a `1.0f` — pone cada byte a `0x01`, lo que en floats resulta en `2.36e-38`, no `1.0`.
- Para inicializar a valores no-uniformes: usa `cudaMemcpy` desde un buffer de host o lanza un kernel de inicialización.

```c
// ✅ correcto: poner a cero (todos los bytes 0 = float 0.0f, int 0)
CUDA_CHECK(cudaMemset(d_A, 0, N * sizeof(float)));

// ❌ incorrecto: NO pone los floats a 1.0
CUDA_CHECK(cudaMemset(d_A, 1, N * sizeof(float)));

// ✅ para valores arbitrarios: kernel de init
initKernel<<<grid, block>>>(d_A, 1.0f, N);
```

| Parámetro | Significado |
|---|---|
| `devPtr` | Puntero device a inicializar |
| `value` | Byte a escribir (0–255) |
| `bytes` | Cuántos bytes escribir |

#### 3.9.4 Copia asíncrona (streams)

```c
cudaMemcpyAsync(dst, src, bytes, direction, stream);
```

Versión no-bloqueante de `cudaMemcpy`: encola la copia en un stream y devuelve el control al host inmediatamente. Es la pieza clave para **solapar transferencias con cómputo** (ver §3.15).

- Requiere memoria **pinned** en el host (`cudaMallocHost`); con memoria paginable se degrada silenciosamente a copia síncrona.
- Tras lanzarla, el buffer de origen **no se puede modificar** ni el de destino leer hasta que la copia termine. Sincronizar con `cudaStreamSynchronize` o `cudaEvent`.
- Si se usa el stream `0` (stream por defecto), la copia se serializa con todo lo demás → no hay solapamiento real.

```c
cudaStream_t s;
CUDA_CHECK(cudaStreamCreate(&s));

// Lanza copia y kernel en paralelo (en streams distintos)
CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, s));
kernel<<<grid, block, 0, s>>>(d_A, d_B);   // mismo stream s → orden garantizado
CUDA_CHECK(cudaMemcpyAsync(h_B, d_B, bytes, cudaMemcpyDeviceToHost, s));

CUDA_CHECK(cudaStreamSynchronize(s));      // esperar a que todo termine
CUDA_CHECK(cudaStreamDestroy(s));
```

| Parámetro | Significado |
|---|---|
| `dst`, `src`, `bytes`, `direction` | Igual que `cudaMemcpy` |
| `stream` | Stream donde se encola la operación (≠ 0 para solape real) |

#### 3.9.5 `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol`

```c
cudaMemcpyToSymbol(symbol, src, bytes);
9;
```

Copian datos entre el host y una **variable global del device** declarada con `__device__` o `__constant__`, identificándola por **nombre del símbolo** (no por puntero). Necesario porque las variables `__constant__` y `__device__` viven en espacios de memoria especiales a los que el host no tiene puntero directo.

- Es el puente típico para **inicializar `__constant__`**: parámetros de filtros, coeficientes, lookup tables pequeñas.
- El `symbol` se pasa por nombre, sin `&`: el compilador lo resuelve a la dirección device correcta.
- Por defecto es síncrono; hay variantes `Async` que reciben un `stream`.
- Las versiones modernas también aceptan un puntero device obtenido vía `cudaGetSymbolAddress`, pero el patrón clásico es pasar el símbolo directamente.

```c
__constant__ float coeffs[256];          // vive en constant memory del device

void initFilter(const float *h_coeffs) {
    // Host → constant memory del device
    CUDA_CHECK(cudaMemcpyToSymbol(coeffs, h_coeffs, 256 * sizeof(float)));
}

__global__ void applyFilter(float *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] *= coeffs[i % 256];   // lectura por broadcast, ~1 ciclo
}
```

| Parámetro | Significado |
|---|---|
| `symbol` | Variable global del device (por nombre, sin `&`) |
| `src` / `dst` | Puntero en el host (origen en `To`, destino en `From`) |
| `bytes` | Bytes a copiar |
| `offset` *(opcional)* | Desplazamiento en bytes dentro del símbolo |
| `kind` *(opcional)* | Dirección de la copia (default: la inferida según `To`/`From`) |

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
