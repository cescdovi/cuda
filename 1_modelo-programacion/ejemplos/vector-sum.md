# 1 Flujo general del Ejercicio 1: Suma de vectores en CUDA

## 1.1 Definir datos de entrada (en CPU)

```
n = 10
h_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
h_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
h_c = [?, ?, ?, ?, ?, ?, ?, ?, ?, ?]   ← donde se almacenará el resultado
```

**Resultado esperado (suma elemento a elemento):**

```
h_c = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0]
```

> A diferencia del producto escalar, aquí **el tamaño de la salida coincide con el tamaño de la entrada**. No hay reducción: cada elemento de salida depende de un único par de elementos de entrada. Esto hace que el reparto del trabajo sea trivial: **un thread por elemento**.

---

## 1.2 Reservar memoria en GPU

```cuda
cudaMalloc(&d_a, 10 * sizeof(float));   // 40 bytes
cudaMalloc(&d_b, 10 * sizeof(float));   // 40 bytes
cudaMalloc(&d_c, 10 * sizeof(float));   // 40 bytes (resultado)
```

**Estado de la memoria GPU tras este paso:**

```
d_a = [?, ?, ?, ?, ?, ?, ?, ?, ?, ?]   ← basura, sin inicializar
d_b = [?, ?, ?, ?, ?, ?, ?, ?, ?, ?]   ← basura
d_c = [?, ?, ?, ?, ?, ?, ?, ?, ?, ?]   ← basura
```

> **Diferencia clave con el producto escalar**: aquí los tres vectores GPU tienen tamaño `n`. No hay vector intermedio ni resultado escalar, porque cada elemento de salida se calcula independientemente.

---

## 1.3 Copiar datos CPU → GPU

```cuda
cudaMemcpy(d_a, h_a, 10*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, 10*sizeof(float), cudaMemcpyHostToDevice);
```

> `d_c` **no se copia**: es el vector de salida, lo va a rellenar el kernel. Solo se reserva memoria para él.

**Estado de la memoria GPU:**

```
d_a = [1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0]
d_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
d_c = [?, ?, ?, ?, ?, ?, ?, ?, ?, ?]   ← sigue con basura
```

---

## 1.4 Lanzar el Kernel

### 1.4.1 Cálculo del nº de bloques

A diferencia del producto escalar (donde el número de threads era fijo en 32), aquí necesitamos **al menos un thread por elemento del vector**. Como `blockDim` suele ser un valor cómodo (potencia de 2, múltiplo de 32), calculamos cuántos bloques hacen falta para cubrir `n` elementos:

```cuda
int blocksize = 32;
int nblocks = (n + blocksize - 1) / blocksize;   // ceil(n / blocksize)
```

Para `n = 10` y `blocksize = 32`:

```
nblocks = (10 + 31) / 32 = 41 / 32 = 1   (división entera)
```

Es decir, **basta con 1 bloque de 32 threads** para cubrir los 10 elementos (de hecho sobran 22 threads, que no harán nada útil).

### 1.4.2 Sintaxis y nº de threads

```cuda
dim3 dimGrid(nblocks);       // (1)
dim3 dimBlock(blocksize);    // (32)
compute_kernel<<<dimGrid, dimBlock>>>(n, d_a, d_b, d_c);
```

- **Nº total de threads lanzados** = `gridDim × blockDim = 1 × 32 = 32`.
- Sobran `32 - 10 = 22` threads → necesitarán protección con `if (idx < n)` para no escribir fuera de rango.

> Esta línea se escribe **una sola vez** desde el host, pero lanza **32 ejecuciones simultáneas** del kernel en la GPU. Cada ejecución es un thread con su propio `threadIdx.x` (de 0 a 31) y `blockIdx.x` (siempre 0 en este caso).

### 1.4.3 Código del kernel

```cuda
__global__ void compute_kernel(const unsigned int n, float *d_a, float *d_b, float *d_c) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;   // identidad global
    if (idx < n) {                                      // protección de límites
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}
```

> **Idea clave**: cada thread procesa **exactamente un elemento** del vector. Como `idx` coincide con la posición del elemento, no hace falta bucle ni stride. El reparto del trabajo es one-to-one entre threads e índices.

### 1.4.4 Traza del Thread 0 (threadIdx.x = 0, blockIdx.x = 0)

**Estado inicial:**
```
Thread 0:  threadIdx.x = 0,  blockIdx.x = 0,  blockDim.x = 32
```

**Cálculo del índice global:**
```
idx = 0 + 32 * 0 = 0
```

**Comprobación de límites:**
```
¿0 < 10? SÍ → entra al cuerpo
```

**Ejecución:**
```
Lee d_a[0] = 1.0
Lee d_b[0] = 10.0
Calcula 1.0 + 10.0 = 11.0
Escribe d_c[0] = 11.0
```

### 1.4.5 Traza del Thread 1 (threadIdx.x = 1)

```
idx = 1 + 32 * 0 = 1
¿1 < 10? SÍ
Lee d_a[1] = 2.0
Lee d_b[1] = 20.0
Calcula 2.0 + 20.0 = 22.0
Escribe d_c[1] = 22.0
```

### 1.4.6 Trazas de los Threads 2 a 9 (resumidas)

Cada uno procesa exactamente un elemento:

| Thread | threadIdx.x | idx | d_a[idx] | d_b[idx] | suma | Escribe |
|---|---|---|---|---|---|---|
| 2 | 2 | 2 | 3.0 | 30.0 | 33.0 | `d_c[2] = 33.0` |
| 3 | 3 | 3 | 4.0 | 40.0 | 44.0 | `d_c[3] = 44.0` |
| 4 | 4 | 4 | 5.0 | 50.0 | 55.0 | `d_c[4] = 55.0` |
| 5 | 5 | 5 | 6.0 | 60.0 | 66.0 | `d_c[5] = 66.0` |
| 6 | 6 | 6 | 7.0 | 70.0 | 77.0 | `d_c[6] = 77.0` |
| 7 | 7 | 7 | 8.0 | 80.0 | 88.0 | `d_c[7] = 88.0` |
| 8 | 8 | 8 | 9.0 | 90.0 | 99.0 | `d_c[8] = 99.0` |
| 9 | 9 | 9 | 10.0 | 100.0 | 110.0 | `d_c[9] = 110.0` |

### 1.4.7 Traza del Thread 10 (threadIdx.x = 10)

```
idx = 10 + 32 * 0 = 10
¿10 < 10? NO → NO entra al cuerpo
```

**Resultado:** no hace nada. No escribe en `d_c`.

> **Aquí está el porqué del `if (idx < n)`**: sin esa comprobación, el Thread 10 escribiría en `d_c[10]`, que es una posición **fuera del vector** (memoria no reservada). Esto provocaría un acceso ilegal o, peor aún, sobrescribir memoria de otro vector → resultados aleatorios y muy difíciles de depurar.

### 1.4.8 Trazas de los Threads 11 a 31

Todos hacen lo mismo que el Thread 10: su `idx` es ≥ 10, así que la condición `if (idx < n)` los descarta. **No escriben en ningún sitio**.

```
Thread 11: idx = 11 → fuera de rango → no hace nada
Thread 12: idx = 12 → fuera de rango → no hace nada
...
Thread 31: idx = 31 → fuera de rango → no hace nada
```

### 1.4.9 Estado de la memoria GPU tras el Kernel

```
d_a = [1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0]    ← sin cambios
d_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]   ← sin cambios

d_c = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0]
       └────────────────────────────────────────────────────────┘
                  los 10 threads "útiles" rellenan d_c[0..9]
```

> **Punto clave**: el kernel ha terminado. Solo hay UN kernel en este ejercicio (a diferencia del producto escalar, que necesitaba dos kernels en cascada por la reducción). Como cada elemento de salida es independiente, no hay necesidad de fases ni comunicación entre threads.

---

## 1.5 Copiar resultado GPU → CPU

```cuda
cudaMemcpy(h_c, d_c, 10*sizeof(float), cudaMemcpyDeviceToHost);
// h_c = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0]
```

> **Diferencia con el producto escalar**: aquí copias **todo el vector** (10 elementos) de vuelta a CPU, no un solo escalar. La copia es de tamaño `n * sizeof(float)`.

---

## 1.6 Liberar memoria

```cuda
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

---

## 1.7 Resumen del flujo completo

| Paso | Quién ejecuta | Qué hace | Estado clave después |
|---|---|---|---|
| 1 | Host (CPU) | Define `h_a`, `h_b`, `h_c`, `n` | Datos preparados en CPU |
| 2 | Host | `cudaMalloc` × 3 | Memoria GPU reservada (con basura) |
| 3 | Host | `cudaMemcpy` × 2 (H→D) | `d_a` y `d_b` con datos de entrada |
| 4 | 32 threads GPU | `compute_kernel<<<1, 32>>>` | `d_c = [11, 22, 33, ..., 110]` |
| 5 | Host | `cudaMemcpy` (D→H) | `h_c` con el resultado en CPU |
| 6 | Host | `cudaFree` × 3 | Memoria GPU liberada |

---

## 1.8 Comparación con el producto escalar

Para fijar las diferencias entre los dos patrones:

| Aspecto | Suma de vectores (Ej. 1) | Producto escalar (Ej. 5) |
|---|---|---|
| Tamaño de la salida | `n` (mismo que entrada) | 1 (escalar) |
| Tipo de operación | Mapeo (map) | Reducción (reduce) |
| Threads por elemento | 1 thread = 1 elemento | 1 thread = varios elementos (stride) |
| Nº de threads totales | `≥ n` (típicamente `ceil(n/blocksize) × blocksize`) | Fijo: 32 |
| Nº de kernels necesarios | 1 | 2 (en cascada) |
| Vector intermedio | No hace falta | `d_v` de 32 elementos |
| Protección de límites | `if (idx < n)` | `for (i = t; i < n; i += 32)` |
| Bucle en el kernel | No | Sí (grid-stride loop) |
| Comunicación entre threads | No hay | A través de `d_v` (memoria global) |

---

## 1.9 Ideas clave para recordar

- **`<<<nblocks, blocksize>>>` lanza `nblocks × blocksize` threads en paralelo**. Como queremos **al menos un thread por elemento**, calculamos `nblocks = ceil(n / blocksize)`.
- **Cada thread se identifica con `idx = threadIdx.x + blockDim.x * blockIdx.x`** y usa ese número como índice del elemento que le toca procesar. Es el patrón **one-to-one**: 1 thread = 1 elemento.
- **El `if (idx < n)` es obligatorio**: como `nblocks × blocksize` suele ser mayor que `n` (los threads sobrantes en el último bloque), hay que proteger los accesos a memoria para no escribir fuera del vector.
- **No hay bucle en el kernel**: como cada thread procesa un único elemento, basta una instrucción `d_c[idx] = d_a[idx] + d_b[idx]`. Esto contrasta con el producto escalar, donde cada thread sí necesitaba un bucle (grid-stride loop).
- **No hay vector intermedio ni segundo kernel**: la suma de vectores es un patrón de "mapeo" puro, sin reducción. Cada elemento de salida es independiente, así que un único kernel basta.
- **Coalescencia automática**: los 32 threads del warp acceden a `d_a[0..31]`, `d_b[0..31]` y escriben en `d_c[0..31]`, todo contiguo en memoria → **1 transacción por load/store**, eficiencia del 100%.