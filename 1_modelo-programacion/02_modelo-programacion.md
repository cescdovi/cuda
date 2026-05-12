# 2 Modelo de programación de CUDA

## 2.1 Idea general: abstracción sobre el hardware, escalabilidad

El modelo de programación es la **abstracción** que CUDA ofrece para aplicar paralelismo sin atarse al hardware concreto.

Tú declaras trabajo en términos lógicos (threads, bloques, grids) y CUDA lo mapea automáticamente a los SMs disponibles.

### 2.1.1 La idea clave

En vez de asignar trabajo a cores específicos, **declaras mucho trabajo independiente** y CUDA lo distribuye:

- Divides el problema en **bloques independientes** entre sí.
- Cada bloque se subdivide en **threads** que pueden cooperar.

Permite que CUDA puede ejecutarlos en cualquier orden, en paralelo o en serie, según los SMs disponibles.

### 2.1.2 Escalabilidad automática

El mismo binario corre en GPUs distintas, aprovechando los SMs disponibles:

```
GPU con 2 SMs → 4 bloques por SM (en serie)
GPU con 4 SMs → 2 bloques por SM
GPU con 8 SMs → 1 bloque por SM (todo en paralelo)
```

### 2.1.3 Características importantes

- **Abstracción**: programas en términos lógicos, no en términos de cores físicos.
- **Independencia de bloques**: requisito obligatorio para que la escalabilidad funcione.
    - CUDA reparte los bloques entre los SMs disponibles sin garantizar nada sobre el orden
    - Analogía: pizzeria debe preparar 80 pizzas: 8 bandejas (blocks) de 10 pizzas (threads) cada una, cada bandeja es del mismo tipo (barbacoa, margarita...): las bandejas se deben asignar a los cocineros en bloque (un mismo cocinero prepara una tanda entera)
- **Portabilidad**: mismo código fuente sirve para una GPU pequeña y una H100.
- **Granularidad jerárquica**: cooperación barata dentro del bloque, ninguna entre bloques.

## 2.2 Kernels

### 2.2.1 Qué es un kernel

Un **kernel** es una función C/C++ que se ejecuta en el device (GPU) y que se lanza **muchas veces en paralelo**, una vez por cada thread del grid.

La idea: lo que en CPU sería un bucle `for` de N iteraciones, en CUDA se convierte en un kernel lanzado por N threads simultáneos, cada uno operando sobre un elemento distinto.

```c
// CPU: bucle secuencial
for (int i = 0; i < N; i++)
    C[i] = A[i] + B[i];

// GPU: el bucle desaparece, cada thread hace UNA suma
__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
```

#### Características importantes

- Devuelve siempre `void` (los resultados se escriben en memoria del device).
- Cada thread tiene un identificador único (`threadIdx`) que usa para decidir sobre qué dato operar.
- Su lanzamiento es **asíncrono**: el host no espera a que termine, sigue ejecutando el código siguiente.
    - Para esperar al kernel se usa `cudaDeviceSynchronize()`.

- `__global__`: marca la función como kernel
    - la **llama el host pero la ejecuta el device**.

Existen tres especificadores relacionados:
| Especificador | Llamada | Ejecución | Uso |
|---|---|---|---|
| `__global__` | Host | Device | Definir que una función es un kernel |
| `__device__` | Device | Device | Funciones auxiliares dentro de kernels a ejecutar en el device |
| `__host__` | Host | Host | Función normal de CPU que se debe ejecutar en la CPU |

#### Sintaxis de lanzamiento `<<<grid, block>>>`

Desde el host, indicando cuántos threads se lanzan y cómo se organizan:

```c
VecAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
```

- **`numBlocks`**: cuántos bloques tiene el grid.
- **`threadsPerBlock`**: nº de threads por bloque.
    - limitado a **1024 threads** (límite hardware).
    - debe ser múltiplo de 32 para que se alinee con los warps
- Total de threads = `numBlocks × threadsPerBlock`.

Ambos parámetros pueden ser `int` (1D) o `dim3` (1D, 2D o 3D):

```c
VecAdd<<<1, N>>>(A, B, C);   // 1 bloque, N threads (1D)
VecAdd<<<8, 256>>>(A, B, C); // 8 bloques de 256 threads
                             // (2048 threads totales)

dim3 grid(4, 4); // grid 2D de 4×4 bloques
dim3 block(16, 16); // bloques 2D de 16×16 threads
MatrixOp<<<grid, block>>>(A, B, C);// útil para matrices
```
## 2.3 Jerarquía de threads

### 2.3.1 Thread → Block → Grid (1D/2D/3D)

CUDA organiza los threads en una **jerarquía de tres niveles**, cada uno con un rol distinto:

- **Thread**: la unidad mínima de ejecución, ejecuta el código del kernel sobre un dato.
- **Block**: grupo de threads que **pueden cooperar** entre sí (compartir memoria rápida, sincronizarse).
    - Un block corresponde a un SM completo
    - Todos los threads de un bloque viven en el mismo SM.

- **Grid**: conjunto de todos los bloques de un kernel.
    - Corresponde al device completo
    - Los bloques son **independientes** entre sí.

- **Warp**: unidad mínima de ejecución formada por 32 hilos (threads) que ejecutan la misma instrucción simultáneamente sobre datos distintos (modelo SIMT).
    - avanzan en lockstep: sincronizados paso a paso, ejecutando exactamente la misma instrucción en el mismo ciclo de reloj.

![alt text](images/thread-hierachy.png)
#### Memoria por nivel: qué se comparte y qué no
| Memoria | Nivel | Ámbito | Latencia | Uso típico |
|---|---|---|---|---|
| **Registers** | Thread | Privada del thread (nadie más la ve) | ~1 ciclo | Variables locales |
| **Local memory** | Thread | Privada del thread (vive en global físicamente) | ~400 ciclos | Spilling de registros |
| **Shared memory** (`__shared__`) | Block | Compartida entre todos los threads del **mismo bloque** | ~10-20 ciclos | Cooperación, reuso de datos |
| **Global memory** | Grid | Compartida entre **todos los threads** del grid, persistente | ~400-800 ciclos | Datos de entrada/salida |
| **Constant memory** | Grid | Visible por todos, **solo lectura** | ~varía (cached) | Constantes globales |
| **Texture memory** | Grid | Visible por todos, solo lectura, cache espacial 2D/3D | ~varía (cached) | Acceso con localidad espacial |

![alt text](images/gpu-architecture.png)

Consecuencia práctica: si dos threads necesitan cooperar, mira en qué nivel viven:

- **Mismo bloque** → usa shared memory (barato y rápido) y se pueden sincronizar
    - `__shared__` y `__syncthreads()`
- **Bloques distintos** → solo a través de global memory + atómicos (lento, evítalo cuando puedas) o cortando en dos kernels.

#### Dimensionalidad: 1D, 2D o 3D

Tanto el grid como los bloques pueden tener 1, 2 o 3 dimensiones. Es solo una conveniencia notacional para que el código case con la forma natural del problema:

| Dimensión | Caso de uso típico |
|---|---|
| 1D | Vectores, secuencias |
| 2D | Matrices, imágenes |
| 3D | Volúmenes, simulaciones espaciales |

### 2.3.2 Variables built-in

Dentro de un kernel, CUDA expone **cuatro variables built-in** que cada thread usa para saber dónde está en la jerarquía:

| Variable | Tipo | Qué representa | ¿Igual para todos? |
|---|---|---|---|
| `threadIdx` | `dim3` | Posición del thread **dentro de su bloque** | No (única por thread) |
| `blockIdx` | `dim3` | Posición del bloque **dentro del grid** | No (única por bloque) |
| `blockDim` | `dim3` | Nº de threads por bloque | Sí |
| `gridDim` | `dim3` | Nº de bloques por grid | Sí |

Las cuatro variables son `dim3`, así que tienen tres componentes.



### 2.3.3 Cálculo de índice global

- `threadIdx` solo identifica al thread dentro de su bloque. 
- Al lanzar varios bloques, necesitas un **índice global** que identifique al thread dentro de todo el grid.
    - `blockIdx * blockDim + threadIdx`
- Un bloque 1D tiene `blockDim.y == blockDim.z == 1`.

#### Dado un thread, calcular (FILA, COLUMNA) --> (I,J) global que le toca

```c
// 1D
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int i = blockIdx.x * blockDim.x + threadIdx.x;   // columna
int j = blockIdx.y * blockDim.y + threadIdx.y;   // fila

// 3D
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int k = blockIdx.z * blockDim.z + threadIdx.z;
```

#### Dado una celda global, calcular que thread la procesa

```c
blockIdx.x  = i_global / blockDim.x       // división entera (cociente)
threadIdx.x = i_global % blockDim.x       // módulo (resto)

blockIdx.y  = j_global / blockDim.y
threadIdx.y = j_global % blockDim.y

```

#### Ejemplo visual (1D)

**Configuración:**
- `gridDim.x` = 2 → **2 bloques**
- `blockDim.x` = 4 → **4 threads por bloque**
- **Total: 8 threads**

```
              blockIdx.x = 0                  blockIdx.x = 1
        ┌────┬────┬────┬────┐          ┌────┬────┬────┬────┐
        │ T0 │ T1 │ T2 │ T3 │          │ T0 │ T1 │ T2 │ T3 │
        └────┴────┴────┴────┘          └────┴────┴────┴────┘
         tIdx 0  1   2   3              tIdx 0  1   2   3
```

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

| `i` global | `threadIdx.x` | `blockIdx.x * blockDim.x` |
|---|---|---|
| **0** | 0 | 0 · 4 = 0 |
| **1** | 1 | 0 · 4 = 0 |
| **2** | 2 | 0 · 4 = 0 |
| **3** | 3 | 0 · 4 = 0 |
| **4** | 0 | 1 · 4 = 4 |
| **5** | 1 | 1 · 4 = 4 |
| **6** | 2 | 1 · 4 = 4 |
| **7** | 3 | 1 · 4 = 4 |

![alt text](images/thread-indexing.png)

#### Ejemplo visual (2D)
Queremos procesar una matriz 6x6
**Configuración:**
- `gridDim` = (2, 2) → **4 bloques**
- `blockDim` = (3, 3) → **9 threads por bloque**
- **Total: 36 threads** (matriz 6×6)

``` 
Matriz 3×4:           Memoria lineal (12 elementos):
[ a  b  c  d ]        [ a b c d | e f g h | i j k l ]
[ e  f  g  h ]          fila 0     fila 1    fila 2
[ i  j  k  l ]
```

```
                  blockIdx.x = 0          blockIdx.x = 1
                ┌──────────────┐        ┌──────────────┐
                │ ·   ·   ·    │        │ ·   ·   ·    │
blockIdx.y = 0  │ ·   ·   ·    │        │ ·   ·   ·    │
                │ ·   ·   ·    │        │ ·   ·   ·    │
                ├──────────────┤        ├──────────────┤
                │ ·   ·   ·    │        │ ·   ·   ·    │
blockIdx.y = 1  │ ·   ·   ·    │        │ ·   ·   ·    │
                │ ·   ·   ·    │        │ ·   ·   ·    │
                └──────────────┘        └──────────────┘
```

```c
// DADO UN THREAD, CALCULAR QUE CELDA DE LA MATRIZ ORIGINAL PROCESA (i,j) 
int i = blockIdx.x * blockDim.x + threadIdx.x;   // COLUMNA GLOBAL (0-5)
int j = blockIdx.y * blockDim.y + threadIdx.y;   // FILA GLOBAL (0-5)
```

| Thread origen | `blockIdx.x · blockDim.x` | `+ threadIdx.x` | `blockIdx.y · blockDim.y` | `+ threadIdx.y` | Celda `(i, j)` |
|---|---|---|---|---|---|
| bloque (0, 0), thread (0, 0) | 0 · 3 = 0 | 0 + 0 = 0 | 0 · 3 = 0 | 0 + 0 = 0 | **(0, 0)** |
| bloque (0, 0), thread (2, 2) | 0 · 3 = 0 | 0 + 2 = 2 | 0 · 3 = 0 | 0 + 2 = 2 | **(2, 2)** |
| bloque (1, 0), thread (0, 0) | 1 · 3 = 3 | 3 + 0 = 3 | 0 · 3 = 0 | 0 + 0 = 0 | **(3, 0)** |
| bloque (1, 0), thread (2, 1) | 1 · 3 = 3 | 3 + 2 = 5 | 0 · 3 = 0 | 0 + 1 = 1 | **(5, 1)** |
| bloque (0, 1), thread (0, 0) | 0 · 3 = 0 | 0 + 0 = 0 | 1 · 3 = 3 | 3 + 0 = 3 | **(0, 3)** |
| bloque (1, 1), thread (1, 1) | 1 · 3 = 3 | 3 + 1 = 4 | 1 · 3 = 3 | 3 + 1 = 4 | **(4, 4)** |
| bloque (1, 1), thread (2, 2) | 1 · 3 = 3 | 3 + 2 = 5 | 1 · 3 = 3 | 3 + 2 = 5 | **(5, 5)** |

```c
// DADA UNA CELDA (i,j) DE LA MATRIZ ORIGINAL, CALCULAR QUÉ THREAD LA PROCESA
blockIdx.x  = i / blockDim.x       threadIdx.x = i % blockDim.x
blockIdx.y  = j / blockDim.y       threadIdx.y = j % blockDim.y
```

- El **cociente** (`/`) indica en qué bloque cae la celda.
- El **resto** (`%`) indica la posición del thread dentro del bloque.

| Celda `(i, j)` | `i / blockDim.x` | `i % blockDim.x` | `j / blockDim.y` | `j % blockDim.y` | Thread resultante |
|---|---|---|---|---|---|
| **(0, 0)** | 0 / 3 = 0 | 0 % 3 = 0 | 0 / 3 = 0 | 0 % 3 = 0 | bloque (0, 0), thread (0, 0) |
| **(2, 2)** | 2 / 3 = 0 | 2 % 3 = 2 | 2 / 3 = 0 | 2 % 3 = 2 | bloque (0, 0), thread (2, 2) |
| **(3, 0)** | 3 / 3 = 1 | 3 % 3 = 0 | 0 / 3 = 0 | 0 % 3 = 0 | bloque (1, 0), thread (0, 0) |
| **(5, 1)** | 5 / 3 = 1 | 5 % 3 = 2 | 1 / 3 = 0 | 1 % 3 = 1 | bloque (1, 0), thread (2, 1) |
| **(0, 3)** | 0 / 3 = 0 | 0 % 3 = 0 | 3 / 3 = 1 | 3 % 3 = 0 | bloque (0, 1), thread (0, 0) |
| **(4, 4)** | 4 / 3 = 1 | 4 % 3 = 1 | 4 / 3 = 1 | 4 % 3 = 1 | bloque (1, 1), thread (1, 1) |
| **(5, 5)** | 5 / 3 = 1 | 5 % 3 = 2 | 5 / 3 = 1 | 5 % 3 = 2 | bloque (1, 1), thread (2, 2) |

#### Ejemplo visual (3D)

**Configuración:**
- `gridDim` = (2, 2, 2) → **8 bloques**
- `blockDim` = (2, 2, 2) → **8 threads por bloque**
- **Total: 64 threads** (volumen 4×4×4)

```
       k=3  ┌───┬───┬───┬───┐
            │   │   │   │   │       Cada celda es un thread con
       k=2  ├───┼───┼───┼───┤       coordenadas globales (i, j, k).
            │   │   │   │   │
       k=1  ├───┼───┼───┼───┤       8 bloques apilados en 3D:
            │   │   │   │   │       - 2 en x (i: 0-1 | 2-3)
       k=0  ├───┼───┼───┼───┤       - 2 en y (j: 0-1 | 2-3)
            │   │   │   │   │       - 2 en z (k: 0-1 | 2-3)
            └───┴───┴───┴───┘
             i=0  i=1  i=2  i=3
```

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;   // 0-3
int j = blockIdx.y * blockDim.y + threadIdx.y;   // 0-3
int k = blockIdx.z * blockDim.z + threadIdx.z;   // 0-3
```

| Thread origen | `blockIdx.x · blockDim.x` | `+ threadIdx.x` | `blockIdx.y · blockDim.y` | `+ threadIdx.y` | `blockIdx.z · blockDim.z` | `+ threadIdx.z` | Celda `(i, j, k)` |
|---|---|---|---|---|---|---|---|
| bloque (0, 0, 0), thread (0, 0, 0) | 0 · 2 = 0 | 0 + 0 = 0 | 0 · 2 = 0 | 0 + 0 = 0 | 0 · 2 = 0 | 0 + 0 = 0 | **(0, 0, 0)** |
| bloque (0, 0, 0), thread (1, 1, 1) | 0 · 2 = 0 | 0 + 1 = 1 | 0 · 2 = 0 | 0 + 1 = 1 | 0 · 2 = 0 | 0 + 1 = 1 | **(1, 1, 1)** |
| bloque (1, 0, 0), thread (0, 0, 0) | 1 · 2 = 2 | 2 + 0 = 2 | 0 · 2 = 0 | 0 + 0 = 0 | 0 · 2 = 0 | 0 + 0 = 0 | **(2, 0, 0)** |
| bloque (1, 1, 0), thread (1, 1, 0) | 1 · 2 = 2 | 2 + 1 = 3 | 1 · 2 = 2 | 2 + 1 = 3 | 0 · 2 = 0 | 0 + 0 = 0 | **(3, 3, 0)** |
| bloque (0, 0, 1), thread (0, 0, 0) | 0 · 2 = 0 | 0 + 0 = 0 | 0 · 2 = 0 | 0 + 0 = 0 | 1 · 2 = 2 | 2 + 0 = 2 | **(0, 0, 2)** |
| bloque (1, 1, 1), thread (1, 1, 1) | 1 · 2 = 2 | 2 + 1 = 3 | 1 · 2 = 2 | 2 + 1 = 3 | 1 · 2 = 2 | 2 + 1 = 3 | **(3, 3, 3)** |

```c
// DADA UNA CELDA (i,j,k) GLOBAL, CALCULAR QUÉ THREAD LA PROCESA
blockIdx.x  = i / blockDim.x       threadIdx.x = i % blockDim.x
blockIdx.y  = j / blockDim.y       threadIdx.y = j % blockDim.y
blockIdx.z  = k / blockDim.z       threadIdx.z = k % blockDim.z
```

| Celda `(i, j, k)` | `i / blockDim.x` | `i % blockDim.x` | `j / blockDim.y` | `j % blockDim.y` | `k / blockDim.z` | `k % blockDim.z` | Thread resultante |
|---|---|---|---|---|---|---|---|
| **(0, 0, 0)** | 0 / 2 = 0 | 0 % 2 = 0 | 0 / 2 = 0 | 0 % 2 = 0 | 0 / 2 = 0 | 0 % 2 = 0 | bloque (0, 0, 0), thread (0, 0, 0) |
| **(1, 1, 1)** | 1 / 2 = 0 | 1 % 2 = 1 | 1 / 2 = 0 | 1 % 2 = 1 | 1 / 2 = 0 | 1 % 2 = 1 | bloque (0, 0, 0), thread (1, 1, 1) |
| **(2, 0, 0)** | 2 / 2 = 1 | 2 % 2 = 0 | 0 / 2 = 0 | 0 % 2 = 0 | 0 / 2 = 0 | 0 % 2 = 0 | bloque (1, 0, 0), thread (0, 0, 0) |
| **(3, 3, 0)** | 3 / 2 = 1 | 3 % 2 = 1 | 3 / 2 = 1 | 3 % 2 = 1 | 0 / 2 = 0 | 0 % 2 = 0 | bloque (1, 1, 0), thread (1, 1, 0) |
| **(0, 0, 2)** | 0 / 2 = 0 | 0 % 2 = 0 | 0 / 2 = 0 | 0 % 2 = 0 | 2 / 2 = 1 | 2 % 2 = 0 | bloque (0, 0, 1), thread (0, 0, 0) |
| **(3, 3, 3)** | 3 / 2 = 1 | 3 % 2 = 1 | 3 / 2 = 1 | 3 % 2 = 1 | 3 / 2 = 1 | 3 % 2 = 1 | bloque (1, 1, 1), thread (1, 1, 1) |

#### Check de límites

**Problema**: como `blockDim` es fijo (múltiplo de 32) y casi nunca divide el tamaño del problema, lanzas **más threads de los que necesitas**. Los sobrantes accederían fuera del array → crash o corrupción silenciosa.

**Ejemplo**: sumar dos vectores de N=1000 con bloques de 256:

```
numBlocks = (1000 + 255) / 256 = 4   ← ceil
total threads = 4 · 256 = 1024       ← sobran 24
```

**Solución**: añadir un `if (i < N)` que descarta los threads sobrantes.

```c
__global__ void VecAdd(int N, float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {                     // ← threads i = 1000..1023 no entran
        C[i] = A[i] + B[i];
    }
}
```

**Reglas:**

- Número de bloques (ceil): `numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock`.
- En 2D/3D, misma fórmula por eje y check combinado: `if (i < W && j < H)`.

### 2.3.4 Independencia de bloques → escalabilidad

Este es **el invariante más importante** del modelo CUDA, el que hace posible toda la escalabilidad automática:

> Los bloques de un grid deben poder ejecutarse **en cualquier orden, en paralelo o en serie**, sin alterar el resultado.

#### Qué significa en la práctica

| Restricción | Consecuencia |
|---|---|
| No se puede asumir orden de ejecución entre bloques | Si tu lógica requiere que el bloque 0 termine antes que el 1, el código está roto |
| No se puede sincronizar bloques entre sí | CUDA no ofrece primitivas para que un bloque espere a otro dentro del mismo kernel |
| No se pueden comunicar bloques | Solo a través de global memory + atómicos (lento), o cortando en dos kernels |

#### Por qué esta restricción

Solo así CUDA puede repartir bloques libremente entre los SMs disponibles. Si los bloques tuvieran dependencias, el planificador no podría ejecutarlos en cualquier orden y la **escalabilidad automática se rompería**.

```
GPU con 2 SMs                    GPU con 4 SMs
┌─────────┬─────────┐            ┌────┬────┬────┬────┐
│  SM 0   │  SM 1   │            │SM 0│SM 1│SM 2│SM 3│
├─────────┼─────────┤            ├────┼────┼────┼────┤
│ Block 0 │ Block 1 │            │ B0 │ B1 │ B2 │ B3 │
│ Block 2 │ Block 3 │            │ B4 │ B5 │ B6 │ B7 │
│ Block 4 │ Block 5 │            └────┴────┴────┴────┘
│ Block 6 │ Block 7 │
└─────────┴─────────┘
```

Mismo binario, distinto reparto, mismo resultado correcto.

#### Características importantes

- **Cooperación local barata, global imposible**: dentro del bloque puedes cooperar libremente; fuera, no.
- **Diseño guiado por la independencia**: al diseñar un kernel, lo primero es preguntarse "¿cómo divido el trabajo en piezas que no necesiten hablarse entre sí?".
- **Si necesitas sincronización global**: corta el algoritmo en varios kernels (ver siguiente punto).

### 2.3.5 Sincronización (`__syncthreads`, barreras entre kernels)

Aunque los bloques no se pueden sincronizar entre sí, CUDA sí ofrece sincronización a otros niveles:

| Nivel | Mecanismo | Coste |
|---|---|---|
| **Warp** (32 threads) | Implícita por hardware (lockstep) | Gratis |
| **Bloque** | `__syncthreads()` | Barato (~10s ciclos) |
| **Grid** | Cortar en kernels separados (barrera implícita) | Caro (lanzamiento de kernel) |
| **Entre bloques (mismo kernel)** | **No existe** en el modelo básico | — |

#### `__syncthreads()` — barrera dentro del bloque

Es una **barrera** que pausa todos los threads del bloque hasta que **todos** han llegado al mismo punto. Después, todos siguen.

```c
__global__ void kernel(...) {
    cargar_datos_en_shared_memory();

    __syncthreads();   // espera a que TODOS los threads del bloque terminen

    procesar_datos_de_shared_memory();
}
```

Es **fundamental** cuando los threads cooperan a través de shared memory: si el thread A escribe un valor que el thread B va a leer, hay que asegurar que A ha terminado de escribir antes de que B lea. Sin la barrera, hay race condition.

Reglas:

- Solo sincroniza threads del **mismo bloque**, no del grid.
- **Todos los threads del bloque** deben llegar a la misma `__syncthreads()`. Si la pones dentro de un `if` que solo algunos threads ejecutan, los que no entran nunca llegan a la barrera y se cuelga el kernel (deadlock).

#### Barrera implícita entre kernels

Cuando lanzas dos kernels secuencialmente desde el host, el segundo **no empieza hasta que el primero ha terminado completamente**:

```c
kernelA<<<grid, block>>>(a, b, c);
kernelB<<<grid, block>>>(c, d);    // espera implícitamente a kernelA
```

Esta es la **única forma de tener sincronización a nivel de grid**: cortar el algoritmo en varios kernels. Por eso operaciones como reducciones grandes se implementan a veces con varios kernels encadenados.

#### Características importantes

- **Sincronización inter-bloques no existe** en el modelo básico (existe `cooperative_groups` con CC ≥ 6.0 pero requiere lanzamiento especial).
- **`__syncthreads()` debe ser alcanzable por todos los threads del bloque**, o produce deadlock.
- **Cortar en kernels** es la herramienta para sincronizar globalmente, aunque tiene coste por el lanzamiento.

---
## 2.4 Warps: del modelo lógico al hardware

Hasta ahora hemos hablado de threads como unidades lógicas independientes.
- El hardware **no ejecuta threads uno a uno**: los ejecuta en grupos de 32 llamados **warps**.
- Entender el modelo lógico (threads) y el modelo físico (warps) es clave para escribir kernels que rindan, no solo que funcionen.

### 2.4.1 Por qué importa el warp al programar

- Un **warp** es un grupo de **32 threads consecutivos** que el SM ejecuta **en lockstep**:
    - avanzan sincronizados paso a paso, ejecutando exactamente la misma instrucción en el mismo ciclo de reloj.


```
Warp (32 threads):  T0  T1  T2  T3  ...  T31
                     │   │   │   │        │
ciclo k:    todos ejecutan   add r1, r2, r3
ciclo k+1:  todos ejecutan   load r4, [addr]
ciclo k+2:  todos ejecutan   mul r5, r1, r4
```

Consecuencias prácticas que condicionan tus decisiones de diseño:

| Consecuencia | Implicación |
|---|---|
| **Granularidad de ejecución** | El SM ejecuta warps enteros. Si lanzas 33 threads, ejecuta 2 warps (uno con 31 threads "apagados") → desperdicio |
| **Granularidad de divergencia** | Si dentro de un warp hay un branch, el warp ejecuta los dos caminos secuencialmente |
| **Granularidad de acceso a memoria** | Las lecturas a memoria ocurren por warp; si los 32 threads acceden a datos contiguos, una sola transacción los sirve |

#### Características importantes

- **Tamaño fijo del warp = 32** en todas las arquitecturas NVIDIA (desde 2006).
- Es una **propiedad del hardware**, no del modelo de programación: tú no declaras warps, los gestiona el SM.
- Los warps se forman **automáticamente** agrupando threads consecutivos del bloque (T0-T31 = warp 0, T32-T63 = warp 1, etc.).

### 2.4.2 Elección del tamaño de bloque (múltiplos de 32)

Cuando eliges `threadsPerBlock` en `<<<grid, block>>>`, el número **siempre debe ser múltiplo de 32**. Si no lo es, el último warp del bloque tendrá threads "apagados" que ocupan recursos sin hacer trabajo útil.

```
blockDim = 33 threads → 2 warps:
  Warp 0: T0..T31    (32 threads activos) ✓
  Warp 1: T32        (1 activo, 31 apagados) ✗ desperdicio
```

#### Tamaños típicos

| Threads/bloque | Warps | Comentario |
|---|---|---|
| 32 | 1 | Mínimo razonable, ocupación baja |
| 64 | 2 | Suele rendir mal (poco paralelismo) |
| 128 | 4 | Buen punto de partida |
| **256** | 8 | **El más usado**, suele dar buena ocupación |
| 512 | 16 | Limita ocupación si el kernel usa muchos registros |
| 1024 | 32 | Máximo permitido, raramente óptimo |

#### Cómo elegir

- **Sin medir**: empieza con **256**. Es el default razonable que funciona bien en la mayoría de casos.
- **Con medición** (Nsight Compute): refinas según la **ocupación** del kernel (warps activos por SM / warps máximos por SM), que depende del uso de registros y shared memory.
- **Regla de grid**: lanza **al menos tantos bloques como SMs** tenga la GPU, idealmente varios por SM. Si lanzas 4 bloques en una GPU con 132 SMs, dejas 128 SMs parados.

### 2.4.3 Divergencia de warp (branches y rendimiento)

Como los 32 threads de un warp van en lockstep ejecutando la **misma instrucción**, los branches que dependen del thread son problemáticos:

```c
__global__ void kernel(int* data) {
    int i = threadIdx.x;
    if (data[i] > 0) {
        operacion_A();    // camino A
    } else {
        operacion_B();    // camino B
    }
}
```

Si en un warp la mitad de los threads cumplen `data[i] > 0` y la otra mitad no, el hardware **ejecuta los dos caminos secuencialmente**:

```
Ciclo 1: ejecuta camino A (threads del else apagados)
Ciclo 2: ejecuta camino B (threads del if  apagados)
```

El warp tarda **A + B**, aunque cada thread individual solo necesite uno. Esto es la **divergencia de warp**.

#### Dónde sí y dónde no penaliza

| Caso | ¿Penaliza? | Por qué |
|---|---|---|
| Branch divergente **dentro de un warp** | **Sí** | Los dos caminos se ejecutan en serie |
| Branch divergente **entre warps** del mismo bloque | No | Cada warp toma un solo camino |
| Branch divergente **entre bloques** | No | Cada bloque es independiente |
| Branch que toman **todos los threads del warp** | No | Solo se ejecuta un camino |

#### Cómo evitarla o minimizarla

- **Reorganizar datos** para que threads consecutivos tomen el mismo camino (ordenar previamente).
- **Mover la decisión a nivel de bloque o grid**: si todos los threads del bloque van por el mismo camino, no hay divergencia dentro del warp.
- **Branches cortos**: el compilador a veces los convierte en operaciones predicadas sin branch real.
- **Aceptarla cuando es inevitable**: un branch poco frecuente o muy corto suele ser asumible.

### 2.4.4 Sincronización implícita dentro del warp

Como los 32 threads de un warp **van en lockstep por hardware**, están sincronizados sin necesidad de `__syncthreads()`. Esta propiedad habilita primitivas optimizadas a nivel de warp que evitan pasar por shared memory:

| Primitiva | Qué hace | Uso típico |
|---|---|---|
| **Warp shuffle** (`__shfl_sync`, `__shfl_down_sync`...) | Intercambia valores entre threads del warp directamente desde registros | Reducciones, scans, transposiciones |
| **Warp vote** (`__all_sync`, `__any_sync`, `__ballot_sync`) | Operaciones de votación entre los 32 threads | Decisiones colectivas |
| **Warp matrix** (con Tensor Cores) | Operaciones matriciales colectivas a nivel de warp | Deep learning (matmul) |

Esto es lo que usan internamente kernels optimizados como **FlashAttention**, los kernels de **cuBLAS** o las reducciones de softmax y layer norm: aprovechan la sincronización gratuita del warp para evitar tráfico innecesario a shared memory.

#### Características importantes

- **Sincronización gratuita** dentro del warp: no necesitas `__syncthreads()` para coordinar threads del mismo warp.
- **Versiones `_sync` explícitas**: desde Volta (CC 7.0) con *independent thread scheduling*, los threads de un warp ya no van garantizadamente en lockstep en todos los casos. Las versiones modernas requieren una **máscara explícita** indicando qué threads participan (las antiguas sin sync están deprecated).
- **Niveles de sincronización resumidos**:

| Nivel | Mecanismo | Coste |
|---|---|---|
| Warp (32 threads) | Implícita por hardware | Gratis |
| Bloque | `__syncthreads()` | Barato |
| Grid | Cortar en kernels separados | Caro |
| Entre bloques (mismo kernel) | No existe | — |

# Cheatsheet: flujo mecánico para problemas CUDA

| Paso | Quién decide | Cómo |
|---|---|---|
| **1. Dimensionalidad** | El **problema** | 1D vector, 2D matriz, 3D volumen |
| **2. Tamaño** | Los **datos** | N, M×N, M×N×K |
| **3. Hilos/bloque** ⭐ | **Tú** + hardware | Múltiplo de 32, default 256 totales |
| **4. Bloques/grid** | **Fórmula** | `ceil(tamaño / hilos_por_bloque)` |
| **5. Lanzamiento** | Ejecutar | `kernel<<<grid, block>>>(args)` |

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
    if (i < N) {    // ← check de límites SIEMPRE
        // operar
    }
}
```