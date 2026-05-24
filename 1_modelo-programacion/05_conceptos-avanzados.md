# 1 Sincronización de threads en CUDA

- La sincronización garantiza que ciertas operaciones se completen antes que otras cuando hay paralelismo.

## TL;DR

- **Entre hilos de un warp**: sincronización **implícita** por hardware (lock-step, SIMT).
    - Atención a la **warp divergence** cuando un `if` separa a los threads del warp.
    - Desde Volta (SM 7.0+) ya no se puede asumir lock-step automático: hay que usar `__syncwarp()` explícitamente cuando hace falta.
- **Entre hilos de un bloque**: sincronización explícita con `__syncthreads()`.
    - Todos los threads del bloque deben alcanzar la barrera o se produce deadlock.
- **Entre bloques de un mismo kernel**: **no existe** mecanismo de sincronización.
    - Solución: **partir el trabajo en 2 kernels** y dejar que el final del primero actúe como barrera global.

---

## 1.1 Jerarquía de threads en CUDA

- **Thread**: la unidad básica de ejecución.
- **Warp**: grupo de **32 threads** que ejecutan **en lock-step** (la misma instrucción al mismo tiempo).
    - Cada thread dentro del warp ocupa una **lane** (pista). El warp tiene 32 lanes numeradas 0..31.
- **Bloque**: agrupación de hasta 1024 threads. Comparten una memoria `__shared__` y pueden sincronizarse entre sí.
- **Grid**: conjunto de todos los bloques lanzados en un kernel.

## 1.2 Niveles de sincronización disponibles

CUDA ofrece **distintos mecanismos según el nivel jerárquico** donde necesites sincronizar. 
> **No existe sincronización entre bloques dentro de un mismo kernel.** La única barrera global fiable es el **final del kernel**, que actúa como sincronización implícita gestionada por el host.

| Nivel | Mecanismo | Alcance | Coste |
|---|---|---|---|
| Threads dentro de un warp | `__syncwarp(mask)` | 32 threads | Muy bajo |
| Threads dentro de un bloque | `__syncthreads()` | Hasta 1024 threads | Bajo |
| Bloques dentro de un grid | **Fin de kernel** | Todo el grid | Alto |
| Múltiples kernels / streams | `cudaDeviceSynchronize()` (host) | Todo el device | Muy alto |

---

<!-- AMPLIADO: SIMT vs SIMD añadido aquí (1.1.3 del análisis) -->
## 1.3 Modelo de ejecución: SIMT vs SIMD

Las GPUs de NVIDIA ejecutan código bajo el modelo **SIMT (Single Instruction, Multiple Thread)**, que extiende la taxonomía clásica de Flynn (que tiene SISD, SIMD, MISD, MIMD) con una quinta categoría. SIMT se parece a SIMD pero introduce una diferencia importante:

- En **SIMD**, una sola instrucción vectorial se aplica en paralelo a un vector de datos. El flujo de control es único.
- En **SIMT**, múltiples **threads independientes** emiten en paralelo la misma instrucción sobre datos arbitrarios. Cada thread tiene su propio flujo de control y sus propios registros; pueden divergir, aunque con coste.

| Aspecto | SIMD | SIMT |
|---|---|---|
| Unidad de ejecución | Una instrucción aplicada a un vector | Múltiples threads emitiendo la misma instrucción |
| Flujo de control | Único | Cada thread puede divergir (con coste) |
| Registros | Compartidos a nivel de unidad vectorial | Cada thread tiene los suyos privados |
| Direccionamiento | Vectorial coalescido | Cada thread accede a su propia dirección |
| Ejemplo | AVX-512, ARM NEON | CUDA, OpenCL |

Esta distinción explica por qué en CUDA cada thread puede tomar un camino distinto en un `if` (a costa de divergencia) y por qué necesitamos primitivas como las máscaras de warp: en SIMD puro nunca harían falta.

---

## 1.4 Sincronización intra-warp

Los 32 threads de un warp ejecutan **la misma instrucción al mismo tiempo** (SIMT). Antes de Volta (CC < 7.0) esto se podía aprovechar para escribir código "warp-synchronous" sin sincronización explícita. Desde **Volta (CC ≥ 7.0)** existe **Independent Thread Scheduling**: los threads del warp **pueden no estar en la misma instrucción** en todo momento, y ya no se puede asumir lock-step automático.

### 1.4.1 Warp divergence

- Cuando un `if` divide a los threads de un warp (unos toman la rama TRUE, otros FALSE), el warp ejecuta **ambas ramas en serie**, desactivando los threads que no corresponden.
- A esto se le llama **warp divergence** y puede degradar el rendimiento hasta 2x.

```
Warp con 32 threads y un if:

  Caso uniforme (todos cumplen la condición):
    [thread 0..31] → rama TRUE → 1 paso ✓ sin divergencia

  Caso divergente (mitad y mitad):
    [thread 0..15] → rama TRUE,  resto idle  → paso 1
    [thread 16..31] → rama FALSE, resto idle → paso 2
    ✗ Coste doble
```

### 1.4.2 Cuándo importa

- **Sí importa** cuando la condición depende de **datos aleatorios** que varían dentro del warp.
- **No importa** cuando la condición depende de un índice predecible (ej. `if (i < n)` solo diverge en el último warp).

<!-- AMPLIADO: __syncwarp añadido como sección 1.4.3 (1.1.1 del análisis) -->
### 1.4.3 `__syncwarp()`: barrera explícita de warp

Con Independent Thread Scheduling, si los threads de un warp divergen y luego necesitan cooperar (típicamente leyendo y escribiendo en shared memory), hace falta una **barrera de warp explícita**:

```cpp
void __syncwarp(unsigned mask = 0xFFFFFFFF);
```

- Hace que el thread en ejecución espere hasta que **todos los threads indicados en la máscara** hayan ejecutado la misma llamada.
- Es análoga a `__syncthreads()` pero **solo sincroniza un warp** (mucho más barata que sincronizar todo un bloque).
- Las primitivas `_sync` (como `__shfl_down_sync`) ya incluyen una sincronización implícita, pero `__syncwarp()` se usa cuando necesitas sincronizar **lecturas/escrituras a shared memory dentro del warp** sin pagar el coste de `__syncthreads()`.

**Ejemplo — transposición 4×8 dentro de un warp**:

```cpp
float val = get_value(...);
__shared__ float smem[4][8];

// Cada thread escribe en una posición (fila y1, columna x1)
int x1 = threadIdx.x % 8;
int y1 = threadIdx.x / 8;
smem[y1][x1] = val;

__syncwarp();           // ← barrera de warp: TODOS han escrito

// Cada thread lee en orden transpuesto (fila y2, columna x2)
int x2 = threadIdx.x / 4;
int y2 = threadIdx.x % 4;
val = smem[y2][x2];
```

Aquí los 32 threads de un único warp escriben y luego leen la misma shared memory en patrón transpuesto. Sin `__syncwarp()`, en arquitecturas Volta+ algunos threads podrían intentar leer antes de que otros del mismo warp hubieran escrito.

| Comparativa | `__syncwarp(mask)` | `__syncthreads()` |
|---|---|---|
| Alcance | Threads indicados por `mask` dentro del warp | Todos los threads del bloque |
| Coste | Muy bajo (~1 ciclo) | Bajo, pero mayor (toda una barrera de bloque) |
| Uso típico | Sincronizar shared memory dentro de un warp | Sincronizar shared memory entre warps |

---

## 1.5 Sincronización intra-bloque: `__syncthreads()`

`__syncthreads()` es una **barrera de bloque**: ningún thread pasa de esa línea hasta que **todos** los threads del bloque la hayan alcanzado, y las escrituras a memoria realizadas antes son visibles para el resto del bloque tras la barrera. Se usa para coordinar lecturas/escrituras en shared memory:

```c
__shared__ float s[BLOCKSIZE];

s[threadIdx.x] = d[i];     // Fase 1: cada thread escribe en s
__syncthreads();            // Barrera: todos han escrito
float val = s[other_idx];  // Fase 2: cada thread lee de s (¡ahora seguro!)
```

Sin la barrera, un warp puede llegar a la fase 2 **antes** de que otro warp ejecute la fase 1. Ejemplo: invertir un vector con un bloque de 64 threads (2 warps):

```c
// VERSIÓN INCORRECTA (sin barrera)
s[t] = d[t];          // (1) escribir
d[t] = s[n-t-1];      // (2) leer el espejo
```

```
Tiempo →
Warp 0:  s[0..31]=d[0..31]  →  d[t]=s[espejo]  ← lee s[32..63] ¡VACÍO!
Warp 1:                                          s[32..63]=d[32..63]
```

### 1.5.1 Regla crítica: TODOS los threads deben alcanzar la barrera

Si `__syncthreads()` se coloca dentro de un `if` que no todos los threads cumplen, el comportamiento es **indefinido** (deadlock, corrupción, o "funciona por casualidad"):

```c
// MAL: si i>=n para algunos threads, esos nunca llegan a la barrera
if (i < n) {
    s[idx] = d_x[i] * d_y[i];
    __syncthreads();      // ← BUG: dentro del if
}

// BIEN: la barrera está fuera del condicional
if (i < n) {
    s[idx] = d_x[i] * d_y[i];
}
__syncthreads();           // ← todos los threads la alcanzan
```

Si el siguiente paso es una **reducción**, los threads fuera de rango deben dejar en `s` el **elemento neutro** para no contaminar el resultado:

```c
s[idx] = (i < n) ? d_x[i] * d_y[i] : 0.0f;
__syncthreads();
```

| Operación posterior | Elemento neutro |
|---|---|
| `sum` | `0` |
| `prod` | `1` |
| `max` | `-INFINITY` |
| `min` | `+INFINITY` |

Patrón omnipresente en deep learning: `Softmax` y *attention masking* rellenan con `-inf`, reducciones de `mean`/`sum` con `0`.

---

## 1.6 Sincronización inter-bloque: NO existe (dentro de un kernel)

Los bloques se asignan **dinámicamente** a los SMs (Streaming Multiprocessors) del device, y uno puede haber **terminado** mientras otros **aún no han sido lanzados** (si hay más bloques que SMs disponibles). No existe ningún mecanismo seguro para que un bloque espere a otro: si el SM ya está ocupado por bloques esperando, no quedaría espacio para lanzar los que aún tienen que llegar → deadlock garantizado.

### 1.6.1 El bug típico

```c
__global__ void bad_kernel(float *d_v, ...) {
    // Fase 1: cada bloque escribe su parte
    d_v[blockIdx.x] = compute_partial();

    // Fase 2: ¿esperar a que todos los bloques terminen? NO SE PUEDE
    if (blockIdx.x == 0) {
        // Intenta leer d_v[k] para todos los bloques
        for (int k = 0; k < gridDim.x; k++)
            sum += d_v[k];   // ← puede leer valores AÚN NO ESCRITOS
    }
}
```

El bloque 0 puede llegar a la fase 2 **antes** de que los demás hayan ejecutado siquiera la fase 1 → lee basura o ceros.

### 1.6.2 La solución: dividir en dos kernels

La única forma fiable de sincronizar globalmente es **terminar el kernel actual y lanzar otro**. El fin de kernel es una barrera global implícita gestionada por el host.

```c
// Host:
phase1_kernel<<<nblocks, BLOCKSIZE>>>(d_v, ...);  // todos los bloques terminan
phase2_kernel<<<1, 1>>>(d_v, ...);                 // ahora d_v está listo
```

```
Host: lanza phase1_kernel ──┐
                            ▼
GPU:  [B0][B1][B2]...[BN]  ejecutan en paralelo
                            │
              ── FIN KERNEL = barrera global ──
                            │
Host: lanza phase2_kernel ──┘
                            ▼
GPU:  [trabajo posterior, ahora con datos consistentes]
```

---

## 1.7 Patrón canónico: producto escalar (reducción)

El producto escalar (`dot product`) ilustra todos los conceptos anteriores en dos fases: **A** — cada thread calcula `x[i]*y[i]` y los threads del bloque agregan en shared memory; **B** — alguien suma los parciales de todos los bloques. Como no hay sincronización inter-bloque dentro de un kernel, la implementación canónica usa **dos kernels** (el fin del primero actúa de barrera global):

```c
__global__ void compute_kernel1(unsigned int n, float *d_x, float *d_y, float *d_v) {
    __shared__ float s[BLOCKSIZE];

    int idx = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Todos los threads escriben en s (elemento neutro si están fuera de rango)
    s[idx] = (i < n) ? d_x[i] * d_y[i] : 0.0f;

    // Barrera alcanzada por TODOS los threads
    __syncthreads();

    // Reducción local: thread 0 suma los BLOCKSIZE elementos
    if (idx == 0) {
        float a = 0.0f;
        for (int k = 0; k < BLOCKSIZE; k++) a += s[k];
        d_v[blockIdx.x] = a;
    }
}

__global__ void compute_kernel2(unsigned int nblocks, float *d_v) {
    float a = 0.0f;
    for (unsigned int k = 0; k < nblocks; k++) a += d_v[k];
    d_result = a;
}

// Host:
compute_kernel1<<<nblocks, BLOCKSIZE>>>(n, d_x, d_y, d_v);
compute_kernel2<<<1, 1>>>(nblocks, d_v);   // barrera global vía fin de kernel1
```

**Tres errores típicos a evitar**:

| Error | Síntoma | Corrección |
|---|---|---|
| `__syncthreads()` dentro de `if (i < n)` | Deadlock o corrupción en el último bloque | Sacarlo fuera del `if` |
| No inicializar `s[idx]` para threads con `i ≥ n` | Resultado contaminado con basura de shared memory | Usar ternario con elemento neutro |
| Reducir entre bloques dentro del mismo kernel | Lectura de valores aún no escritos | Dividir en dos kernels |

---
# 2 Primitivas a nivel de warp en CUDA

- Las primitivas a nivel de warp son instrucciones especiales que permiten a los 32 hilos de un warp **intercambiar datos directamente entre sus registros privados**, sin pasar por memoria compartida ni memoria global gracias a una red hardware (crossbar) dedicada dentro del SM.
    - Son muy potentes para optimizar kernels gracias a su muy baja latencia (~1 ciclo).
```
SIN primitivas:                CON primitivas:
Hilo A → shared mem → Hilo B   Hilo A → [crossbar] → Hilo B
        (~25 ciclos)                    (~1 ciclo)
```

<!-- AMPLIADO: 3 categorías formales CUDA 9 añadido como 2.1 (1.2.1 del análisis) -->
## 2.1 Las tres categorías formales de CUDA 9

CUDA 9 introdujo formalmente **tres categorías** de primitivas a nivel de warp, todas con sufijo `_sync` (obligatorio desde CUDA 9; las versiones antiguas sin él están deprecadas):

| Categoría | Función | Primitivas |
|---|---|---|
| **1. Intercambio sincronizado de datos** | Mover valores entre lanes del warp | `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`, `__all_sync`, `__any_sync`, `__uni_sync`, `__ballot_sync`, `__match_any_sync`, `__match_all_sync` |
| **2. Consulta de máscara activa** | Devuelve qué lanes del warp están activos en el punto actual | `__activemask` |
| **3. Sincronización de warp** | Barrera de warp + memory fence | `__syncwarp` |

Las primitivas de categoría 1 se suelen agrupar funcionalmente en tres familias (Shuffle, Vote, Match) según resuelvan distintos tipos de problema.

## 2.2 Familias de primitivas (categoría 1)

| Familia | Función | Ejemplos | Caso típico |
|---|---|---|---|
| **Shuffle** (`__shfl_*_sync`) | Mover valores entre lanes | `__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync` | Reducciones, broadcasts, scans |
| **Vote** (`__*_sync`) | Reducir un predicado booleano a un único valor | `__all_sync`, `__any_sync`, `__ballot_sync` | Decisiones colectivas, conteo de condiciones |
| **Match** (`__match_*_sync`) | Encontrar lanes con el mismo valor | `__match_any_sync`, `__match_all_sync` | Agrupar hilos por clave (histogramas) |

> **Nota importante**: todas terminan en `_sync` y reciben una **máscara de 32 bits** indicando qué lanes participan.

## 2.3 La máscara: `mask`

- Las primitivas `_sync` reciben siempre como primer argumento una **máscara de 32 bits** en la que el bit `i` indica si el lane `i` participa en la operación (1) o no (0). 
- Su existencia se debe a la **divergencia de warp**: tras un `if`, no todos los lanes están "presentes" en la primitiva, y el hardware necesita saber a quién está esperando para no leer valores de lanes inactivos.

| Máscara hex | Significado |
|---|---|
| `0xFFFFFFFF` | Los 32 lanes participan (caso normal, sin divergencia). |
| `0x0000FFFF` / `0xFFFF0000` | Solo la mitad baja (lanes 0–15) / alta (lanes 16–31). |
| `0x55555555` / `0xAAAAAAAA` | Solo lanes pares / impares. |
| `__activemask()` | Devuelve dinámicamente la máscara de lanes activos en ese punto del código. |

```cpp
if (threadIdx.x < 16) {
    // Solo lanes 0..15 están aquí → la mask debe reflejarlo
    val = __shfl_down_sync(0x0000FFFF, val, 1);
}
```

<!-- AMPLIADO: peligro de __activemask dentro de if (1.1.2 del análisis) -->
### 2.3.1 Peligro: `__activemask()` dentro de una rama divergente

`__activemask()` parece la opción más segura para construir la máscara dentro de un `if`, **pero es peligrosa en Volta y posteriores** debido al Independent Thread Scheduling:

```cpp
// EJEMPLO INCORRECTO — patrón a evitar
if (threadIdx.x < NUM_ELEMENTS) {
    unsigned mask = __activemask();   // ← ¿qué lanes están aquí REALMENTE?
    val = input[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
}
```

**El problema**: `__activemask()` devuelve los lanes que están **en esa instrucción concreta en ese ciclo**, no los que entraron en el `if`. Con Independent Thread Scheduling, el hardware **no garantiza** que todos los lanes que entraron al `if` lleguen al `__activemask()` simultáneamente. Podrían estar repartidos en distintos ciclos, y cada uno vería una máscara diferente → resultado incorrecto silencioso.

**La solución correcta**: construir la máscara **antes** del `if` con `__ballot_sync()`, que sí sincroniza:

```cpp
// EJEMPLO CORRECTO
unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < NUM_ELEMENTS);
if (threadIdx.x < NUM_ELEMENTS) {
    val = input[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
}
```

| Cuándo usar `__activemask()` | Cuándo NO usar `__activemask()` |
|---|---|
| En código sin divergencia previa (todos los lanes activos) | Justo después de un `if` que diverge |
| Como referencia para depuración | Para construir la máscara de un `__shfl_*_sync` posterior dentro del `if` |
| Dentro de `coalesced_threads()` (Cooperative Groups, que sí maneja esto correctamente) | Como sustituto de `__ballot_sync()` |

**Regla práctica**: si vas a hacer una operación colectiva dentro de un `if`, calcula la máscara con `__ballot_sync()` **antes** del `if`.

## 2.4 El offset en reducciones

- El parámetro `offset` (o `delta`) indica **cuántas posiciones de distancia** hay entre el lane que lee y el lane que aporta el dato. 
- Para reducir N elementos en árbol binario se empieza con `offset = N/2` y se divide por 2 en cada paso, en total `log₂(N)` pasos. 
    - Para un warp completo: **16 → 8 → 4 → 2 → 1** (5 pasos).

Empezando por el offset grande, en cada paso la mitad baja del warp absorbe limpiamente a la mitad alta, y el resultado se concentra en el lane 0:

```
Paso 1 (off=16): lanes 0-15 absorben lanes 16-31
Paso 2 (off=8):  lanes 0-7  absorben lanes 8-15
Paso 3 (off=4):  lanes 0-3  absorben lanes 4-7
Paso 4 (off=2):  lanes 0-1  absorben lanes 2-3
Paso 5 (off=1):  lane  0    absorbe lane 1   → resultado en lane 0
```

Cuando `laneId + offset >= 32` el shuffle **no falla**: el lane se queda con su propio valor (los lanes altos acumulan basura, pero solo leemos el lane 0). Esto permite escribir el bucle sin comprobaciones de rango:

```cpp
// warpSize está predefinido y vale 32
for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(mask, val, offset);
}
```

## 2.5 Familia Shuffle

Las primitivas de shuffle permiten **mover valores entre los registros de distintos lanes** dentro del mismo warp. Son las más usadas en kernels optimizados.

| Función | Definición |
|---|---|
| `__shfl_sync(mask, var, srcLane, width=32)` | Cada lane recibe el valor de `var` que tiene el lane indicado por `srcLane` (broadcast desde un lane a todos los demás); si `srcLane` se sale del rango se aplica módulo `width`. |
| `__shfl_down_sync(mask, var, delta, width=32)` | El lane `i` recibe el valor de `var` del lane `i + delta` (desplazamiento hacia laneIds mayores); si `i + delta >= width` conserva su propio `var`. Base de las **reducciones**. |
| `__shfl_up_sync(mask, var, delta, width=32)` | El lane `i` recibe el valor de `var` del lane `i - delta` (desplazamiento hacia laneIds menores); si `i - delta < 0` conserva su propio `var`. Base de los **prefix sums (scans)**. |

**Ejemplo 1 — broadcast con `__shfl_sync`**: el lane 0 reparte un coeficiente a todo el warp.

```cpp
__global__ void scaleByLeader(float *data) {
    int laneId = threadIdx.x % 32;
    float val = data[threadIdx.x];

    // Solo el lane 0 lee el coeficiente desde memoria global
    float coef = 0.0f;
    if (laneId == 0) {
        coef = data[1000];
    }

    // Broadcast: TODOS los lanes leen el coef del lane 0
    coef = __shfl_sync(0xFFFFFFFF, coef, 0);

    // Ahora todos tienen coef, sin que hayan tocado memoria
    data[threadIdx.x] = val * coef;
}
```

**Ejemplo 2 — reducción de suma con `__shfl_down_sync`**:

```cpp
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // resultado válido solo en lane 0
}
```

**Ejemplo 3 — prefix sum inclusivo con `__shfl_up_sync`**:

```cpp
__device__ int warpInclusiveScan(int val) {
    int laneId = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (laneId >= offset) {
            val += neighbor;
        }
    }
    return val;
}
```

Con valores `[1,1,1,1,...]` el resultado sería `[1,2,3,4,...]`: cada lane tiene la suma acumulada hasta su posición.

## 2.6 Familia Vote

Las primitivas de Vote permiten a los 32 lanes "votar" sobre un predicado booleano y obtener un resultado colectivo (el mismo en todos los lanes, por lo que el `if` posterior no diverge).

| Función | Definición |
|---|---|
| `__all_sync(mask, predicate)` | Todos los lanes activos reciben `1` si y solo si **todos** ellos evalúan `predicate` como cierto (AND lógico colectivo). |
| `__any_sync(mask, predicate)` | Todos los lanes activos reciben `1` si **al menos uno** de ellos evalúa `predicate` como cierto (OR lógico colectivo). |
| `__ballot_sync(mask, predicate)` | Cada lane activo recibe la misma bitmask de 32 bits en la que el bit `i` vale 1 si el lane con rank `i` evalúa `predicate` como cierto. Imprescindible para construir máscaras antes de ramas divergentes. |

**Ejemplo 1 — `__all_sync` para elegir camino rápido si todos los valores son válidos**:

```cpp
__global__ void checkAllValid(float *data) {
    float val = data[threadIdx.x];
    bool isValid = (val >= 0.0f) && !isnan(val);

    // ¿están TODOS bien?
    int allOk = __all_sync(0xFFFFFFFF, isValid);

    if (allOk) {
        // Camino rápido: todos válidos
        data[threadIdx.x] = sqrtf(val);
    } else {
        // Camino lento: hay algún problema
        data[threadIdx.x] = (isValid) ? sqrtf(val) : 0.0f;
    }
}
```

**Ejemplo 2 — `__any_sync` para detectar convergencia en un solver iterativo**:

```cpp
__global__ void iterativeSolver(float *x, float *x_new, float tol) {
    int tid = threadIdx.x;
    x_new[tid] = computeUpdate(x, tid);
    float error = fabsf(x_new[tid] - x[tid]);

    // ¿Algún hilo aún no ha convergido?
    int anyDiverged = __any_sync(0xFFFFFFFF, error > tol);

    if (!anyDiverged) {
        // Todos del warp han convergido
        if (tid % 32 == 0) atomicAdd(&convergedWarps, 1);
    }
}
```

## 2.7 Familia Match

Las primitivas de Match permiten **comparar valores entre lanes y agruparlos**. Disponibles desde Volta (CC 7.0+).

| Función | Definición |
|---|---|
| `__match_any_sync(mask, value)` | Cada lane recibe una bitmask de 32 bits con un `1` en cada lane activo que tenga el mismo `value` que él (cada lane recibe **una máscara distinta**). Base de histogramas y reducciones por clave. |
| `__match_all_sync(mask, value, *pred)` | Comprueba si **todos** los lanes activos tienen el mismo `value`: devuelve `mask` y escribe `*pred = 1` si hay unanimidad, o `0` y `*pred = 0` si no. |

**Ejemplo 1 — `__match_any_sync` para histograma sin atomics dispersos**:

```cpp
__global__ void countLabels(int *labels, int *count) {
    int laneId = threadIdx.x % 32;
    int label = labels[threadIdx.x];

    // ¿Qué lanes tienen mi misma etiqueta?
    unsigned same = __match_any_sync(0xFFFFFFFF, label);

    // El líder del grupo es el lane activo más bajo
    int leader = __ffs(same) - 1;

    // Solo el líder reporta el conteo del grupo
    if (laneId == leader) {
        int groupSize = __popc(same);
        atomicAdd(&count[label], groupSize);
    }
}
```

En lugar de 32 `atomicAdd` dispersos se hace **uno por grupo**: mejora de 5-10× cuando hay valores repetidos.

**Ejemplo 2 — `__match_all_sync` para detectar acceso uniforme a memoria**:

```cpp
__global__ void specializedAccess(int *addresses, float *data) {
    int laneId = threadIdx.x % 32;
    int addr = addresses[threadIdx.x];
    int blockId = addr / 128;

    // ¿Todos los hilos van al mismo bloque de memoria?
    int pred;
    unsigned same = __match_all_sync(0xFFFFFFFF, blockId, &pred);

    if (pred) {
        // Acceso uniforme: solo el lane 0 lee el bloque entero
        if (laneId == 0) prefetchBlock(blockId);
    } else {
        // Acceso disperso: cada hilo lee su parte
        data[threadIdx.x] = readScalar(addr);
    }
}
```

## 2.8 Ejemplo completo: reducción jerárquica

Una reducción eficiente trabaja en **tres niveles** que reflejan la jerarquía CUDA:

| Nivel | Técnica |
|---|---|
| **Warp** (32 hilos) | `__shfl_down_sync` entre lanes |
| **Bloque** (varios warps) | Shared memory entre warps + `warpReduceSum` final |
| **Grid** (todos los bloques) | `atomicAdd` o segunda pasada |

```cpp
// Nivel 1: reducción dentro de un warp
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Nivel 2: reducción dentro de un bloque
__device__ float blockReduceSum(float val) {
    static __shared__ float sharedBuf[32];
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    val = warpReduceSum(val);  // reduce dentro del warp

    if (laneId == 0) sharedBuf[warpId] = val;
    __syncthreads();

    int numWarps = blockDim.x / 32;
    val = (threadIdx.x < numWarps) ? sharedBuf[laneId] : 0.0f;

    if (warpId == 0) val = warpReduceSum(val);  // segunda reducción
    return val;
}

// Nivel 3: kernel completo con grid-stride loop
__global__ void reduceKernel(const float *input, float *output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Cada hilo acumula varios elementos en su registro
    float sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        sum += input[i];
    }

    sum = blockReduceSum(sum);

    // Solo un hilo por bloque escribe al resultado global
    if (threadIdx.x == 0) atomicAdd(output, sum);
}
```

**Flujo de datos**: cada hilo acumula varios elementos del input con un *grid-stride loop* en un registro local → `warpReduceSum` concentra cada warp en su `lane 0` → los lane-0 escriben en shared memory → un segundo `warpReduceSum` reduce esos parciales al thread 0 del bloque → `atomicAdd` agrega los resultados de cada bloque al output global.

### 2.8.1 Reducción de warp paso a paso

Partiendo de los 32 lanes con `[1, 2, 3, ..., 32]` (suma esperada = 528). En cada paso la **zona útil se reduce a la mitad** y cada lane útil contiene el doble de información: árbol binario de `log₂(32) = 5` pasos.

| Iteración | Zona útil | val del lane 0 | Elementos sumados en lane 0 |
|---|---|---|---|
| Inicio | lanes 0-31 | 1 | {1} |
| Tras off=16 | lanes 0-15 | 18 | {1, 17} |
| Tras off=8 | lanes 0-7 | 52 | {1, 17, 9, 25} |
| Tras off=4 | lanes 0-3 | 120 | 8 elementos |
| Tras off=2 | lanes 0-1 | 256 | 16 elementos |
| Tras off=1 | lane 0 | **528** | **los 32** ✓ |

---
# 3 Operaciones atómicas en CUDA

- Las operaciones atómicas son el mecanismo que CUDA ofrece para resolver el problema de los accesos concurrentes de escritura sobre la misma posición de memoria.

## 3.1 El problema que resuelven: race conditions

En CUDA lanzas miles de threads ejecutando el mismo kernel en paralelo. Si varios de ellos intentan **leer-modificar-escribir** la misma posición de memoria simultáneamente, el resultado es indeterminado.

### 3.1.1 Anatomía de una race condition

Una operación que en código parece atómica, como `counter += 1`, **no lo es a nivel de hardware**. Se descompone en tres instrucciones separadas:

- `LOAD`: leer el valor actual de `counter` desde memoria a un registro
- `ADD`: sumar 1 en el registro
- `STORE`: escribir el resultado de vuelta a memoria

Si dos threads ejecutan esto a la vez sobre el mismo `counter` (valor inicial 5):

```
Tiempo →
T1:  LOAD(5) ──── ADD(6) ──── STORE(6)
T2:        LOAD(5) ──── ADD(6) ──── STORE(6)
```

Resultado: `counter = 6` cuando debería ser `7`. **Se ha perdido una actualización**. Y lo peligroso es que no hay error ni excepción: el código compila, corre, y a veces incluso da el resultado correcto por casualidad.

### 3.1.2 Regla para detectar race conditions

Sobre cada línea de un kernel, hazte estas dos preguntas:

| Pregunta | Si la respuesta es... |
|---|---|
| ¿Esta línea **escribe** en memoria global o shared? | Si no, no hay race. Lecturas concurrentes son seguras. |
| ¿La dirección de escritura es la **misma** para varios threads simultáneamente? | Si no (cada thread escribe a su posición), no hay race. Si sí, **race condition**. |

Las variables locales (registros, `tid`, contadores de bucle) **nunca** producen race condition porque son privadas de cada thread.

## 3.2 Qué es una operación atómica

Una operación atómica es una operación **read-modify-write** que el hardware garantiza que se ejecuta como una **unidad indivisible**: ningún otro thread puede intervenir en medio.

- El hardware **serializa** los accesos concurrentes a la misma dirección: si N threads hacen `atomicAdd` sobre la misma variable, se procesan uno detrás de otro, pero **todas** las actualizaciones se aplican correctamente.
- La serialización ocurre **solo entre threads que acceden a la misma dirección**. Si 10.000 threads hacen `atomicAdd` a 10.000 direcciones distintas, no hay serialización: corren en paralelo total.

### 3.2.1 Comparativa rápida

| Aspecto | Sin atómicas (`x += 1`) | Con atómicas (`atomicAdd(&x, 1)`) |
|---|---|---|
| Granularidad | 3 instrucciones separadas | 1 unidad indivisible |
| Race conditions | Sí | No |
| Resultado con N threads | Indeterminado | Siempre correcto |
| Coste por operación | Muy bajo | Mayor (serialización si hay contención) |

## 3.3 Funcionamiento interno paso a paso

Cuando un thread invoca `atomicAdd(&x, 1)`:

- **Paso 1**: el thread envía una petición atómica a la unidad de memoria correspondiente (L2 cache para memoria global, unidad de shared memory para `__shared__`). **No** ejecuta LD/ADD/ST en sus propios registros.
- **Paso 2**: el hardware identifica la dirección objetivo. Si hay otras peticiones atómicas pendientes sobre **la misma dirección**, las pone en cola.
- **Paso 3**: el hardware procesa cada petición de forma atómica:
  - Lee el valor actual
  - Aplica la operación
  - Escribe el resultado
  - Devuelve el **valor antiguo** al thread peticionario
- **Paso 4**: el siguiente thread en cola se procesa, y así sucesivamente.

```python
def hardware_atomic_add(address, val):
    acquire_lock(address)          # Bloquear esa dirección
    old = memory[address]          # LOAD
    memory[address] = old + val    # ADD + STORE como unidad
    release_lock(address)          # Liberar
    return old                     # Devolver el valor que había antes
```

- **No es un lock software** tipo mutex: es un mecanismo hardware en L2 cache, sin context switch ni intervención del SO.
- **El "lock" es por línea de caché** (~128 bytes), no por byte. Dos atómicas a variables distintas pero en la misma línea contienden → *false sharing*.
- **Se devuelve el valor antiguo, no el nuevo**. Esto es deliberado y se usa muchísimo (ver 3.6).

## 3.4 Ejemplo trazado: 4 threads, mismo contador

`counter = 0`, 4 threads ejecutan el incremento. Resultado esperado: `counter = 4`.

### 3.4.1 Sin atómica (incorrecto)

```cuda
__global__ void incrementBroken(int* counter) {
    *counter += 1;
}
// Lanzado con <<<1, 4>>>
```

Traza con un orden posible de intercalado:

```
Tiempo │ T0              T1              T2              T3              │ counter
───────┼─────────────────────────────────────────────────────────────────┼────────
  t0   │ LD R1←0                                                          │ 0
  t1   │                 LD R1←0                                          │ 0
  t2   │ ADD R1=1        ADD R1=1        LD R1←0                          │ 0
  t3   │ ST counter←1                    ADD R1=1        LD R1←0          │ 1
  t4   │                 ST counter←1    ST counter←1    ADD R1=1         │ 1
  t5   │                                                 ST counter←1     │ 1
Resultado: counter = 1   ❌
```

Cada thread leyó `0`, sumó `1` en su registro privado, y escribió `1` encima del trabajo de los demás.

### 3.4.2 Con atómica (correcto)

```cuda
__global__ void incrementCorrect(int* counter) {
    atomicAdd(counter, 1);
}
```

Traza: los 4 threads envían su petición simultáneamente, pero el hardware las encola y procesa una a una.

```
Tiempo │ T0              T1              T2              T3              │ Cola sobre &counter │ counter
───────┼─────────────────────────────────────────────────────────────────┼─────────────────────┼────────
  t0   │ req atomicAdd   req atomicAdd   req atomicAdd   req atomicAdd   │ [T0,T1,T2,T3]       │ 0
  t1   │ ← devuelve 0    (esperando)     (esperando)     (esperando)     │ procesa T1          │ 1
  t2   │                 ← devuelve 1    (esperando)     (esperando)     │ procesa T2          │ 2
  t3   │                                 ← devuelve 2    (esperando)     │ procesa T3          │ 3
  t4   │                                                 ← devuelve 3    │ vacía               │ 4
Resultado: counter = 4   ✅
```

Observaciones clave:

- Los 4 threads **piden** en paralelo, el hardware **procesa** secuencialmente.
- Cada thread recibe un **valor antiguo distinto** (0, 1, 2, 3) → "ticket único" por thread.
- La serialización solo aplica por dirección.

## 3.5 Catálogo de operaciones atómicas

Todas las funciones siguen el mismo esquema: reciben un puntero a la posición de memoria sobre la que operar, aplican el cambio como unidad indivisible y devuelven el **valor antiguo** (antes de la operación).

| Función | Definición |
|---|---|
| `atomicAdd(addr, val)` / `atomicSub(addr, val)` | Suma (o resta) `val` al contenido de `*addr` y devuelve el valor antiguo. Uso típico: contadores, histogramas, reducciones. |
| `atomicMin(addr, val)` / `atomicMax(addr, val)` | Reemplaza `*addr` por `min(*addr, val)` o `max(*addr, val)` y devuelve el antiguo. Uso típico: encontrar extremos. |
| `atomicExch(addr, val)` | Escribe `val` en `*addr` y devuelve el valor antiguo (intercambio incondicional). Uso típico: locks, swaps. |
| `atomicCAS(addr, compare, val)` | Compare-And-Swap: si `*addr == compare`, escribe `val`; devuelve siempre el valor antiguo (haya habido swap o no). **Primitiva universal** — con ella se construye cualquier otra atómica. |
| `atomicAnd(addr, val)` / `atomicOr(addr, val)` / `atomicXor(addr, val)` | Aplica la operación bitwise correspondiente entre `*addr` y `val`, guarda el resultado y devuelve el antiguo. Uso típico: bitmasks, flags. |
| `atomicInc(addr, max)` / `atomicDec(addr, max)` | Incrementa (o decrementa) `*addr` con **wrap-around** cuando alcanza `max`. Uso típico: contadores cíclicos. |

<!-- AMPLIADO: tipos soportados expandido con vectorizados y half (1.2.3 del análisis) -->
### 3.5.1 Tipos soportados por `atomicAdd`

`atomicAdd` admite muchos más tipos de los que uno suele recordar. Útil sobre todo para **inferencia y entrenamiento de redes neuronales**, donde half/bfloat16 y vectorizados son omnipresentes:

| Tipo | Disponible desde | Caso de uso |
|---|---|---|
| `int`, `unsigned int` | Siempre | Contadores, índices |
| `unsigned long long int` | Siempre | Contadores 64 bits |
| `float` | CC 2.0+ | Reducciones FP32 |
| `double` | CC 6.0+ (Pascal) | Reducciones FP64 |
| `__half` | CC 7.0+ (Volta) | Inferencia FP16 |
| `__half2` | CC 6.0+ | Dos `__half` empaquetados (2× throughput) |
| `__nv_bfloat16` | CC 8.0+ (Ampere) | Entrenamiento bf16 |
| `__nv_bfloat162` | CC 8.0+ | Dos `__nv_bfloat16` empaquetados |
| `float2` | CC 9.0+ (Hopper) | Acumular complejos o vec2 |
| `float4` | CC 9.0+ (Hopper) | Acumular vec4 (RGBA, gradientes, etc.) |

**Por qué importa en deep learning**:
- `atomicAdd` sobre `__half2` permite acumular dos valores FP16 en una sola operación atómica → mitad de tráfico a memoria, doble de throughput.
- Las versiones vectorizadas (`float2`, `float4`) son clave en kernels de scatter/gather donde se acumulan gradientes (backward pass) sobre embeddings o tensores.
- Frameworks como PyTorch y TensorRT usan estas variantes internamente en kernels como `index_add_`, `scatter_add_`, o en operaciones de reducción de attention.

**Ejemplo — implementar `atomicAdd` con `atomicCAS`** (patrón *lock-free retry loop*, base de muchas estructuras concurrentes y forma en que CUDA emula atómicas no soportadas nativamente, como `atomicAdd` sobre `double` en arquitecturas antiguas):

```cuda
__device__ int myAtomicAdd(int* addr, int val) {
    int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed + val);
    } while (assumed != old);   // Reintentar si alguien me ganó
    return old;
}
```

## 3.6 Patrón clave: aprovechar el valor antiguo devuelto

El hecho de que `atomicAdd` devuelva el valor **antes** de la suma es lo que permite usarlo como **generador de índices únicos**. Ejemplo de *stream compaction* (filtrar positivos sin colisiones):

```cuda
__global__ void compactPositives(int* input, int N, int* output, int* outCount) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N && input[tid] > 0) {
        int idx = atomicAdd(outCount, 1);   // ← me reserva un slot único
        output[idx] = input[tid];           // ← lo lleno sin pisar a nadie
    }
}
```


## 3.7 Coste y rendimiento

Una atómica sobre memoria global puede ser **10-100× más lenta** que una operación normal, según el nivel de **contención** (cuántos threads compiten por la misma dirección simultáneamente).

- Contención baja (cada thread a una dirección distinta): casi sin penalización.
- Contención alta (todos al mismo contador): serialización masiva, GPU infrautilizada.

### 3.7.1 Memoria global vs shared

Las atómicas sobre `__shared__` son **mucho más rápidas** que sobre memoria global, porque la unidad de shared está dentro del SM. Esto motiva el patrón canónico de optimización:

> **Agregación local en shared → flush atómico a global**

```cuda
__global__ void countPositivesFast(int* data, int N, int* count) {
    __shared__ int blockCount;
    if (threadIdx.x == 0) blockCount = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N && data[tid] > 0) {
        atomicAdd(&blockCount, 1);   // Atómica en shared: rápida
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, blockCount); // 1 flush por bloque a global
    }
}
```

Con 1M de threads en bloques de 256 → pasas de **1M atómicas en global** a **~3.900 atómicas en global**.

### 3.7.2 Tabla de estrategias

| Estrategia | Contención global | Velocidad |
|---|---|---|
| Atómica directa a global por cada thread | Altísima | Lenta |
| Agregación en shared + flush por bloque | Baja | Rápida |
| Warp-level primitives (`__shfl_*`) + atómica final | Mínima | Muy rápida |
| Reducción jerárquica pura sin atómicas | Cero | La más rápida si aplica |

## 3.8 Caso de estudio: suma de un vector

Ejemplo típico de slide de teoría — sirve para ver que la race condition no siempre está donde el ojo la busca.

### 3.8.1 Versión con race

```cuda
__global__ void vectSumRace(int* d_vect, size_t size, int* result) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < size) {
        *result += d_vect[tid];                  // ← race aquí
        tid += blockDim.x * gridDim.x;           // grid-stride loop, OK
    }
}
```

Localización exacta de la race con la regla de 3.1.2:

| Línea | ¿Escribe? | ¿Misma dirección entre threads? | Veredicto |
|---|---|---|---|
| `tid = blockIdx.x * ...` | A registro | No (privado) | OK |
| `tid < size` | No escribe | — | OK |
| `*result += d_vect[tid]` | Sí, a `result` | **Sí, todos** | **RACE** |
| `tid += blockDim.x * gridDim.x` | A registro | No (privado) | OK |

Cada thread lee una posición distinta de `d_vect` (sin problema), pero **todos escriben sobre `*result`** (problema). El `d_vect[tid]` en la parte derecha distrae del verdadero issue: el `*result += ...` es exactamente el mismo patrón LD/ADD/ST del contador.

### 3.8.2 Versión correcta con atomicAdd

```cuda
__global__ void vectSumAtomic(int* d_vect, size_t size, int* result) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < size) {
        atomicAdd(result, d_vect[tid]);
        tid += blockDim.x * gridDim.x;
    }
}
```

Correcto, pero **no óptimo**: con N elementos hay N atómicas contendiendo sobre la misma dirección. La forma rápida es **reducción jerárquica con atómica de flush por bloque** (patrón 3.7.1).

---
# 4 Paralelismo Dinámico en CUDA

El **paralelismo dinámico** (DP) permite que un kernel ya en ejecución en la GPU **lance otros kernels desde el propio device**, sin volver al host. Introducido en CUDA 5.0 (compute capability ≥ 3.5).

## 4.1 Modelo clásico vs. modelo dinámico

En el **modelo clásico** solo la CPU lanza kernels: la configuración `<<<gridDim, blockDim>>>` se decide *antes* del lanzamiento, en base a información conocida en el host. Si durante la ejecución se descubre que hace falta más paralelismo, hay que terminar el kernel, volver a CPU, decidir y relanzar.

En el **modelo dinámico**, un kernel que ya corre en GPU ejecuta él mismo `child_kernel<<<g, b>>>(...)` y las decisiones de cuánto paralelismo lanzar se toman **en función de los datos vistos en la GPU**.

```
MODELO CLÁSICO                  MODELO DINÁMICO
─────────────────               ──────────────────────
CPU ── launch ──> GPU           CPU ── launch ──> GPU (parent)
       (kernel A)                                  │
                                                   ├── launch ──> GPU (child 1)
CPU ── launch ──> GPU                              ├── launch ──> GPU (child 2)
       (kernel B)                                  └── launch ──> GPU (child 3)
```

## 4.2 Funcionamiento interno

| Mecanismo | Descripción |
|---|---|
| **Lanzamiento desde el padre** | Dentro del kernel padre se escribe `child_kernel<<<grid, block>>>(args...)` con la **misma sintaxis** que desde el host. Por convención solo **un thread del bloque** (típicamente el de `threadIdx.x == 0`) hace el lanzamiento, para no lanzar el hijo N veces. |
| **Streams device-side** | Cada lanzamiento se asocia a un stream del device. Por defecto, los hijos lanzados desde un mismo bloque comparten un stream implícito y se serializan. Para concurrencia entre hijos hay que crear streams explícitos con `cudaStreamCreateWithFlags`. |
| **Sincronización padre-hijo (tail launch)** | El padre **no espera** a los hijos al terminar su código. El modelo actual (CUDA 12+) es *tail launch*: el runtime garantiza que todos los hijos terminen antes de que el padre se considere completado desde el host. La antigua API `cudaDeviceSynchronize()` device-side fue deprecada. |
| **Memoria compartida** | Los hijos **ven la memoria global** del padre (mismo address space) pero **no** ven la shared memory ni los registros (recursos por bloque y por thread). Para pasar datos: por argumentos o escribiéndolos a global memory antes del lanzamiento. |
| **Recursión real** | Un kernel puede llamarse **a sí mismo** desde el device (con tamaño de grid distinto). Esto habilita algoritmos recursivos como Mariani-Silver o quicksort directamente en GPU. |

## 4.3 Casos de uso

DP brilla cuando **la cantidad de paralelismo no se conoce a priori** o **varía mucho por región**:

- **Algoritmos recursivos**: quicksort, recorrido de árboles (BVH en ray tracing, octrees).
- **Mallas adaptativas (AMR)**: refinar más una región solo donde se detecte alta variabilidad. Aplicación canónica: fractales como **Mandelbrot** (ver 4.5).
- **Grafos irregulares**: BFS/DFS donde unos nodos tienen 2 vecinos y otros 10.000.
- **Workloads desbalanceados**: un thread descubre que su porción de trabajo es enorme y lanza un sub-grid para paralelizarla.
- **Algoritmos jerárquicos**: procesar a grano grueso y refinar con un kernel hijo donde haga falta.

## 4.4 Ejemplo 1: tareas con tamaño variable

Array de 8 tareas con tamaños muy distintos. Las pequeñas se procesan rápido con un solo thread; las grandes necesitan paralelizarse:

```
Tarea:    0   1   2   3   4   5   6   7
Tamaño:  10  500  20  800  15  30  600  25
```

Sin DP habría que lanzar un kernel uniforme con el peor caso (800 threads para todas) o volver a CPU para decidir. Con DP, un único kernel padre procesa todas las tareas y **cada thread decide en tiempo de ejecución**: si su tarea es pequeña la resuelve él mismo en un bucle; si es grande lanza un kernel hijo.

```
PARENT KERNEL: 1 thread por tarea (8 threads en total)

  thread 0 → tarea pequeña (10)  → la procesa solo
  thread 1 → tarea GRANDE (500)  ──> lanza CHILD KERNEL<<<2, 256>>>
  thread 2 → tarea pequeña (20)  → la procesa solo
  thread 3 → tarea GRANDE (800)  ──> lanza CHILD KERNEL<<<4, 256>>>
  thread 4 → tarea pequeña (15)  → la procesa solo
  thread 5 → tarea pequeña (30)  → la procesa solo
  thread 6 → tarea GRANDE (600)  ──> lanza CHILD KERNEL<<<3, 256>>>
  thread 7 → tarea pequeña (25)  → la procesa solo
```

```cuda
#define THRESHOLD 100   // umbral para paralelizar con un hijo

// KERNEL HIJO: procesa una tarea grande en paralelo
__global__ void processBigTask(float* data, int offset, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[offset + idx] = sqrtf(data[offset + idx]) * 2.0f;
    }
}

// KERNEL PADRE: un thread por tarea
__global__ void parentKernel(float* data, int* offsets, int* sizes, int nTasks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nTasks) return;

    int offset = offsets[tid];
    int size   = sizes[tid];

    if (size < THRESHOLD) {
        // Tarea pequeña → la proceso yo mismo, secuencialmente
        for (int i = 0; i < size; ++i) {
            data[offset + i] = sqrtf(data[offset + i]) * 2.0f;
        }
    } else {
        // Tarea grande → LANZAMIENTO DINÁMICO de un hijo
        int threads = 256;
        int blocks  = (size + threads - 1) / threads;
        processBigTask<<<blocks, threads>>>(data, offset, size);
    }
}

// HOST: un único lanzamiento; todo lo demás lo decide la GPU
parentKernel<<<1, nTasks>>>(d_data, d_offsets, d_sizes, nTasks);
cudaDeviceSynchronize();
```

**Compilación** — DP no compila con `nvcc` por defecto:

```bash
nvcc -arch=sm_75 -rdc=true main.cu -o main -lcudadevrt
```

| Flag | Para qué |
|---|---|
| `-arch=sm_XX` | Compute capability ≥ 3.5. Usa el `sm_` correspondiente a tu GPU. |
| `-rdc=true` | *Relocatable Device Code*: necesario para que el padre vea el símbolo del hijo en runtime. |
| `-lcudadevrt` | Linkea el *CUDA Device Runtime*, que permite llamar APIs CUDA desde el device. |

<!-- AMPLIADO: Caso Mandelbrot / Mariani-Silver añadido como 4.5 (1.1.4 del análisis) -->
## 4.5 Ejemplo 2: el conjunto de Mandelbrot con Mariani-Silver

Este es el **caso de estudio canónico de paralelismo dinámico** en el curso. Ilustra cómo DP con recursión real permite implementar algoritmos jerárquicos adaptativos de forma natural.

### 4.5.1 El conjunto de Mandelbrot

El conjunto de Mandelbrot se define como:

$$z_0 = c,\quad z_{n+1} = z_n^2 + c,\quad M = \{c \in \mathbb{C} : \exists R\ \forall n: |z_n| < R\}$$

Para cada punto `c` del plano complejo se itera la fórmula hasta que `|z|` excede un radio o se alcanza un máximo de iteraciones (`MAX_DWELL`). El número de iteraciones (`dwell`) determina el color del píxel:

- Si `dwell == MAX_DWELL`: el punto pertenece al conjunto (suele pintarse negro).
- Si `dwell < MAX_DWELL`: no pertenece, y se colorea según el valor de `dwell`.

### 4.5.2 Versión naive: kernel por píxel

```cuda
__global__ void mandelbrot_k(int *dwells, int w, int h, complex cmin, complex cmax) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h)
        dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
}
```

**Problema**: este algoritmo desperdicia recursos masivamente.
- Las regiones **dentro** del conjunto iteran `MAX_DWELL` veces por píxel sin aportar información (todas dan negro).
- Las regiones grandes **fuera** del conjunto con `dwell` constante también se evalúan píxel a píxel innecesariamente.
- El cálculo real solo es necesario en el **borde fractal**, que es una fracción mínima del área total.

### 4.5.3 El algoritmo Mariani-Silver

Se basa en una propiedad geométrica del conjunto de Mandelbrot: **es conexo**. De ahí:

> Si el **borde** de una región rectangular tiene un `dwell` constante, **toda la región** tiene ese mismo `dwell`.

Esto permite saltarse el cálculo de los píxeles interiores cuando el borde es uniforme. La estrategia es **recursiva**:

```
mariani_silver(rectangle):
    if (borde de rectangle tiene dwell constante):
        rellenar rectangle entero con ese dwell    ← caso base: ahorro masivo
    else if (rectangle es pequeño):
        evaluar píxel a píxel                       ← caso base: precisión
    else:
        for cada sub_rectángulo in subdivide(rectangle):
            mariani_silver(sub_rectangle)           ← recursión
```

### 4.5.4 Implementación con paralelismo dinámico

El algoritmo encaja perfectamente con DP: cada bloque procesa un rectángulo, y si necesita subdividir, **lanza versiones más pequeñas de sí mismo** como kernels hijos.

```cuda
#define MAX_DWELL 512
#define MAX_DEPTH 4       // profundidad máxima de recursión
#define MIN_SIZE 32       // por debajo de este tamaño → evaluar píxel a píxel
#define SUBDIV 4          // factor de subdivisión por eje (16 hijos por padre)

__global__ void mandelbrot_block_k(int *dwells, int w, int h,
                                    complex cmin, complex cmax,
                                    int x0, int y0, int d, int depth) {
    // Cada bloque procesa un sub-rectángulo
    x0 += d * blockIdx.x;
    y0 += d * blockIdx.y;

    // Reducción paralela: ¿el borde tiene dwell común?
    int common_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);

    // El thread 0 del bloque decide qué hacer
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (common_dwell != DIFF_DWELL) {
            // CASO 1: borde uniforme → rellenar el rectángulo entero
            dwell_fill<<<grid, bs>>>(dwells, w, x0, y0, d, common_dwell);
        }
        else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
            // CASO 2: subdividir recursivamente (LANZAMIENTO DEL MISMO KERNEL)
            mandelbrot_block_k<<<dim3(SUBDIV, SUBDIV), bs>>>(
                dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1);
        }
        else {
            // CASO 3: rectángulo muy pequeño → evaluación píxel a píxel
            mandelbrot_pixel_k<<<grid, bs>>>(dwells, w, h, cmin, cmax, x0, y0, d);
        }
    }
}
```

Funciones auxiliares:
- `border_dwell`: una **reducción paralela** dentro del bloque que comprueba si todos los píxeles del borde tienen el mismo `dwell`.
- `dwell_fill`: kernel hijo que pinta una región entera con un valor de `dwell` constante.
- `mandelbrot_pixel_k`: el kernel original por píxel, pero aplicado solo a un sub-rectángulo.

**Lanzamiento desde host**:

```cuda
int width = 8192, height = 8192;
mandelbrot_block_k<<<dim3(INIT_SUBDIV, INIT_SUBDIV), dim3(bsx, bsy)>>>(
    dwells, width, height, complex(-1.5, -1), complex(0.5, 1),
    0, 0, width / INIT_SUBDIV, /*depth=*/0);
```

### 4.5.5 Por qué es el ejemplo canónico

- **Recursión real**: el kernel `mandelbrot_block_k` se llama a sí mismo. Esto es el caso "fuerte" de DP — no solo el padre lanza hijos distintos, sino que el árbol de llamadas tiene profundidad variable.
- **Granularidad adaptativa**: el grid no es uniforme. Las zonas con borde uniforme se procesan con un solo kernel `dwell_fill`, las zonas complejas con muchas subdivisiones.
- **Decisión basada en datos**: la elección de subdividir o no depende de los datos vistos en GPU. Esto no se puede hacer eficientemente desde el host.
- **Ahorro masivo**: para una imagen 8192×8192, Mariani-Silver con DP puede ser **10-100× más rápido** que el kernel naive por píxel.

### 4.5.6 Visualización del resultado

```
Imagen procesada con Mariani-Silver:

  ┌──────────────────────────────────────┐
  │      ┌─┐                              │   ← rectángulos grandes (fuera del set)
  │      │ │   ┌─────────────┐           │      rellenados con un solo dwell
  │  ┌───┘ └───┘             │           │
  │  │     ███████           │  ┌──┐     │
  │  │   ███████████████     │  │  │     │   ← bordes fractales:
  │  │  █████████████████    │  └──┘     │      subdivididos recursivamente
  │  │   ███████████████     │            │      hasta MIN_SIZE
  │  │     ███████           │            │
  │  └───┐ ┌───┐             │            │
  │      │ │   └─────────────┘           │
  │      └─┘                              │
  └──────────────────────────────────────┘
```

## 4.6 Comparativa: sin DP vs. con DP

| Aspecto | Sin DP (host-driven) | Con DP (device-driven) |
|---|---|---|
| Quién lanza kernels | Solo la CPU | CPU y también la propia GPU |
| Round-trips CPU↔GPU | Múltiples (uno por nivel de decisión) | Uno solo (lanzamiento inicial) |
| Latencia de decisión | Alta (transferencia + lógica en CPU) | Baja (decisión en SM) |
| Granularidad adaptativa | Limitada por tu pipeline | Natural y recursiva |
| Recursión | Solo simulada (con bucles + relanzamientos) | Real (kernel se llama a sí mismo) |
| Complejidad de código | Más kernels separados + lógica en host | Lógica unificada en el kernel padre |
| Overhead por lanzamiento hijo | N/A | Existe, no despreciable (~µs) |
| Útil para | Workloads uniformes y predecibles | Workloads irregulares, recursivos, adaptativos |

## 4.7 Limitaciones y precauciones

- **Overhead de lanzamiento**: cada `child_kernel<<<...>>>` cuesta microsegundos. Si los hijos son triviales, el overhead domina y vas más lento que sin DP.
- **Profundidad limitada**: jerarquía padre→hijo→nieto típicamente hasta 24 niveles (configurable). Recursión muy profunda → desbordamiento.
- **Memoria de la cola de lanzamientos**: ajustar `cudaLimitDevRuntimePendingLaunchCount` si lanzas muchísimos hijos.
- **Alternativas**: *persistent kernels* o *work-stealing queues* pueden ser más eficientes para algunos patrones irregulares.
- **CUDA 12**: el `cudaDeviceSynchronize()` device-side fue deprecado en favor del modelo *tail launch*; si lees código antiguo de DP, verás esa API que ya no se recomienda.

---

# 5 Cooperative Groups en CUDA

## 5.1 Definición y motivación

- Los **Cooperative Groups** son una API introducida en CUDA 9 que permite definir, sincronizar y operar sobre **grupos de hilos** a granularidades distintas a las tradicionales (warp, bloque, grid). 
- IDEA: tratar al grupo de hilos como un grupo con el que vas a trabajar, qué tamaño tiene, y sobre él invocas operaciones colectivas (sync, reduce, shuffle, etc.).

### 5.1.1 Problemas que resuelven

Antes de Cooperative Groups, CUDA solo ofrecía 2 primitivas de sincronización:      
- `__syncthreads()`: a nivel de bloque 
- sincronización implícita del warp. 

    Esto causaba:
    - **Independent Thread Scheduling (Volta+, SM 7.0)**: la suposición de que los 32 hilos de un warp ejecutaban en lockstep dejó de ser segura. El código "warp-synchronous" basado en `volatile` y sin sincronización explícita se rompió.
    - No había forma  de sincronizar solo un subconjunto de hilos (8 ó 16).
    - **Sin sincronización entre bloques**: dentro de un kernel no podías sincronizar el grid completo (solo entre lanzamientos de kernels distintos).

Cooperative Groups resuelve los cuatro problemas con una abstracción uniforme.

### 5.1.2 Jerarquía de grupos

CUDA expone una jerarquía que refleja el hardware pero añade granularidades intermedias y superiores.

```
multi_grid_group         (varias GPUs — deprecated desde CUDA 11.3, se usa NCCL)
  └── grid_group         (todos los hilos del kernel)
        └── thread_block (un bloque CUDA — equivale a __syncthreads)
              └── thread_block_tile<N>  (sub-grupo estático de N hilos)
                    └── coalesced_group (hilos activos tras divergencia)
```

### 5.1.3 Headers y compilación

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;
```

- Flag `-rdc=true` (relocatable device code): **solo** si usas `grid_group`/`grid.sync()`.
- Tiles ≤ 32 hilos: cualquier arquitectura SM 3.0+.
- `grid.sync()`: SM 6.0+ (Pascal en adelante).
- `memcpy_async` integrado con CG: SM 8.0+ (Ampere en adelante).

---

## 5.2 Qué es un tile

- Un **tile** es un **subconjunto contiguo y de tamaño fijo de hilos** dentro de un bloque, tratado como una unidad de cooperación independiente. 
    - El grupo de N hilos trabajan juntos en una sub-tarea, mientras otro grupo de N hilos trabaja en otra sub-tarea en paralelo.
- El objeto `tile` es un handle ligero (un par de registros) que codifica esta información más la máscara de warp para shuffles.
### 5.2.1 Intuición

- Un bloque CUDA con 32 hilos es como una clase de 32 alumnos. Sin tiles, o todos colaboran en lo mismo, o cada uno trabaja aislado. Con tiles, divides la clase en, por ejemplo, 4 grupos de 8 alumnos, y cada grupo resuelve un sub-problema cooperando internamente sin molestar a los demás.

```
Bloque de 32 hilos:    [0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | 16 ... 23 | 24 ... 31]
                             Tile 0                     Tile 1               Tile 2       Tile 3
Rango interno del tile: 0  1  2  3  4  5  6  7    0  1  2  3  4  5  6  7    0 ...  7    0 ...  7
```

- Cada tile **reinicia su rango interno desde 0**. 
    - El hilo global 8 es el "hilo 0" del Tile 1
    - El hilo global 16 es el "hilo 0" del Tile 2.

### 5.2.2 Propiedades clave

- **Tamaño fijo en tiempo de compilación**: `tiled_partition<N>`, donde `N` es una constante (típicamente 2, 4, 8, 16 o 32).
- **N suele ser ≤ 32**: porque un warp tiene 32 hilos. 
    - Tiles ≤ 32 caben enteros dentro de un warp y heredan sus garantías de ejecución.
- **Particiones contiguas**: Tile 0 son los primeros N hilos, Tile 1 los siguientes N, etc.
- **Mismo rango entre tiles**: cada tile tiene hilos numerados `0..N-1` internamente.
- **Sincronización barata**: `tile.sync()` sobre un tile ≤ 32 es prácticamente gratis.

### 5.2.3 Funcionamiento interno

Cuando declaras `auto tile = cg::tiled_partition<8>(block)`:

El compilador calcula, para cada hilo:
- tile: `threadIdx.x / 8`
- rango interno dentro del tile:`threadIdx.x % 8`
- El objeto `tile` es un handle ligero (un par de registros) que codifica esta información más la máscara de warp para shuffles.

Llamadas como `tile.shfl(val, 0)` se compilan a `shfl.sync` con la máscara correcta — solo los 8 hilos del tile participan.

### 5.2.4 Qué tamaño elegir

| N | Cuándo usarlo | Ejemplo |
|---|---|---|
| 2 ó 4 | Operaciones por pares o quads | Suma real + imaginaria de un complejo |
| 8 | Sub-warp moderado, datos en grupos de 8 | 8 columnas de matriz, 8 elementos SIMD |
| 16 | Mitad de warp | Half-warp histogram, FFT radix-16 |
| 32 | **Warp completo — caso por defecto** | Reducciones, scans, shuffles típicos |

---

## 5.3 Métodos principales y parámetros

Los métodos siguen un patrón claro. **El grupo es el primer argumento** (implícito como `g.metodo(...)` o explícito en funciones libres `cg::reduce(g, ...)`).

Casi todos los métodos son **SPMD**: los N hilos del grupo ejecutan la misma llamada conjuntamente, y cada uno recibe **su** resultado según su rango.

### 5.3.1 Métodos de información (sin parámetros)

| Método | Definición |
|---|---|
| `g.size()` | Devuelve el número total de hilos que componen el grupo `g`. |
| `g.thread_rank()` | Devuelve el índice de este hilo dentro del grupo, en el rango `[0, size)`. |
| `g.meta_group_size()` | (Solo tiles) Devuelve cuántos tiles del mismo tamaño hay dentro del grupo padre. |
| `g.meta_group_rank()` | (Solo tiles) Devuelve a qué tile pertenece este hilo dentro del grupo padre, en `[0, meta_group_size)`. |
| `g.group_index()` | (Solo `thread_block`) Devuelve un `dim3` con el índice del bloque dentro del grid. Equivalente a `blockIdx`. |
| `g.thread_index()` | (Solo `thread_block`) Devuelve un `dim3` con el índice del hilo dentro del bloque. Equivalente a `threadIdx`. |

```cpp
auto block = cg::this_thread_block();        // 32 hilos
auto tile  = cg::tiled_partition<8>(block);  // 4 tiles de 8

int tile_rank = tile.thread_rank();   // 0..7  (dentro del tile)
int tile_id   = tile.meta_group_rank(); // 0..3  (qué tile soy)
int n_tiles   = tile.meta_group_size(); // 4
int tile_sz   = tile.size();           // 8

dim3 bid      = block.group_index();   // == blockIdx
dim3 tid      = block.thread_index();  // == threadIdx
```

### 5.3.2 Sincronización

- **`g.sync()`**: barrera del grupo. Bloquea a cada hilo del grupo `g` hasta que todos los demás hilos del mismo grupo hayan llegado a esta misma llamada; tras la barrera, las escrituras a memoria realizadas antes son visibles para el resto del grupo.

```cpp
__shared__ float buffer[256];
auto block = cg::this_thread_block();

buffer[threadIdx.x] = threadIdx.x * 2.0f;
block.sync();   // espera a que TODOS escriban antes de leer
float vecino = buffer[(threadIdx.x + 1) % blockDim.x];
```

**Sintaxis equivalentes** (todas hacen lo mismo para un `thread_block`):

```cpp
__syncthreads();                          // forma clásica de CUDA
block.sync();                             // método del objeto
cg::synchronize(block);                   // función libre
this_thread_block().sync();               // sin variable intermedia
cg::synchronize(this_thread_block());     // función libre, sin variable
```

### 5.3.3 Shuffles (comunicación entre registros)

Mueven valores **directamente entre registros**, sin pasar por memoria. Solo en tiles ≤ 32.

| Método | Definición |
|---|---|
| `g.shfl(val, src_rank)` | Cada hilo del grupo `g` recibe el valor de `val` que tiene el hilo cuyo rank dentro del grupo es `src_rank` (broadcast desde un hilo a todos). |
| `g.shfl_up(val, delta)` | Cada hilo recibe el valor de `val` que tiene el hilo situado `delta` posiciones por debajo (rank − delta); los hilos sin fuente válida (rank < delta) conservan su propio `val`. |
| `g.shfl_down(val, delta)` | Cada hilo recibe el valor de `val` que tiene el hilo situado `delta` posiciones por encima (rank + delta); los hilos sin fuente válida (rank + delta ≥ size) conservan su propio `val`. |
| `g.shfl_xor(val, mask)` | Cada hilo recibe el valor de `val` que tiene el hilo cuyo rank es `rank XOR mask` (intercambio butterfly, base de reducciones y FFT). |

```cpp
auto tile = cg::tiled_partition<8>(cg::this_thread_block());
int v = threadIdx.x * 10;          // hilos: 0,10,20,30,40,50,60,70

int broadcast = tile.shfl(v, 0);   // todos reciben 0 (el valor del rank 0)
int abajo     = tile.shfl_down(v, 1);
// rank 0->10, rank 1->20, ..., rank 6->70, rank 7->70 (sin cambio en borde)
int pares     = tile.shfl_xor(v, 1);
// pares (0,1), (2,3), (4,5), (6,7) intercambian: 0<->10, 20<->30, ...
```

### 5.3.4 Votación (predicados booleanos)

| Método | Definición |
|---|---|
| `g.any(pred)` | Todos los hilos reciben `true` si al menos un hilo del grupo evalúa el predicado `pred` como cierto, y `false` en caso contrario. |
| `g.all(pred)` | Todos los hilos reciben `true` si y solo si todos los hilos del grupo evalúan el predicado `pred` como cierto. |
| `g.ballot(pred)` | Todos los hilos reciben la misma bitmask de 32 bits en la que el bit `i` vale 1 si el hilo con rank `i` cumple `pred` y 0 si no. |

```cpp
auto warp = cg::tiled_partition<32>(cg::this_thread_block());
float val = leer_dato();

bool hay_negativo    = warp.any(val < 0.0f);
bool todos_positivos = warp.all(val > 0.0f);

unsigned mask = warp.ballot(val > threshold);
int cuantos   = __popc(mask);   // popcount: cuántos cumplen
```

### 5.3.5 Operaciones colectivas: reduce y scan

**Funciones libres** del namespace `cg`. Reciben el grupo como **primer argumento**.

| Función | Definición |
|---|---|
| `cg::reduce(g, val, op)` | Combina los valores `val` de todos los hilos del grupo `g` usando el operador binario `op` y devuelve el mismo resultado final en cada hilo. |
| `cg::inclusive_scan(g, val, op)` | Cada hilo con rank `i` recibe la combinación con `op` de los valores `val` de los hilos con rank `0..i` (incluyendo el suyo). |
| `cg::exclusive_scan(g, val, op)` | Cada hilo con rank `i` recibe la combinación con `op` de los valores `val` de los hilos con rank `0..i-1` (sin incluir el suyo); el hilo 0 recibe la identidad del operador. |

Operadores disponibles: `cg::plus<T>()`, `cg::less<T>()` (mínimo), `cg::greater<T>()` (máximo), `cg::bit_and<T>()`, `cg::bit_or<T>()`, `cg::bit_xor<T>()`.

```cpp
auto tile = cg::tiled_partition<8>(cg::this_thread_block());
int v = 1;   // todos los hilos: 1

int suma = cg::reduce(tile, v, cg::plus<int>());          // todos reciben 8
int incl = cg::inclusive_scan(tile, v, cg::plus<int>());  // rank i recibe i+1
int excl = cg::exclusive_scan(tile, v, cg::plus<int>());  // rank i recibe i
```

### 5.3.6 Partición (crear sub-grupos)

| Función | Definición |
|---|---|
| `cg::tiled_partition<N>(parent)` | Divide el grupo `parent` en sub-grupos contiguos de tamaño fijo `N` y devuelve a cada hilo el handle (`thread_block_tile<N>`) del tile al que pertenece. |
| `cg::coalesced_threads()` | Devuelve un `coalesced_group` que contiene únicamente los hilos del warp que están activos en este punto de la ejecución (útil tras una divergencia para operar solo sobre los hilos que entraron en una rama). |

```cpp
auto block = cg::this_thread_block();          // 128 hilos
auto warp  = cg::tiled_partition<32>(block);   // 4 warps
auto tile8 = cg::tiled_partition<8>(warp);     // dentro del warp, 4 tiles de 8

if (mi_condicion) {
    auto active = cg::coalesced_threads();
    int rank = active.thread_rank();   // numera solo los hilos que entraron al if
}
```

---

## 5.4 Ejemplos completos

### 5.4.1 Suma por tile

Cada grupo de 8 hilos suma sus 8 valores.

```cpp
__global__ void sumar_por_tile(float* entrada, float* salida) {
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<8>(block);

    float val = entrada[threadIdx.x];

    // Reducción colectiva dentro del tile
    float suma_tile = cg::reduce(tile, val, cg::plus<float>());

    // Solo el líder escribe a memoria global
    if (tile.thread_rank() == 0) {
        salida[tile.meta_group_rank()] = suma_tile;
    }
}
```

Para entrada `[1, 2, ..., 32]` → salida `[36, 100, 164, 228]`.

### 5.4.2 Reducción warp-level (sustituye `__shfl_down_sync`)

```cpp
__device__ float warp_reduce_sum(float val) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    return cg::reduce(warp, val, cg::plus<float>());
}
```

Más legible, más segura ante Independent Thread Scheduling, y más portable entre arquitecturas que el clásico:

```cpp
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

### 5.4.3 Argmax por tile (reduce + ballot + ffs)

Cada tile de 8 hilos calcula el máximo y el rango del hilo que lo tenía.

```cpp
__global__ void max_por_tile(float* scores, float* max_out, int* argmax_out) {
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<8>(block);

    float mi_score = scores[threadIdx.x];

    // 1. Máximo del tile
    float max_tile = cg::reduce(tile, mi_score, cg::greater<float>());

    // 2. ¿Soy yo el del máximo?
    bool soy_max = (mi_score == max_tile);

    // 3. Bitmask de quién(es) tiene(n) el máximo
    unsigned mask = tile.ballot(soy_max);

    // 4. Primer bit a 1 = rango del primer ganador (resuelve empates por orden)
    int argmax = __ffs(mask) - 1;

    // 5. El líder del tile escribe
    if (tile.thread_rank() == 0) {
        int tile_id = tile.meta_group_rank();
        max_out[tile_id]    = max_tile;
        argmax_out[tile_id] = argmax;
    }
}
```

Patrón canónico en GPU para softmax, NMS, top-k.

### 5.4.4 Warp-aggregated atomics con `coalesced_group`

Muchos hilos quieren incrementar un contador global, pero solo algunos cumplen un predicado. Pasamos de N atomics a 1 atomic por warp.

```cpp
if (deteccion.score > threshold) {
    auto active = cg::coalesced_threads();

    int leader_offset;
    if (active.thread_rank() == 0)
        leader_offset = atomicAdd(counter, active.size()); // 1 atomic por warp

    leader_offset = active.shfl(leader_offset, 0);
    int my_slot = leader_offset + active.thread_rank();
    out[my_slot] = deteccion;
}
```

### 5.4.5 Kernel persistente con `grid.sync()`

Algoritmos iterativos (BFS, PageRank, solvers) sin tener que relanzar el kernel en cada iteración.

```cpp
__global__ void bfs_kernel(...) {
    auto grid = cg::this_grid();
    while (!converged) {
        relaja_aristas_locales();
        grid.sync();                 // espera a todos los bloques
        if (grid.thread_rank() == 0)
            actualiza_frontier_global();
        grid.sync();
    }
}
```

Lanzas con `cudaLaunchCooperativeKernel(...)`. Requiere que **todos los bloques quepan simultáneamente en la GPU** (residencia total).