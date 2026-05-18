# 1 Ejemplo: Suma de vectores de 1M datos con grid-stride loop

## 1.1 Cambio de patrón respecto al ejemplo anterior

En el ejemplo anterior usábamos el patrón **"1 thread = 1 elemento"**: lanzábamos tantos threads como elementos (1M threads para 1M datos). Ahora vamos a usar el patrón **"grid-stride loop"**: cada thread procesa **varios elementos** en bucle.

### 1.1.1 ¿Por qué hacer esto?

| Razón | Detalle |
|---|---|
| **Independencia del tamaño del problema** | El mismo kernel funciona para `n=1000` o `n=10⁹` sin cambiar el lanzamiento |
| **Mejor reutilización de threads** | Crear/destruir threads tiene coste; mejor que un thread vivo haga más trabajo |
| **Ajuste a la GPU concreta** | Lanzas un grid del tamaño óptimo para tu hardware, no del tamaño del problema |
| **Portabilidad** | El mismo binario rinde bien en GPUs pequeñas y grandes |

## 1.2 Planteamiento del problema

**Operación:** `c[i] = a[i] + b[i]` para `i = 0..n-1`.

**Tamaño:** `n = 1M = 2²⁰ = 1 048 576` elementos.

**Hardware asumido:** GPU con **128 SMs** y **4 warp schedulers por SM**.

## 1.3 Estrategia: dimensionar el grid según la GPU, no según `n`

La idea es lanzar **exactamente los suficientes threads para llenar la GPU**, y dejar que cada thread procese varios elementos en bucle.

### 1.3.1 ¿Cuántos threads necesitamos para llenar la GPU?

```
Threads residentes máximos = N_SMs × max_threads_por_SM
                           = 128 × 2 048
                           = 262 144 threads
```

Lanzar más threads que eso no aporta paralelismo extra (solo aumenta la cola de bloques esperando). Lanzar menos infrautiliza la GPU.

**Regla práctica**: lanza un grid que pueda llenar la GPU **2-4 veces** para tolerar desbalanceos.

```
N_threads_objetivo = 2 × 262 144 = 524 288 threads
```

### 1.3.2 Cálculo del grid

```
blockDim = 256 hilos/bloque (estándar)
nblocks  = 524 288 / 256 = 2 048 bloques
```

### 1.3.3 Llamada al kernel

```cuda
compute_kernel<<<2048, 256>>>(n, d_a, d_b, d_c);
```

### 1.3.4 Comparación con el patrón anterior

| Aspecto | Patrón 1:1 (anterior) | Patrón grid-stride (este) |
|---|---|---|
| `nblocks` | 4 096 | 2 048 |
| Threads lanzados | 1 048 576 | 524 288 |
| Elementos por thread | 1 | 2 (= n / N_threads) |
| Bucle en el kernel | No | Sí |
| Threads sobrantes en último bloque | 0 (justo) | 0 (estos no escriben fuera porque tienen bucle con condición) |

## 1.4 Código del kernel

```cuda
__global__ void compute_kernel(const unsigned int n, float *d_a, float *d_b, float *d_c) {
    int idx    = threadIdx.x + blockIdx.x * blockDim.x;   // identidad global
    int stride = blockDim.x * gridDim.x;                  // total de threads del grid
    
    for (int i = idx; i < n; i += stride) {
        d_c[i] = d_a[i] + d_b[i];
    }
}
```

### 1.4.1 Desglose

- **`idx`**: identificador global del thread (rango 0..524 287 en nuestro caso).
- **`stride`**: número total de threads del grid = `blockDim × gridDim = 256 × 2048 = 524 288`.
- **`for (i = idx; i < n; i += stride)`**: cada thread empieza en su `idx` y avanza dando saltos de `stride` hasta agotar el vector.
- **Sin `if (idx < n)`**: la condición `i < n` del `for` ya protege los accesos fuera de rango.

### 1.4.2 Cálculo de iteraciones por thread

```
Iteraciones por thread = n / N_threads = 1 048 576 / 524 288 = 2
```

Cada thread procesa **exactamente 2 elementos** (porque `n` es divisible por `N_threads`).

## 1.5 Visualización del reparto del trabajo

Veamos qué elementos procesa cada thread:

```
                Iter 1            Iter 2
                ──────            ──────
Thread 0:       elem 0            elem 524 288
Thread 1:       elem 1            elem 524 289
Thread 2:       elem 2            elem 524 290
...
Thread 31:      elem 31           elem 524 319    ← fin del warp 0
Thread 32:      elem 32           elem 524 320    ← inicio del warp 1
...
Thread 524 287: elem 524 287      elem 1 048 575  ← último thread, último elemento
```

### 1.5.1 Coalescencia: ¿se mantiene?

**Sí**, y es la razón principal por la que el grid-stride loop se diseña así:

- **Iteración 1**: los 32 threads del warp 0 leen `d_a[0..31]` contiguos → 1 transacción coalescente.
- **Iteración 2**: los 32 threads del warp 0 leen `d_a[524288..524319]` contiguos → 1 transacción coalescente.

> **Punto clave**: aunque cada thread "salta" elementos lejanos entre sus iteraciones, **dentro de cada iteración los threads del warp acceden a posiciones consecutivas**. Por eso el stride es `blockDim × gridDim` y no otra cosa.

## 1.6 Cómo se distribuye el trabajo en la GPU

### 1.6.1 Bloques residentes

```
Bloques residentes por SM = 2048 / 256 = 8 bloques/SM
Bloques residentes totales = 128 × 8 = 1 024 bloques en vuelo
```

### 1.6.2 Threads residentes

```
Threads residentes = 128 × 2 048 = 262 144 threads en vuelo
```

### 1.6.3 Concurrencia real (threads ejecutando por ciclo)

```
Threads/ciclo = 128 SMs × 4 schedulers × 32 = 16 384 threads en lockstep
```

### 1.6.4 Oleadas

```
Bloques lanzados: 2 048
Bloques residentes simultáneos: 1 024
Oleadas necesarias: 2 048 / 1 024 = 2 oleadas
```

## 1.7 Tabla comparativa: los dos enfoques lado a lado

| Métrica | Patrón 1:1 | Patrón grid-stride |
|---|---|---|
| `<<<gridDim, blockDim>>>` | `<<<4096, 256>>>` | `<<<2048, 256>>>` |
| Threads lanzados | 1 048 576 | 524 288 |
| Elementos por thread | 1 | 2 |
| Bucle interno | No | Sí (2 iteraciones) |
| Bloques residentes simultáneos | 1 024 | 1 024 |
| Threads residentes | 262 144 | 262 144 |
| Threads ejecutando/ciclo | 16 384 | 16 384 |
| Oleadas | ~4 | ~2 |
| Lecturas/escrituras totales | 1 048 576 × 3 = 3 145 728 | 1 048 576 × 3 = 3 145 728 |

> **Conclusión**: el trabajo total y la concurrencia real son **idénticos**. Lo que cambia es la **organización**: menos bloques pero cada uno con más trabajo, en lugar de muchos bloques con poco trabajo cada uno. Para una suma de vectores simple los dos enfoques rinden igual, pero el grid-stride se generaliza mejor a problemas más grandes.

## 1.8 Visualización jerárquica del cálculo

```
┌─────────────────────────────────────────────────────────────────┐
│  PROBLEMA: suma de vectores, n = 1 048 576                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ESTRATEGIA: dimensionar el grid para llenar la GPU 2×          │
│  → 2 × 262 144 = 524 288 threads objetivo                       │
│  → 524 288 / 256 = 2 048 bloques                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LANZAMIENTO: <<<2048, 256>>>                                   │
│  → 2 048 bloques × 256 threads = 524 288 threads totales        │
│  → Cada thread procesa 1 048 576 / 524 288 = 2 elementos        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESIDENCIA EN HARDWARE                                         │
│  → 1 024 bloques residentes a la vez (128 SMs × 8 bloques)      │
│  → 262 144 threads en vuelo                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONCURRENCIA REAL ESTRICTA                                     │
│  → 512 warps emitiendo instrucciones por ciclo                  │
│  → 16 384 threads en lockstep en este ciclo                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  EJECUCIÓN POR OLEADAS                                          │
│  → 2 048 bloques / 1 024 residentes = 2 oleadas                 │
│  → Dentro de cada bloque, cada thread itera 2 veces             │
└─────────────────────────────────────────────────────────────────┘
```

## 1.9 Traza conceptual de un thread

Veamos qué hace el **Thread con `idx = 5`**:

```
idx    = 5
stride = 524 288

Iteración 1:
   i = 5
   ¿5 < 1 048 576? SÍ
   Lee d_a[5], d_b[5]
   Calcula d_a[5] + d_b[5]
   Escribe d_c[5]
   i += 524 288 → i = 524 293

Iteración 2:
   i = 524 293
   ¿524 293 < 1 048 576? SÍ
   Lee d_a[524 293], d_b[524 293]
   Calcula d_a[524 293] + d_b[524 293]
   Escribe d_c[524 293]
   i += 524 288 → i = 1 048 581

Iteración 3:
   i = 1 048 581
   ¿1 048 581 < 1 048 576? NO → sale del bucle
```

**Resultado**: el Thread 5 ha procesado los elementos 5 y 524 293, exactamente 2 elementos.

## 1.10 ¿Y si `n` no fuera divisible por `N_threads`?

Pongamos `n = 1 048 600` (24 más que 1M) con el mismo lanzamiento `<<<2048, 256>>>`:

```
N_threads = 524 288
n / N_threads = 1 048 600 / 524 288 = 2 con resto 24

→ Threads 0..23   procesan 3 elementos cada uno (idx, idx+stride, idx+2·stride)
→ Threads 24..524 287 procesan 2 elementos cada uno
```

El reparto ya no es perfectamente uniforme: hay un poco de **divergencia del warp** en la tercera iteración (solo los warps 0 entran a la 3ª iteración, los demás salen). Pero el algoritmo sigue siendo correcto gracias a la condición `i < n` del `for`.

> **El grid-stride loop es robusto a cualquier tamaño de `n`**: el bucle siempre cubre todos los elementos exactos sin desbordamientos ni huecos, da igual la relación entre `n` y el número de threads.

## 1.11 Tres ideas clave para fijar

- **El grid-stride loop desacopla el tamaño del problema del tamaño del grid**. Lanzas un grid del tamaño óptimo para la GPU (típicamente 2-4× los threads residentes máximos), y cada thread itera tantas veces como haga falta para cubrir `n`.
- **La coalescencia se mantiene** porque el stride es exactamente el número total de threads del grid (`blockDim × gridDim`). En cada iteración, los 32 threads del warp acceden a 32 posiciones contiguas → 1 transacción coalescente por iteración.
- **El trabajo total y la concurrencia real no cambian** respecto al patrón 1:1. Lo que cambia es la organización: menos bloques pero cada uno con más trabajo. Esto es preferible para problemas grandes porque generaliza mejor y reduce el overhead de gestión de bloques.