# 1 Características hardware de una GPU: origen de cada dato

## 1.1 Conceptos previos: ¿qué significa "residente", "elegible" y "ejecutando"?

Antes de meternos en los números, hay que tener clarísimos **tres estados** en los que puede estar un thread (o un warp) dentro de la GPU. Confundirlos es la causa más común de malentendidos sobre rendimiento.

### 1.1.1 Threads/warps **residentes** (en vuelo, "in-flight")

> **Threads que el hardware ha aceptado y está gestionando simultáneamente**, con sus registros reservados y su estado vivo en el SM. No necesariamente están ejecutándose ahora mismo; pueden estar esperando memoria, esperando un `__syncthreads()`, o simplemente sin haber sido elegidos por un scheduler en este ciclo.

**Características:**

- Sus **registros están reservados** en el banco del SM. Nadie más puede usarlos hasta que el thread termine.
- Su **memoria compartida está reservada** (si su bloque la usa).
- Su `threadIdx`, `blockIdx`, contadores de programa, etc., están almacenados en el SM.
- Pueden quedarse en este estado **muchos ciclos** mientras esperan algo.

**Limitado por:**

- `max_threads_por_SM` (típicamente 2 048).
- `max_bloques_por_SM` (típicamente 32).
- Registros disponibles en el SM.
- Memoria compartida disponible en el SM.

### 1.1.2 Warps **elegibles**

> **Warps residentes que en este ciclo concreto están listos para ejecutar su siguiente instrucción**. Es un subconjunto dinámico de los residentes que cambia ciclo a ciclo.

**Un warp NO es elegible si:**

- Está **esperando un dato de memoria global** (latencia ~400 ciclos).
- Está **esperando una instrucción de cómputo** que aún no terminó (p. ej. división, raíz cuadrada).
- Está **detenido en un `__syncthreads()`** esperando a los demás warps del bloque.
- Está en una rama de divergencia mientras otra rama se ejecuta.

**Limitado por:**

- Disponibilidad de datos (memoria) y sincronización.
- No hay un máximo numérico fijo; es **dinámico**.

### 1.1.3 Warps/threads **en ejecución** (este ciclo)

> **Los warps que los warp schedulers han elegido en este ciclo concreto para emitir una instrucción a los CUDA cores**. Es el conjunto más pequeño y el de "concurrencia real estricta".

**Características:**

- En cada ciclo, cada SM tiene **4 warp schedulers**. Cada scheduler elige UN warp elegible y le hace ejecutar UNA instrucción.
- Por tanto, **4 warps × 32 threads = 128 threads** ejecutan por ciclo y por SM.
- En la GPU completa: `N_SMs × 4 × 32 = 16 384 threads` en un ciclo (para 128 SMs).

**Limitado por:**

- Número de warp schedulers por SM (4 fijo en arquitecturas modernas).
- Número de SMs.

### 1.1.4 Visualización de los tres estados

```
┌──────────────────────────────────────────────────────────────────────┐
│  COLA DE LANZAMIENTO (threads pedidos en <<<...>>>)                  │
│  Tantos como quieras (hasta ~2·10⁹)                                  │
│  → esperando a que un SM tenga sitio para su bloque                  │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ entran cuando un SM libera espacio
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RESIDENTES en el SM (con registros y shared reservados)             │
│  Hasta 2 048 threads/SM × 128 SMs = 262 144 threads simultáneos      │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  ELEGIBLES (residentes que ahora pueden ejecutar)           │   │
│   │  Subconjunto dinámico (los que no esperan memoria/sync)     │   │
│   │                                                             │   │
│   │   ┌──────────────────────────────────────────────────┐      │   │
│   │   │  EJECUTANDO ESTE CICLO                           │      │   │
│   │   │  4 warps por SM × 128 SMs = 512 warps            │      │   │
│   │   │  = 16 384 threads emitiendo instrucción AHORA    │      │   │
│   │   └──────────────────────────────────────────────────┘      │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1.5 Por qué importan estos tres niveles

Cada uno está limitado por algo distinto y se optimiza de forma distinta:

| Estado | Maximizado por | Métrica relacionada |
|---|---|---|
| **Lanzados** | Tu elección de `gridDim × blockDim` | Suficientes para llenar la GPU |
| **Residentes** | Bajo uso de registros y shared / `blockDim` adecuado | **Ocupación** (occupancy) |
| **Elegibles** | Alta ocupación + buen mix de instrucciones (memoria/cómputo) | "Eligible warps per cycle" en Nsight |
| **Ejecutando** | Fijo por hardware: `SMs × schedulers × 32` | Es el techo absoluto de concurrencia |

> **Regla mental**: tener muchos threads residentes es bueno porque aumenta la probabilidad de que siempre haya warps elegibles. Los warp schedulers entonces tienen opciones y los cores no se quedan ociosos. Esto es el **latency hiding**, la idea fundamental que hace que las GPUs sean rápidas.

---

## 1.2 Leyenda de orígenes

| Símbolo | Origen | Significado |
|---|---|---|
| 🏭 | **Fijo por arquitectura** | Lo decide NVIDIA al diseñar el chip; consultable en el datasheet de cada GPU |
| 🔒 | **Límite hardware constante** | Igual en todas las GPUs NVIDIA modernas (no varía por modelo) |
| 🧑‍💻 | **Decisión del programador** | Lo eliges tú en `<<<...>>>` o al escribir el kernel |
| 🧮 | **Calculado** | Se deriva de otros valores con una fórmula |
| 📊 | **Medido/observado** | Se obtiene con herramientas (`-Xptxas=-v`, `nvprof`, Nsight) |

---

## 1.3 Tabla principal: parámetros hardware y de qué dependen

### 1.3.1 Estructura: SMs y cores

| Parámetro | Valor | Origen | De dónde sale |
|---|---|---|---|
| Streaming Multiprocessors (SMs) | 128 | 🏭 | Especificación del chip (RTX 4090: 128 SMs, H100: 132, RTX 3060: 28). Consulta el datasheet |
| CUDA cores por SM | 128 | 🏭 | Depende de la arquitectura: Pascal 128, Volta/Turing 64, Ampere/Ada 128, Hopper 128 |
| **CUDA cores totales** | **16 384** | 🧮 | `N_SMs × cores_por_SM = 128 × 128` |

### 1.3.2 Planificación: warp schedulers y warps

| Parámetro | Valor | Origen | De dónde sale |
|---|---|---|---|
| Warp schedulers por SM | 4 | 🏭 | Fijo por arquitectura (Volta/Turing/Ampere/Ada/Hopper = 4; Pascal = 2) |
| **Tamaño del warp** | **32 threads** | 🔒 | **Constante en todas las GPUs NVIDIA** (no cambia entre modelos) |

### 1.3.3 Límites de bloque y SM (afectan a la RESIDENCIA)

| Parámetro | Valor | Origen | De dónde sale |
|---|---|---|---|
| Max threads por bloque | 1 024 | 🔒 | Constante hardware (todas las GPUs CUDA modernas) |
| Max bloques residentes por SM | 32 | 🏭 | Específico de arquitectura (Ampere/Ada: 16-32; Hopper: 32) |
| Max threads residentes por SM | 2 048 | 🏭 | Específico de arquitectura (Pascal/Volta/Turing/Ampere: 2048; Ada: 1536; Hopper: 2048) |
| **Max warps residentes por SM** | **64** | 🧮 | `max_threads_por_SM / warp_size = 2048 / 32` |

### 1.3.4 Recursos: registros (limitan los RESIDENTES)

| Parámetro | Valor | Origen | De dónde sale |
|---|---|---|---|
| Registros por SM | 65 536 (× 32 bits) | 🏭 | Fijo por arquitectura (constante desde Maxwell hasta Hopper) |
| Max registros por thread | 255 | 🔒 | Constante hardware |
| **Registros por thread con 100% residencia** | **32** | 🧮 | `registros_por_SM / max_threads_por_SM = 65 536 / 2 048` |
| Registros usados por TU kernel | variable | 📊 | Lo reporta `nvcc -Xptxas=-v` al compilar |

### 1.3.5 Recursos: memoria compartida (también limita los RESIDENTES)

| Parámetro | Valor | Origen | De dónde sale |
|---|---|---|---|
| Memoria compartida por SM | 100 KB | 🏭 | Específico de arquitectura (Ampere: 100-164 KB; Ada: 100 KB; Hopper: 228 KB) |
| Memoria compartida por bloque | hasta 99 KB | 🏭 | Específico de arquitectura (típicamente shared/SM - 1 KB reservado) |
| Shared memory usada por TU kernel | variable | 📊 | Lo reporta `nvcc -Xptxas=-v` |

### 1.3.6 Jerarquía de memoria

| Parámetro | Valor | Origen | De dónde sale |
|---|---|---|---|
| Cache L1 por SM | 128 KB | 🏭 | Específico de arquitectura (compartido con shared en Ampere/Ada) |
| Cache L2 total | 96 MB | 🏭 | Específico de cada modelo (RTX 4090: 72 MB; H100: 50 MB; A100: 40 MB) |
| Memoria global (DRAM) | 24 GB GDDR6X | 🏭 | Específico de cada modelo (varía entre 8 GB y 80 GB) |
| Ancho de banda DRAM | ~1 TB/s | 🏭 | Específico de cada modelo (RTX 4090: 1 TB/s; H100: 3 TB/s; RTX 3060: 360 GB/s) |

### 1.3.7 Velocidad

| Parámetro | Valor | Origen | De dónde sale |
|---|---|---|---|
| Frecuencia de reloj | ~2.5 GHz | 🏭 | Específico de cada modelo (varía con boost y carga térmica) |

---

## 1.4 Los tres niveles de concurrencia: cómo se calcula cada uno

| Nivel | Valor | Origen | Fórmula | Estado |
|---|---|---|---|---|
| **Threads lanzables (en cola)** | ~2·10⁹ | 🔒 | Límite de `gridDim.x = 2³¹ - 1` (constante hardware) | Esperando |
| **Threads residentes totales** | 262 144 | 🧮 | `N_SMs × max_threads_por_SM = 128 × 2 048` | Residentes |
| **Warps residentes totales** | 8 192 | 🧮 | `N_SMs × max_warps_por_SM = 128 × 64` | Residentes |
| **Warps elegibles por SM** | variable | 📊 | Dinámico (mide Nsight con `eligible_warps_per_cycle`) | Elegibles |
| **Warps emitiendo por ciclo** | 512 | 🧮 | `N_SMs × schedulers = 128 × 4` | Ejecutando |
| **Threads ejecutando por ciclo (toda la GPU)** | 16 384 | 🧮 | `N_SMs × schedulers × warp_size = 128 × 4 × 32` | Ejecutando |
| **Ratio residentes / ejecutando** | 16× | 🧮 | `262 144 / 16 384` (cuántos warps en la piscina por cada uno emitiendo) | — |

> **Por qué el ratio 16× es importante**: significa que cada warp scheduler tiene ~16 warps residentes entre los que elegir. Como cada acceso a memoria global tarda ~400 ciclos y una instrucción dura ~1 ciclo, **necesitas al menos `400 / N_schedulers` warps residentes** para esconder latencia perfectamente. Con 4 schedulers, eso son 100 warps residentes por SM, pero solo caben 64. Por eso en kernels memory-bound, el latency hiding no es perfecto y la DRAM acaba siendo el cuello de botella.

---

## 1.5 Configuración de tu lanzamiento: lo que tú decides

| Parámetro | Origen | De dónde sale |
|---|---|---|
| `blockDim` (threads por bloque) | 🧑‍💻 | **Tú lo eliges** en `<<<gridDim, blockDim>>>`. Restricciones: múltiplo de 32, ≤ 1024 |
| `gridDim` (número de bloques) | 🧑‍💻 | **Tú lo eliges**. Típicamente calculado: `ceil(n / blockDim)` o un múltiplo de `N_SMs` |
| Threads lanzados | 🧮 | `gridDim × blockDim` |
| Memoria compartida solicitada | 🧑‍💻 | Tú la declaras con `__shared__` en el kernel |
| Registros usados | 📊 | El compilador decide; tú influyes con la complejidad del kernel y `__launch_bounds__` |

---

## 1.6 Métricas derivadas: lo que se calcula a partir de tu kernel

Estos valores **dependen de las decisiones del programador combinadas con el hardware**:

| Métrica | Origen | Fórmula | Estado afectado |
|---|---|---|---|
| Bloques residentes por SM | 🧮 | `min(max_bloques_por_SM, max_threads_por_SM/blockDim, registros_por_SM/(blockDim × regs_por_thread), shared_por_SM/shared_por_bloque)` | Residentes |
| Threads residentes por SM (efectivos) | 🧮 | `bloques_residentes_por_SM × blockDim` | Residentes |
| **Ocupación (occupancy)** | 🧮 | `threads_residentes_efectivos / max_threads_por_SM` | Residentes |
| Oleadas de procesamiento | 🧮 | `gridDim / bloques_residentes_totales` | Lanzados |
| Iteraciones por thread (grid-stride) | 🧮 | `ceil(n / threads_lanzados)` | Por thread |

> **Nota importante**: la fórmula de "bloques residentes por SM" es un `min` de cuatro restricciones (límite de bloques, límite de threads, límite de registros, límite de shared). El que tenga el valor más bajo es el cuello de botella de tu kernel.

---

## 1.7 Aplicación al ejemplo de suma de vectores 1M (grid-stride)

| Parámetro | Valor | Origen | De dónde | Estado |
|---|---|---|---|---|
| `n` (tamaño del vector) | 1 048 576 | 🧑‍💻 | Decisión del programador (problema dado) | — |
| `blockDim` | 256 | 🧑‍💻 | Elegido (múltiplo de 32 estándar) | — |
| `gridDim` | 2 048 | 🧮 | `2 × N_SMs × bloques_por_SM = 2 × 128 × 8` (factor 2 para llenar 2× la GPU) | — |
| Threads lanzados | 524 288 | 🧮 | `gridDim × blockDim = 2048 × 256` | Lanzados |
| Iteraciones por thread | 2 | 🧮 | `n / threads_lanzados = 1 048 576 / 524 288` | — |
| `stride` del bucle | 524 288 | 🧮 | `blockDim × gridDim` (igual a threads_lanzados) | — |
| Bloques residentes por SM | 8 | 🧮 | `max_threads_por_SM / blockDim = 2 048 / 256` (asumiendo no limita registros ni shared) | Residentes |
| Bloques residentes totales | 1 024 | 🧮 | `N_SMs × bloques_por_SM = 128 × 8` | Residentes |
| **Threads residentes totales** | **262 144** | 🧮 | `N_SMs × max_threads_por_SM = 128 × 2 048` | **Residentes** |
| **Threads ejecutando por ciclo** | **16 384** | 🧮 | `N_SMs × schedulers × warp_size = 128 × 4 × 32` | **Ejecutando** |
| Oleadas necesarias | 2 | 🧮 | `gridDim / bloques_residentes_totales = 2 048 / 1 024` | — |
| Ocupación | 100% | 🧮 | `(bloques × blockDim) / max_threads_SM = (8 × 256) / 2 048` | Residentes |

---

## 1.8 Cómo obtener cada categoría en la práctica

### 1.8.1 🏭 Específicos de arquitectura/modelo

**Cómo consultarlos**:

```bash
# Programáticamente desde código
deviceQuery       # ejemplo del CUDA samples
nvidia-smi -q     # información detallada del driver

# Desde Python
import torch
torch.cuda.get_device_properties(0)
```

**Lista de los más útiles**:

```
multiProcessorCount         → N_SMs
maxThreadsPerMultiProcessor → max threads por SM
maxThreadsPerBlock          → 1024 (constante)
regsPerMultiprocessor       → registros por SM
sharedMemPerMultiprocessor  → shared por SM
warpSize                    → 32 (constante)
```

### 1.8.2 🔒 Constantes hardware (todas las GPUs NVIDIA)

| Constante | Valor |
|---|---|
| Tamaño del warp | 32 |
| Max threads por bloque | 1 024 |
| Max registros por thread | 255 |
| Max `gridDim.x` | 2³¹ - 1 |
| Max `gridDim.y`, `.z` | 65 535 |

Estos NO cambian entre modelos. Puedes asumirlos siempre.

### 1.8.3 🧑‍💻 Decisiones del programador

Lo que tú controlas en cada lanzamiento:

```cuda
kernel<<<gridDim, blockDim, sharedMemBytes>>>(args);
```

Y dentro del kernel, qué variables declaras como `__shared__`, qué algoritmo usas, etc.

### 1.8.4 📊 Métricas medidas del kernel

**Al compilar**:

```bash
nvcc -Xptxas=-v -lineinfo kernel.cu -o kernel
```

Te dice (entre otras cosas):

```
ptxas info: Used 32 registers, 4096 bytes smem, 360 bytes cmem[0]
```

**Al ejecutar**:

```bash
nvprof ./mi_programa                       # antiguo
ncu --metrics achieved_occupancy ./mi_programa   # moderno (Nsight Compute)
```

Métricas clave para profiling:

| Métrica | Qué mide | Estado |
|---|---|---|
| `achieved_occupancy` | Ocupación real (no teórica) | Residentes |
| `eligible_warps_per_cycle` | Cuántos warps están listos por ciclo | Elegibles |
| `gld_efficiency` | % de bytes útiles en lecturas globales (coalescencia) | — |
| `gst_efficiency` | Idem para escrituras | — |
| `dram_read_throughput` | GB/s real leídos de DRAM | — |
| `warp_execution_efficiency` | % de threads activos por warp (mide divergencia) | Ejecutando |

---

## 1.9 Tabla maestra: todos los parámetros con su origen

| # | Parámetro | Valor (ejemplo) | Origen | Cómo se obtiene | Estado |
|---|---|---|---|---|---|
| 1 | SMs | 128 | 🏭 | Datasheet de la GPU | — |
| 2 | Cores/SM | 128 | 🏭 | Datasheet (depende de arquitectura) | — |
| 3 | Cores totales | 16 384 | 🧮 | `SMs × cores/SM` | — |
| 4 | Schedulers/SM | 4 | 🏭 | Datasheet (4 en Volta+) | — |
| 5 | Warp size | 32 | 🔒 | Constante NVIDIA | — |
| 6 | Max threads/block | 1 024 | 🔒 | Constante hardware | — |
| 7 | Max blocks/SM | 32 | 🏭 | Datasheet (varía por arquitectura) | Residentes |
| 8 | Max threads/SM | 2 048 | 🏭 | Datasheet | Residentes |
| 9 | Max warps/SM | 64 | 🧮 | `max threads/SM ÷ 32` | Residentes |
| 10 | Registers/SM | 65 536 | 🏭 | Datasheet | Residentes |
| 11 | Max regs/thread | 255 | 🔒 | Constante hardware | — |
| 12 | Regs/thread @ 100% occ | 32 | 🧮 | `regs/SM ÷ max threads/SM` | Residentes |
| 13 | Shared/SM | 100 KB | 🏭 | Datasheet | Residentes |
| 14 | Shared/block | hasta 99 KB | 🏭 | Datasheet | Residentes |
| 15 | L1/SM | 128 KB | 🏭 | Datasheet | — |
| 16 | L2 total | 96 MB | 🏭 | Datasheet de cada modelo | — |
| 17 | DRAM | 24 GB | 🏭 | Datasheet de cada modelo | — |
| 18 | Ancho de banda | 1 TB/s | 🏭 | Datasheet | — |
| 19 | Threads ejecutando/ciclo | 16 384 | 🧮 | `SMs × schedulers × 32` | **Ejecutando** |
| 20 | Warps/ciclo | 512 | 🧮 | `SMs × schedulers` | **Ejecutando** |
| 21 | Threads residentes totales | 262 144 | 🧮 | `SMs × max threads/SM` | **Residentes** |
| 22 | Threads lanzables | ~2·10⁹ | 🔒 | Límite de `gridDim` | Lanzados |
| 23 | `blockDim` | 256 | 🧑‍💻 | Tu elección | — |
| 24 | `gridDim` | depende | 🧑‍💻🧮 | Tu elección o derivado | — |
| 25 | Threads lanzados | depende | 🧮 | `gridDim × blockDim` | Lanzados |
| 26 | Regs por thread (kernel) | variable | 📊 | `nvcc -Xptxas=-v` | Residentes |
| 27 | Shared por bloque (kernel) | variable | 📊 | `nvcc -Xptxas=-v` | Residentes |
| 28 | Bloques residentes/SM | depende | 🧮 | `min` de 4 restricciones | Residentes |
| 29 | Ocupación | depende | 🧮 | `threads_efectivos/SM ÷ 2 048` | Residentes |
| 30 | Oleadas | depende | 🧮 | `gridDim ÷ bloques residentes totales` | Lanzados |
| 31 | Warps elegibles | dinámico | 📊 | Nsight Compute | **Elegibles** |

---
