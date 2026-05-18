# 1 Ejemplo corregido: Suma de vectores de 1M datos

## 1.1 Planteamiento del problema

**Operación:** suma elemento a elemento de dos vectores `c[i] = a[i] + b[i]`.

**Tamaño:** `n = 1M = 2²⁰ = 1 048 576` elementos.

**Hardware asumido:** GPU con **128 SMs** y **4 warp schedulers por SM**.

## 1.2 Configuración de lanzamiento del kernel

### 1.2.1 Geometría del grid

```
Grid 1D, blockDim = 256 hilos/bloque (múltiplo de 32, valor estándar)
```

### 1.2.2 Cálculo del número de bloques

Necesitamos al menos un thread por elemento (patrón "1 thread = 1 elemento"):

```
nblocks = ceil(n / blockDim)
        = (1 048 576 + 256 - 1) / 256
        = 1 048 831 / 256
        = 4 096 bloques
```

### 1.2.3 Llamada al kernel

```cuda
compute_kernel<<<4096, 256>>>(n, d_a, d_b, d_c);
```

### 1.2.4 Threads totales lanzados (en cola)

```
n_total = nblocks × blockDim = 4 096 × 256 = 1 048 576 threads
```

Coincide exactamente con `n` → no sobran threads en este caso particular (pero el `if (idx < n)` en el kernel sigue siendo buena práctica).

## 1.3 Cómo se distribuye el trabajo en la GPU

Aquí está el punto que más confunde: **los bloques NO se mapean 1-a-1 con los SMs**. La relación correcta es:

> **Cada bloque se asigna a UN SM, pero un SM alberga VARIOS bloques residentes simultáneamente.**

### 1.3.1 Cuántos bloques caben por SM

Con la arquitectura asumida (típica moderna, 2048 threads/SM como máximo):

```
Bloques residentes por SM ≈ max_threads_por_SM / blockDim
                          = 2048 / 256
                          = 8 bloques por SM
```

### 1.3.2 Bloques residentes en toda la GPU simultáneamente

```
Bloques residentes totales = N_SMs × bloques_por_SM
                           = 128 × 8
                           = 1 024 bloques en vuelo a la vez
```

### 1.3.3 Threads residentes en toda la GPU

```
Threads residentes = N_SMs × max_threads_por_SM
                   = 128 × 2 048
                   = 262 144 threads "vivos" simultáneamente
```

## 1.4 Concurrencia real: cuántos threads ejecutan por ciclo

Aquí está el cálculo que tenías mal. La clave es que **cada warp scheduler emite una instrucción a un warp completo (32 threads)**, no a un solo thread:

```
Warps ejecutando por ciclo  = N_SMs × N_schedulers
                            = 128 × 4
                            = 512 warps por ciclo

Threads ejecutando por ciclo = 512 warps × 32 threads/warp
                             = 16 384 threads en lockstep
```

> **El error común es olvidarse del ×32**. Si en un cálculo de "threads concurrentes" no aparece el factor 32, casi seguro que estás contando warps por error.

## 1.5 Procesamiento por oleadas

Como hay **4 096 bloques lanzados** y caben **1 024 a la vez**, la GPU procesa el kernel en aproximadamente:

```
Oleadas necesarias ≈ nblocks_lanzados / bloques_residentes
                   = 4 096 / 1 024
                   ≈ 4 oleadas
```

```
Oleada 1: bloques    0..1023  →  128 SMs × 8 bloques cada uno
Oleada 2: bloques 1024..2047  →  cuando algunos de la oleada 1 terminan
Oleada 3: bloques 2048..3071
Oleada 4: bloques 3072..4095
```

En la práctica, las oleadas se **solapan** (no se espera a que toda una oleada termine para empezar la siguiente): en cuanto un bloque libera un SM, otro entra. Es más una cinta transportadora continua que oleadas discretas.

## 1.6 Resumen en una tabla: los tres niveles de concurrencia

| Nivel | Significado | Número |
|---|---|---|
| **Threads lanzados** (en cola) | Lo que pides en `<<<...>>>` | **1 048 576** |
| **Bloques residentes simultáneos** | Bloques compartiendo los SMs a la vez | **~1 024** |
| **Threads residentes** (en vuelo) | Threads gestionados simultáneamente por el hardware | **~262 144** |
| **Warps ejecutando por ciclo** | Warps que emiten una instrucción en este ciclo | **512** |
| **Threads ejecutando por ciclo** | Threads en lockstep dentro de los warps activos | **16 384** ← *concurrencia real estricta* |
| **Oleadas necesarias** | Pasadas para procesar todos los bloques | **~4** |

## 1.7 Visualización jerárquica del cálculo

```
┌─────────────────────────────────────────────────────────────────┐
│  PROBLEMA: suma de vectores, n = 1 048 576                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LANZAMIENTO: <<<4096, 256>>>                                   │
│  → 4 096 bloques de 256 threads cada uno                        │
│  → 1 048 576 threads TOTALES en cola                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESIDENCIA EN HARDWARE (en vuelo a la vez)                     │
│                                                                 │
│  Por SM:   8 bloques × 256 threads = 2 048 threads/SM           │
│  Total:    128 SMs × 8 bloques     = 1 024 bloques residentes   │
│            128 SMs × 2 048 threads = 262 144 threads residentes │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  EJECUCIÓN EN ESTE CICLO (concurrencia real estricta)           │
│                                                                 │
│  Por SM:   4 schedulers × 32 threads = 128 threads/ciclo/SM     │
│  Total:    128 SMs × 128 threads     = 16 384 threads/ciclo     │
│            (= 512 warps emitiendo a la vez)                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  RATIO RESIDENTES/EJECUTANDO: 262 144 / 16 384 = 16×            │
│  → cada warp scheduler tiene ~16 warps "en la piscina"          │
│  → suficiente para esconder latencias de memoria global         │
└─────────────────────────────────────────────────────────────────┘
```

## 1.8 Verificación de la eficiencia del diseño

Comprobaciones rápidas de que el diseño es bueno:

| Comprobación | Valor | ¿Bien? |
|---|---|---|
| `blockDim` múltiplo de 32 | 256 = 8 warps | ✅ |
| `nblocks` mucho mayor que `N_SMs` | 4096 ≫ 128 | ✅ (32× más bloques que SMs) |
| Ocupación máxima por SM | 8 bloques × 256 = 2048 threads = 100% | ✅ |
| Threads residentes / ejecutando | 16× | ✅ (buen latency hiding) |

## 1.9 Tres ideas clave para fijar

- **El número de bloques NO está ligado al número de SMs**. Lanzas tantos bloques como elementos / blockDim, y la GPU los procesa por oleadas. Cuantos más bloques (siempre que tengan trabajo útil), mejor reparto y más tolerancia al desbalanceo.
- **La concurrencia "real" (16 384 threads/ciclo) es mucho menor que los threads lanzados (1 048 576)**. La GPU es paralela pero finita; la magia está en rotar warps rápidamente para esconder latencias de memoria.
- **Multiplica siempre por 32 al calcular threads ejecutando**. El warp scheduler emite a un warp entero (32 threads). Si tu cálculo da 512 o 1024 "threads concurrentes", probablemente sean warps y te falta el `× 32`.