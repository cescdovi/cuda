# 1 Flujo general del Ejercicio 5: Producto escalar en CUDA (con visión a nivel de warp)

## 1.1 Definir datos de entrada (en CPU)

```
n = 80
h_x = [1.0, 2.0, 3.0, ..., 80.0]    ← los números del 1 al 80
h_y = [1.0, 1.0, 1.0, ..., 1.0]     ← 80 unos
```

**Resultado esperado:**

```
1·1 + 2·1 + 3·1 + ... + 80·1 = 80·81/2 = 3240.0
```

> Usamos `n = 80` para que cada thread tenga que iterar varias veces dentro del kernel (los threads 0..15 iterarán 3 veces, los 16..31 iterarán 2 veces). Esto nos permite ver cómo avanza el warp en lockstep a lo largo de varias iteraciones.

---

## 1.2 Reservar memoria en GPU

```cuda
cudaMalloc(&d_x, 80 * sizeof(float));      // 320 bytes
cudaMalloc(&d_y, 80 * sizeof(float));      // 320 bytes
cudaMalloc(&d_v, 32 * sizeof(float));      // 128 bytes (1 parcial por thread del kernel 1)
cudaMalloc(&d_result, 1 * sizeof(float));  // 4 bytes  (resultado escalar)
```

**Estado de la memoria GPU tras este paso:**

```
d_x = [?, ?, ?, ..., ?]                 (80 posiciones, basura)
d_y = [?, ?, ?, ..., ?]                 (80 posiciones, basura)
d_v = [?, ?, ?, ..., ?]                 (32 posiciones, basura)
d_result = [?]                          (basura)
```

> **Por qué `d_v` tiene tamaño 32**: el Kernel 1 se lanza con 32 threads, y cada thread escribe **una** suma parcial en `v[threadIdx.x]`. Por tanto, `tamaño(d_v) = nº de threads del Kernel 1`.

---

## 1.3 Copiar datos CPU → GPU

```cuda
cudaMemcpy(d_x, h_x, 80*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, 80*sizeof(float), cudaMemcpyHostToDevice);
```

**Estado:**

```
d_x = [1.0, 2.0, 3.0, ..., 80.0]
d_y = [1.0, 1.0, 1.0, ..., 1.0]
d_v = [?, ?, ?, ..., ?]                 (sigue con basura)
d_result = [?]                          (sigue con basura)
```

---

## 1.4 Concepto previo: ¿qué es un warp?

Antes de lanzar el kernel, es importante entender la **unidad fundamental de ejecución en GPU**:

> **Un warp es un grupo de 32 threads que ejecutan la MISMA instrucción en el MISMO ciclo de reloj.**

- Las GPUs NVIDIA siempre agrupan los threads en warps de 32. No es configurable.
- Todos los threads de un warp avanzan en **lockstep** (paso firme): instrucción a instrucción, todos al unísono.
- Cada thread tiene **su propio estado** (sus propios registros, su propia `t`, su propia `a`, su propia `i`), pero **todos ejecutan la misma instrucción** a la vez sobre datos distintos.
- Esto es el modelo **SIMT** (Single Instruction, Multiple Threads).

### 1.4.1 ¿Qué pasa cuando lanzas más de 32 threads?

Si lanzas un bloque con `blockDim = 256`, ese bloque se divide en `256 / 32 = 8 warps`. El SM ejecuta varios warps "a la vez" gracias a sus warp schedulers (típicamente 4 por SM), pero la unidad mínima sigue siendo el warp de 32.

### 1.4.2 En este ejercicio

```cuda
compute_kernel1<<<1, 32>>>(...);
```

- 1 bloque × 32 threads = **1 warp único**.
- Los 32 threads del warp avanzarán todos a la vez, instrucción por instrucción.
- No hay otros warps que se intercalen → trazar el kernel se reduce a trazar las instrucciones de ese único warp.

---

## 1.5 Lanzar el Kernel 1

### 1.5.1 Sintaxis y nº de threads

```cuda
compute_kernel1<<<1, 32>>>(80, d_x, d_y, d_v);
```

- **Nº total de threads lanzados** = `gridDim × blockDim = 1 × 32 = 32`.
- Esos 32 threads forman **exactamente 1 warp**.

### 1.5.2 Código del kernel

```cuda
__global__ void compute_kernel1(const unsigned int n, float *d_x, float *d_y, float v[32]) {
    int t = threadIdx.x;            // identidad del thread (0..31)
    float a = 0.0f;                 // acumulador local (privado por thread)
    for (int i = t; i < n; i += 32) {
        a += d_x[i] * d_y[i];
    }
    v[t] = a;                       // cada thread escribe su parcial en SU casilla
}
```

> **Idea clave**: los 32 threads ejecutan el mismo código, pero cada uno usa su `t` para acceder a posiciones distintas del vector. El reparto del trabajo emerge automáticamente del valor de `threadIdx.x`.

### 1.5.3 Ejecución del warp ciclo a ciclo

Aquí está la diferencia con una traza thread a thread: en lugar de seguir UN thread completo y luego otro, vamos a ver **qué hace el warp entero en cada ciclo de reloj**. Los 32 threads avanzan al unísono.

#### 1.5.3.1 Ciclos de inicialización

```
Ciclo 1: TODOS los 32 threads ejecutan "int t = threadIdx.x"
─────────────────────────────────────────────────────────────
Thread 0  → su t = 0
Thread 1  → su t = 1
Thread 2  → su t = 2
...
Thread 31 → su t = 31

Ciclo 2: TODOS ejecutan "float a = 0.0f"
─────────────────────────────────────────────────────────────
Los 32 inicializan SU PROPIA a a 0.0

Ciclo 3: TODOS ejecutan "i = t" (inicialización del for)
─────────────────────────────────────────────────────────────
Thread 0  → i = 0
Thread 1  → i = 1
Thread 2  → i = 2
...
Thread 31 → i = 31
```

#### 1.5.3.2 Primera iteración del bucle (ciclos 4-9)

```
Ciclo 4: TODOS ejecutan "¿i < 80?"
─────────────────────────────────────────────────────────────
Thread 0:  ¿0 < 80?  SÍ
Thread 1:  ¿1 < 80?  SÍ
...
Thread 31: ¿31 < 80? SÍ
→ Todos siguen al cuerpo (no hay divergencia)

Ciclo 5: TODOS ejecutan "load d_x[i]"
─────────────────────────────────────────────────────────────
Thread 0:  carga d_x[0]  = 1.0
Thread 1:  carga d_x[1]  = 2.0
Thread 2:  carga d_x[2]  = 3.0
...
Thread 31: carga d_x[31] = 32.0

⚡ ACCESO COALESCENTE: los 32 accesos son a posiciones consecutivas
   → 1 sola transacción de 128 bytes (32 floats × 4 bytes) para los 32 threads

Ciclo 6: TODOS ejecutan "load d_y[i]"
─────────────────────────────────────────────────────────────
(mismo patrón coalescente; los 32 cargan d_y[0..31] = todo unos)

Ciclo 7: TODOS ejecutan "multiply"
─────────────────────────────────────────────────────────────
Thread 0:  1.0 * 1.0 = 1.0
Thread 1:  2.0 * 1.0 = 2.0
...
Thread 31: 32.0 * 1.0 = 32.0

Ciclo 8: TODOS ejecutan "a += producto"
─────────────────────────────────────────────────────────────
Thread 0:  a = 0 + 1.0  = 1.0
Thread 1:  a = 0 + 2.0  = 2.0
...
Thread 31: a = 0 + 32.0 = 32.0

Ciclo 9: TODOS ejecutan "i += 32"
─────────────────────────────────────────────────────────────
Thread 0:  i = 0 + 32 = 32
Thread 1:  i = 1 + 32 = 33
...
Thread 31: i = 31 + 32 = 63
```

#### 1.5.3.3 Segunda iteración del bucle (ciclos 10-15)

```
Ciclo 10: TODOS ejecutan "¿i < 80?"
─────────────────────────────────────────────────────────────
Thread 0:  ¿32 < 80? SÍ
Thread 1:  ¿33 < 80? SÍ
...
Thread 31: ¿63 < 80? SÍ
→ Todos siguen (sin divergencia todavía)

Ciclo 11: TODOS ejecutan "load d_x[i]"
─────────────────────────────────────────────────────────────
Thread 0:  carga d_x[32] = 33.0
Thread 1:  carga d_x[33] = 34.0
...
Thread 31: carga d_x[63] = 64.0

⚡ Otra vez COALESCENTE: d_x[32..63] son contiguos

Ciclos 12-14: load d_y, multiply, accumulate
─────────────────────────────────────────────────────────────
Thread 0:  a = 1.0  + 33.0 = 34.0
Thread 1:  a = 2.0  + 34.0 = 36.0
...
Thread 31: a = 32.0 + 64.0 = 96.0

Ciclo 15: TODOS ejecutan "i += 32"
─────────────────────────────────────────────────────────────
Thread 0:  i = 64
Thread 1:  i = 65
...
Thread 31: i = 95
```

#### 1.5.3.4 Tercera iteración: ¡DIVERGENCIA DEL WARP!

Aquí es donde la cosa se pone interesante. Vamos a verlo despacio:

```
Ciclo 16: TODOS ejecutan "¿i < 80?"
─────────────────────────────────────────────────────────────
Thread 0:  ¿64 < 80? SÍ  ← quiere seguir
Thread 1:  ¿65 < 80? SÍ  ← quiere seguir
...
Thread 15: ¿79 < 80? SÍ  ← quiere seguir (justo a tiempo)
Thread 16: ¿80 < 80? NO  ← quiere salir
Thread 17: ¿81 < 80? NO  ← quiere salir
...
Thread 31: ¿95 < 80? NO  ← quiere salir

⚠️ El warp DIVERGE: 16 threads quieren entrar al cuerpo, 16 quieren salir
```

¿Qué hace el hardware? **No puede ejecutar dos caminos al mismo tiempo**. Tiene que hacer un pase con cada grupo:

```
PASE 1 (ciclos 17-21): ejecuta el cuerpo del bucle
─────────────────────────────────────────────────────────────
ACTIVOS:    threads 0..15
INACTIVOS:  threads 16..31 (marcados como "masked out", no hacen nada)

Ciclo 17: load d_x[i] solo para threads 0..15
   Thread 0:  carga d_x[64] = 65.0
   Thread 1:  carga d_x[65] = 66.0
   ...
   Thread 15: carga d_x[79] = 80.0
   Threads 16..31: ⚫ inactivos, no participan en este load

Ciclo 18: load d_y[i] (solo 0..15 activos)
Ciclo 19: multiply (solo 0..15 activos)
Ciclo 20: a += producto
   Thread 0:  a = 34.0 + 65.0 = 99.0
   Thread 1:  a = 36.0 + 66.0 = 102.0
   ...
   Thread 15: a = 64.0 + 80.0 = 144.0

Ciclo 21: i += 32
   Thread 0:  i = 96
   Thread 15: i = 111

Ciclo 22: ¿i < 80? para threads 0..15
   Todos NO (96..111 >= 80) → salen del bucle también

→ EL WARP SE REUNIFICA: todos los 32 threads han salido del bucle
```

> **Punto clave**: durante los ciclos 17-21, los threads 16..31 **están "ahí"** (no se han ido a otro lado, ni han acelerado), pero **no producen resultados**. Es como si estuvieran "dormidos". Se llama **predicación**: el hardware ejecuta la instrucción para todo el warp pero descarta los resultados de los threads inactivos. Es trabajo desperdiciado, pero es como funciona SIMT cuando hay divergencia.

#### 1.5.3.5 Final del kernel: escritura en v

```
Ciclo 23: TODOS ejecutan "v[t] = a"
─────────────────────────────────────────────────────────────
Thread 0:  v[0]  = 99.0
Thread 1:  v[1]  = 102.0
...
Thread 15: v[15] = 144.0
Thread 16: v[16] = 66.0    ← su a se quedó en 66 desde la 2ª iteración
Thread 17: v[17] = 68.0
...
Thread 31: v[31] = 96.0

⚡ Escritura COALESCENTE: los 32 escriben en v[0..31] contiguos → 1 transacción
```

### 1.5.4 Visualización completa del avance del warp

```
                  Iter 1       Iter 2       Iter 3
                  (ciclos 4-9) (10-15)      (16-22)
                  ─────────────────────────────────────
Thread 0:         ✓ activo     ✓ activo     ✓ activo    → v[0]  = 99
Thread 1:         ✓ activo     ✓ activo     ✓ activo    → v[1]  = 102
...
Thread 15:        ✓ activo     ✓ activo     ✓ activo    → v[15] = 144
─────────────────────────────────────────────────────────────────────
Thread 16:        ✓ activo     ✓ activo     ⚫ INACTIVO  → v[16] = 66
Thread 17:        ✓ activo     ✓ activo     ⚫ INACTIVO  → v[17] = 68
...
Thread 31:        ✓ activo     ✓ activo     ⚫ INACTIVO  → v[31] = 96

                  ↓ tiempo (ciclos de reloj) ↓
```

### 1.5.5 Tabla resumen de los 32 threads del warp

| Thread (t) | Iter 1: i, suma | Iter 2: i, suma | Iter 3: i, suma | a final = v[t] |
|---|---|---|---|---|
| 0 | i=0, +1 → 1 | i=32, +33 → 34 | i=64, +65 → 99 | **99** |
| 1 | i=1, +2 → 2 | i=33, +34 → 36 | i=65, +66 → 102 | **102** |
| 2 | i=2, +3 → 3 | i=34, +35 → 38 | i=66, +67 → 105 | **105** |
| ... | ... | ... | ... | ... |
| 15 | i=15, +16 → 16 | i=47, +48 → 64 | i=79, +80 → 144 | **144** |
| 16 | i=16, +17 → 17 | i=48, +49 → 66 | ⚫ (i=80 ≥ 80) | **66** |
| 17 | i=17, +18 → 18 | i=49, +50 → 68 | ⚫ | **68** |
| ... | ... | ... | ... | ... |
| 31 | i=31, +32 → 32 | i=63, +64 → 96 | ⚫ | **96** |

### 1.5.6 Estado de la memoria GPU tras el Kernel 1

```
d_x = [1.0, 2.0, 3.0, ..., 80.0]    ← sin cambios
d_y = [1.0, 1.0, 1.0, ..., 1.0]     ← sin cambios

d_v = [ 99, 102, 105, 108, 111, 114, 117, 120,    ← v[0..7]   (3 iteraciones)
       123, 126, 129, 132, 135, 138, 141, 144,    ← v[8..15]  (3 iteraciones)
        66,  68,  70,  72,  74,  76,  78,  80,    ← v[16..23] (2 iteraciones)
        82,  84,  86,  88,  90,  92,  94,  96]    ← v[24..31] (2 iteraciones)

d_result = [?]    ← sigue sin tocarse
```

**Verificación rápida**: sumando los 32 valores de `d_v`:

```
v[0..15]:  99+102+...+144 = 1944
v[16..31]: 66+68+...+96   = 1296
                    TOTAL = 3240  ✓ coincide con 80·81/2
```

> **Punto clave**: el Kernel 1 ha terminado. CUDA garantiza una **barrera implícita** entre kernels lanzados en el mismo stream, así que cuando llamemos al Kernel 2, sabemos con seguridad que `d_v` está completamente escrito.

---

## 1.6 Lanzar el Kernel 2

### 1.6.1 Sintaxis

```cuda
compute_kernel2<<<1, 1>>>(d_v, d_result);
```

- **Nº total de threads lanzados** = `1 × 1 = 1`. Ejecución estrictamente secuencial.
- ⚠️ **Esto es un warp con UN SOLO thread activo**. Los otros 31 "slots" del warp están inactivos (predicados off) durante todo el kernel. Es muy ineficiente, pero el enunciado lo pide así por simplicidad didáctica.

### 1.6.2 Código del kernel

```cuda
__global__ void compute_kernel2(float v[32], float *result) {
    float a = 0.0f;
    for (int i = 0; i < 32; i++) {
        a += v[i];
    }
    *result = a;
}
```

### 1.6.3 Traza del único thread

Suma secuencial de los 32 valores de `d_v`:

| i | v[i] | a antes | a después |
|---|---|---|---|
| 0 | 99 | 0 | **99** |
| 1 | 102 | 99 | **201** |
| 2 | 105 | 201 | **306** |
| ... | ... | ... | ... |
| 15 | 144 | 1800 | **1944** |
| 16 | 66 | 1944 | **2010** |
| ... | ... | ... | ... |
| 31 | 96 | 3144 | **3240** |

**Escribe:** `*result = 3240.0` → `d_result = [3240.0]`

### 1.6.4 Estado de la memoria GPU tras el Kernel 2

```
d_x      = [1.0, 2.0, ..., 80.0]    ← sin cambios
d_y      = [1.0, 1.0, ..., 1.0]     ← sin cambios
d_v      = [99, 102, ..., 96]       ← sin cambios
d_result = [3240.0]   ← ¡resultado final!
```

---

## 1.7 Copiar resultado GPU → CPU

```cuda
float r;
cudaMemcpy(&r, d_result, sizeof(float), cudaMemcpyDeviceToHost);
// r = 3240.0 en CPU
```

---

## 1.8 Liberar memoria

```cuda
cudaFree(d_result);
cudaFree(d_v);
cudaFree(d_x);
cudaFree(d_y);
```

---

## 1.9 Resumen del flujo completo

| Paso | Quién ejecuta | Qué hace | Estado clave después |
|---|---|---|---|
| 1 | Host (CPU) | Define `h_x`, `h_y`, `n=80` | Datos preparados en CPU |
| 2 | Host | `cudaMalloc` × 4 | Memoria GPU reservada (con basura) |
| 3 | Host | `cudaMemcpy` × 2 (H→D) | `d_x` y `d_y` con datos de entrada |
| 4 | 1 warp (32 threads) | `compute_kernel1<<<1,32>>>` | `d_v` con 32 parciales |
| 5 | 1 thread | `compute_kernel2<<<1,1>>>` | `d_result = [3240.0]` |
| 6 | Host | `cudaMemcpy` (D→H) | `r = 3240.0` en CPU |
| 7 | Host | `cudaFree` × 4 | Memoria GPU liberada |

---

## 1.10 Ideas clave sobre warps y ejecución

- **El warp es la unidad mínima de ejecución en GPU**: 32 threads que avanzan en lockstep, ejecutando la misma instrucción en el mismo ciclo (modelo SIMT).
- **Cuando lanzas `<<<1, 32>>>` lanzas exactamente 1 warp**. Por eso este ejercicio es el caso más simple posible: un único warp, sin coordinación entre warps, sin necesidad de `__syncthreads()`.
- **Los 32 threads del warp tienen estado privado** (registros propios para `t`, `a`, `i`), pero **ejecutan la misma instrucción a la vez** sobre datos distintos.
- **Coalescencia**: como los 32 threads del warp acceden a posiciones consecutivas (`d_x[0..31]`, luego `d_x[32..63]`, etc.), cada load se resuelve con **1 sola transacción de memoria** de 128 bytes. Es el caso ideal.
- **Divergencia del warp**: cuando algunos threads quieren seguir y otros no (3ª iteración del bucle en `n=80`), el hardware **ejecuta los dos caminos en serie**, marcando como inactivos a los threads que no aplican en cada pase. Los threads inactivos **no aceleran ni hacen trabajo extra**: simplemente se quedan a la espera de que el warp se reunifique.
- **Mismo número de iteraciones para todos = sin divergencia**. Si `n` fuera múltiplo de 32 (por ejemplo `n=96`), los 32 threads harían exactamente 3 iteraciones cada uno y no habría divergencia. La divergencia aparece solo en el "tramo de cola" cuando `n` no es múltiplo del nº de threads.
- **`<<<1, 1>>>` es un warp con 31 threads inactivos**: brutalmente ineficiente, pero didácticamente claro. En los Ejercicios 6-9 se irán eliminando estas ineficiencias.
- **Los kernels distintos están separados por una barrera implícita**: cuando termina el Kernel 1, todos sus warps han acabado de escribir; el Kernel 2 puede leer `d_v` con seguridad.