# 1 Fundamentos: GPU y CUDA
- 1.1 Programación heterogénea (host + device, DRAMs separadas)
- 1.2 Arquitectura GPU
    - 1.2.1 CPU vs GPU: transistores para cómputo vs control
    - 1.2.2 SMs, CUDA cores, warps (modelo SIMT)
    - 1.2.3 Jerarquía de memoria (registers, shared, global, constant)
- 1.3 Qué es CUDA
    - 1.3.1 Plataforma + modelo de programación + ISA
    - 1.3.2 Ecosistema: librerías y wrappers
    - 1.3.3 CUDA Toolkit
- 1.4 Compute Capability y arquitecturas
- 1.5 Cuándo conviene GPGPU

# 2 Modelo de programación de CUDA
- 2.1 Idea general: abstracción sobre el hardware, escalabilidad
- 2.2 Kernels
    - 2.2.1 Qué es un kernel
    - 2.2.2 Sintaxis: __global__ y <<<grid, block>>>
- 2.3 Jerarquía de threads
    - 2.3.1 Thread → Block → Grid (1D/2D/3D)
    - 2.3.2 Variables built-in
    - 2.3.3 Cálculo de índice global
    - 2.3.4 Independencia de bloques → escalabilidad
    - 2.3.5 Sincronización (__syncthreads, barreras entre kernels)
- 2.4 Warps: del modelo lógico al hardware
    - 2.4.1 Por qué importa el warp al programar
    - 2.4.2 Elección del tamaño de bloque (múltiplos de 32)
    - 2.4.3 Divergencia de warp (branches y rendimiento)
    - 2.4.4 Sincronización implícita dentro del warp
- 2.5 Ejemplos de cálculo de dimensiones

# 3 Cheatsheets
- 3.1 Compilador (nvcc)
- 3.2 Especificadores y variables built-in
- 3.3 Memoria del device (cudaMalloc, cudaMemcpy, ...)
- 3.4 Sincronización
- 3.5 Manejo de errores
- 3.6 Flujo típico host ↔ device
- 3.7 Warps: reglas prácticas (tamaño bloque, divergencia, occupancy)

# 4 Ejemplos
- 4.1 Suma de matrices C = A + B
- 4.2 Suma de vectores z = x + y (monolithic vs grid-stride)
- 4.3 Producto matriz-vector c = A·v
- 4.4 Patrón de cálculo de dimensiones (ceil)