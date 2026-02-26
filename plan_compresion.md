# Plan: Módulo Modular de Compresión para 4DGS Streaming

## TL;DR

Crear un sistema de **compresión post-entrenamiento** que:

- Toma un modelo **4DGS entrenado**
- Aplica estrategias configurables:
  - cuantización
  - pruning
  - reducción SH
  - compresión HexPlane
  - codificación entrópica
- Empaqueta en **chunks** para transmisión por **Mininet**
- En el receptor:
  - descomprime
  - reconstruye **PLY sequences** compatibles con **SuperSplat**

Un script de benchmarking evalúa:

- PSNR  
- SSIM  
- LPIPS  
- VMAF  
- Ratio de compresión  
- Tamaño final  

---

## Pipeline General

Modelo 4DGS entrenado
↓
compress.py (estrategia configurable)
↓
Chunks binarios (.4dgsc)
↓
Transmisión por Mininet
↓
Chunks recibidos
↓
decompress.py
↓
PLY sequences (time_XXXXX.ply)
↓
SuperSplat / render.py
↓
Visualización + métricas

---
# Steps
---
## 1. Crear estructura del módulo de compresión
Carpeta:
Estructura:

Crear carpeta compression/ con la siguiente estructura:

compression/__init__.py — exporta la API pública
compression/base.py — Clase abstracta CompressionStrategy con interfaz compress(model_data) → bytes y decompress(data) → model_data
compression/strategies/ — subcarpeta con cada estrategia como plugin
compression/pipeline.py — Orquestador CompressionPipeline que encadena múltiples estrategias
compression/chunker.py — Divide el modelo comprimido en chunks para transmisión
compression/serializer.py — Serializa/deserializa los datos del modelo a formato binario con header de metadata

---

## 2. Definir la clase base `CompressionStrategy`

En `compression/base.py`:

### Clase abstracta

Métodos:

- `compress(gaussian_data, deformation_data) → CompressedPayload`
- `decompress(payload) → (gaussian_data, deformation_data)`
- `name`
- `get_stats() → dict`

---

### Dataclasses

#### GaussianData

Contiene:

- xyz
- features_dc
- features_rest
- opacity
- scaling
- rotation
- sh_degree

---

#### DeformationData

Contiene:
- state_dict del deformation network
- deformation_table
- deformation_accum

---

#### CompressedPayload

- `data: bytes`
- `metadata: dict`
  - estrategia
  - shapes originales
  - dtype info

---

## 3. Estrategias de compresión

Ubicación:
compression/strategies/

### a) Quantization

Archivo: `quantization.py`

Parámetros:

- target_dtype: float16 / int8 / int16
- per_attribute

Funciones:

- Cuantización de gaussianos
- Min/max por atributo
- Cuantización HexPlane grids
- Cuantización MLP weights

---

### b) Pruning

Archivo: `pruning.py`

Criterios:

- opacity_threshold
- contribution_threshold
- redundancy_radius
- max_gaussians

Elimina gaussianos y ajusta tablas.

---

### c) SH Reduction

Archivo: `sh_reduction.py`

Parámetro:

- target_sh_degree

Ejemplos:

- degree 3 → 2
- degree 3 → 1
- degree 3 → 0

Impacto:

- Reduce hasta **76%** tamaño por gaussiano.

---

### d) HexPlane Compression

Archivo: `hexplane_compression.py`

Métodos:

- cuantización
- SVD truncado
- downsampling

Configuración:

- method
- niveles

---

### e) Entropy Coding

Archivo: `entropy_coding.py`

Algoritmos:

- zlib
- zstd
- lz4

Siempre última etapa.

---

## 4. Pipeline composable

En `compression/pipeline.py`

Clase:

### CompressionPipeline

Funciones:

- `compress()` → ejecuta estrategias en orden
- `decompress()` → ejecuta en orden inverso

Configuración vía YAML:

```yaml
strategies:
  - name: pruning
    params: {opacity_threshold: 0.005}

  - name: sh_reduction
    params: {target_sh_degree: 1}

  - name: quantization
    params: {xyz: float16}

  - name: entropy_coding
    params: {algorithm: zstd}

5. Sistema de chunking para Mininet

Archivo: chunker.py

Formato de chunk
[magic_bytes]
[chunk_id]
[total_chunks]
[chunk_type]
[payload_size]
[payload]
[checksum]


Tipos:

0x01 metadata

0x02 Gaussian data

0x03 deformation network

0x04 deformation table

Soporta tamaños configurables:

1 MB

5 MB

10 MB

6. Script compress.py

Entradas:

model_path

iteration

config

output

chunk_size

Proceso:

Carga modelo

Extrae GaussianData

Ejecuta pipeline

Divide en chunks

Guarda resultados

Imprime estadísticas

7. Script decompress.py

Entradas:

input chunks

output

format

configs

Proceso:

Ensambla chunks

Ejecuta decompress pipeline

Reconstruye GaussianModel

Ejecuta deformation network

Exporta PLY per-frame

Output:

time_XXXXX.ply

.compressed.ply

8. Benchmarking

Script:

benchmark_compression.py


Evalúa:

PSNR

SSIM

LPIPS

VMAF

ratio

tiempos

Outputs:

JSON comparativo

CSV resumen

Plots

9. Configs de ejemplo

Ubicación:

compression/configs/


Archivos:

aggressive.yaml

balanced.yaml

lossless.yaml

quantize_only.yaml

streaming_optimized.yaml

10. Estructura final
compression/
├── base.py
├── pipeline.py
├── chunker.py
├── serializer.py
├── strategies/
└── configs/

compress.py
decompress.py
benchmark_compression.py

Verification
Unitaria

Validar compress → decompress

Integración

Pipeline completo hasta SuperSplat

Calidad

Métricas PSNR / SSIM / LPIPS / VMAF

Red

Transmisión real por Mininet

Decisions
Representación PLY per-frame

SuperSplat solo soporta PLY sequences.

Compresión sobre modelo canónico

Más eficiente:

Modelo: 32–55 MB

PLYs: 7–15 GB

GPU requerida para decodificación
Necesaria para ejecutar deformation network.
Formato .compressed.ply
Soportado nativamente por SuperSplat.
Estrategias composables

Orden recomendado:

pruning → SH reduction → quantization → entropy coding