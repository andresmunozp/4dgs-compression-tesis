# Nuevas Funcionalidades: Análisis de Compresión y Descompresión

Se han agregado dos nuevas páginas al visualizador de métricas para analizar los datos generados por `compress.py` y `decompress.py`.

## 📊 Compression Analysis

**Ruta:** `/compression-analysis`  
**Icono en navegación:** 📦

Esta página muestra análisis comparativo de todas las técnicas de compresión disponibles en `compressed_output/`:

### Características:

1. **Tarjetas de Resumen:**
   - Total de técnicas analizadas
   - Ratio de compresión promedio
   - Porcentaje de ahorro promedio
   - Espacio total ahorrado en MB

2. **Gráficos Comparativos:**
   - **Compression Ratio & Savings**: Gráfico de barras con ratio de compresión y línea de porcentaje de ahorro en eje secundario
   - **File Size Comparison**: Barras agrupadas comparando tamaño original vs comprimido por técnica
   - **Compression Time**: Tiempo de compresión por técnica

3. **Tabla Detallada:**
   - Técnica
   - Escena
   - Archivo de configuración
   - Tamaño original (MB)
   - Tamaño comprimido (MB)
   - Ratio de compresión
   - Porcentaje de ahorro
   - Número de chunks

### Datos Mostrados:

Cada vez que ejecutes `compress.py` con diferentes técnicas (balanced, aggressive, lossless, etc.), se generará un archivo `compression_report.json` en la carpeta correspondiente de `compressed_output/`. El visualizador automáticamente detecta estos archivos y los incluye en el análisis.

**Ejemplo de estructura:**
```
compressed_output/
  ├── balanced/
  │   └── compression_report.json
  ├── aggressive/
  │   └── compression_report.json
  └── lossless/
      └── compression_report.json
```

---

## 🔄 Decompression Analysis

**Ruta:** `/decompression-analysis`  
**Icono en navegación:** 🔄

Esta página analiza el rendimiento de descompresión de las diferentes técnicas en `decompressed_output/`:

### Características:

1. **Tarjetas de Resumen:**
   - Total de técnicas analizadas
   - Tiempo promedio de decodificación
   - Tiempo total promedio
   - Total de Gaussians procesados

2. **Gráficos de Rendimiento:**
   - **Decompression Timing Breakdown**: Gráfico de barras apiladas mostrando:
     - Tiempo de ensamblaje (Assembly)
     - Tiempo de decodificación (Decode)
     - Tiempo de exportación (Export)
   - **Total Decompression Time**: Comparación del tiempo total por técnica
   - **Export Throughput (FPS)**: Frames por segundo durante la exportación

3. **Tabla Detallada:**
   - Técnica
   - Escena
   - Número de frames
   - Número de Gaussians
   - Tiempo de ensamblaje
   - Tiempo de decodificación
   - Tiempo de exportación
   - Tiempo total (destacado)

### Datos Mostrados:

Cada vez que ejecutes `decompress.py`, se genera un archivo `decompression_report.json` con las métricas de tiempo. El visualizador los descubre automáticamente.

**Ejemplo de estructura:**
```
decompressed_output/
  ├── balanced/
  │   └── decompression_report.json
  ├── aggressive/
  │   └── decompression_report.json
  └── lossless/
      └── decompression_report.json
```

---

## 🚀 Cómo Usar

1. **Generar datos de compresión:**
   ```bash
   python compress.py \
       --model_path output/dynerf/coffee_martini \
       --iteration 14000 \
       --config compression/configs/balanced.yaml \
       --output compressed_output/balanced/
   ```

2. **Generar datos de descompresión:**
   ```bash
   python decompress.py \
       --input compressed_output/balanced/ \
       --output decompressed_output/balanced/ \
       --num_frames 50
   ```

3. **Visualizar los resultados:**
   ```bash
   python -m metrics_viewer.app
   ```

4. **Navegar a las nuevas páginas:**
   - Click en "📦 Compression Analysis"
   - Click en "🔄 Decompression Analysis"

---

## ⚡ Auto-Refresh

Las páginas soportan auto-refresh (intervalo configurable con `--refresh-interval`). Si ejecutas compress.py o decompress.py mientras el visualizador está corriendo, los datos se actualizarán automáticamente.

---

## 🎨 Características Visuales

- **Gráficos interactivos** con Plotly (hover para detalles)
- **Dark theme** consistente con el resto del visualizador
- **Tablas responsivas** que se adaptan al tamaño de pantalla
- **Colores distintivos** para cada métrica
- **Formato numérico** optimizado para legibilidad

---

## 📈 Comparaciones Útiles

### Compression Analysis:
- Compara qué técnica da el mejor ratio de compresión
- Identifica qué configuración ahorra más espacio
- Analiza el trade-off entre ratio y tiempo de compresión

### Decompression Analysis:
- Identifica cuellos de botella en el pipeline de descompresión
- Compara throughput entre técnicas
- Optimiza configuraciones para minimizar latencia

---

## 🔧 Técnicas Soportadas

Todas las técnicas de compresión en tus carpetas:
- `balanced` - Balance entre calidad y compresión
- `aggressive` - Máxima compresión
- `lossless` - Sin pérdida de información
- `lightgaussian_balanced` - LightGaussian con balance
- `pruning` - Técnicas de poda
- `sh_reduction` - Reducción de coeficientes SH
- Y cualquier otra que agregues...

---

## 💡 Tips

1. **Compara múltiples configuraciones** ejecutando compress.py con diferentes archivos YAML
2. **Analiza el impacto de parámetros** como chunk_size, iteration, etc.
3. **Identifica la configuración óptima** para tu caso de uso específico
4. **Monitorea el rendimiento** antes y después de optimizaciones

---

¡Disfruta analizando tus datos de compresión y descompresión! 🎉
