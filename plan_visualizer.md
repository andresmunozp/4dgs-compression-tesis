# Plan: Metrics Visualizer — 4DGaussians

## 1. Objetivo

Crear un dashboard web interactivo para visualizar, comparar y analizar todas las métricas generadas por el pipeline de 4DGaussians: benchmarks de compresión, evaluaciones de entrenamiento, calidad de video y streaming QoE. Debe ser fácil de extender con nuevos tipos de resultados y nuevas fuentes de datos.

---

## 2. Arquitectura: Hexagonal (Ports & Adapters)

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Dash/Plotly)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Overview  │  │ Compare  │  │PerFrame  │  │ Streaming QoE    │ │
│  │ Dashboard │  │ Strategies│  │ Timeline │  │ Dashboard        │ │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └────────┬─────────┘ │
│        │              │             │                │           │
│        └──────────────┴─────────────┴────────────────┘           │
│                              │                                   │
│                    ┌─────────▼─────────┐                         │
│                    │   VIEW SERVICE    │    ← Driving Port       │
│                    │  (Orquesta vistas)│                         │
│                    └─────────┬─────────┘                         │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                         DOMAIN CORE                              │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                  MetricsService                            │  │
│  │  - load_results(source) → List[MetricRecord]               │  │
│  │  - compare(records, metrics) → ComparisonResult            │  │
│  │  - aggregate(records, group_by) → AggregatedResult         │  │
│  │  - get_per_frame(record, metric) → TimeSeriesData          │  │
│  └──────┬────────────────────┬───────────────────────┬────────┘  │
│         │                    │                       │            │
│  ┌──────▼──────┐  ┌─────────▼────────┐  ┌───────────▼────────┐  │
│  │MetricRecord │  │ComparisonResult  │  │ TimeSeriesData     │  │
│  │ (Entity)    │  │ (Value Object)   │  │ (Value Object)     │  │
│  └─────────────┘  └──────────────────┘  └────────────────────┘  │
│                                                                  │
│  ── Driven Ports (Interfaces/Protocols) ──────────────────────   │
│  ┌─────────────────────┐  ┌────────────────────────────────┐     │
│  │ IDataSourceReader   │  │ IResultExporter                │     │
│  │  + read() → list    │  │  + export(data, format) → bytes│     │
│  │  + supports(path)   │  │  + supported_formats()         │     │
│  └──────────┬──────────┘  └────────────┬───────────────────┘     │
└─────────────┼──────────────────────────┼────────────────────────┘
              │                          │
┌─────────────┼──────────────────────────┼────────────────────────┐
│          ADAPTERS (Driven / Infrastructure)                      │
│             │                          │                         │
│  ┌──────────▼──────────┐  ┌────────────▼───────────────────┐     │
│  │ JSON Adapter        │  │  CSV Exporter                  │     │
│  │ (BenchmarkJSON,     │  │  PNG Exporter                  │     │
│  │  ResultsJSON,       │  │  HTML Exporter                 │     │
│  │  VmafJSON)          │  └────────────────────────────────┘     │
│  ├─────────────────────┤                                         │
│  │ CSV Adapter         │                                         │
│  │ (BenchmarkCSV)      │                                         │
│  ├─────────────────────┤                                         │
│  │ Directory Scanner   │                                         │
│  │ (auto-discover)     │                                         │
│  └─────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────┘
```

### Principios clave:
- **Domain Core** no depende de ninguna tecnología de I/O ni de visualización.
- **Ports** son interfaces abstractas (`Protocol` de Python) que definen contratos.
- **Adapters** implementan esos contratos para JSON, CSV, VMAF, etc.
- **Agregar un nuevo formato** = crear un nuevo Adapter que implemente `IDataSourceReader`.
- **Cambiar de Dash a Streamlit o cualquier otro framework** = solo cambiar el frontend, el dominio no cambia.

---

## 3. Estructura de Carpetas

```
viewers/
├── __init__.py
├── app.py                          # Entry point: python -m viewers.app
├── config.py                       # Settings, paths, defaults
│
├── domain/                         # DOMAIN CORE (sin dependencias externas)
│   ├── __init__.py
│   ├── models.py                   # Entities y Value Objects
│   ├── ports.py                    # Interfaces (Protocol classes)
│   ├── services.py                 # MetricsService (lógica de negocio)
│   └── enums.py                    # MetricType, ResultCategory, etc.
│
├── adapters/                       # ADAPTERS (implementaciones de ports)
│   ├── __init__.py
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── benchmark_json_reader.py    # Lee benchmark_results.json
│   │   ├── benchmark_csv_reader.py     # Lee benchmark_summary.csv
│   │   ├── training_json_reader.py     # Lee results_json/*.json
│   │   ├── vmaf_json_reader.py         # Lee vmaf.json / vmaf_vs_gt.json
│   │   └── directory_scanner.py        # Auto-descubre archivos de resultados
│   │
│   └── exporters/
│       ├── __init__.py
│       ├── csv_exporter.py
│       ├── png_exporter.py
│       └── html_exporter.py
│
├── ui/                             # FRONTEND (Dash/Plotly)
│   ├── __init__.py
│   ├── layout.py                   # Layout principal con tabs/sidebar
│   ├── callbacks.py                # Callbacks de Dash (interactividad)
│   ├── components/                 # Componentes reutilizables
│   │   ├── __init__.py
│   │   ├── metric_card.py          # Card con valor de métrica
│   │   ├── comparison_table.py     # Tabla comparativa
│   │   ├── per_frame_chart.py      # Gráfico temporal por frame
│   │   ├── bar_chart.py            # Barras para comparar estrategias
│   │   ├── radar_chart.py          # Spider/radar multimétrica
│   │   ├── scatter_plot.py         # Scatter: ratio vs calidad
│   │   ├── heatmap.py              # Heatmap de métricas
│   │   └── filters.py             # Dropdowns, checkboxes de filtros
│   │
│   └── pages/                      # Páginas/Tabs del dashboard
│       ├── __init__.py
│       ├── overview.py             # Resumen general
│       ├── benchmark_compare.py    # Comparar estrategias de compresión
│       ├── training_eval.py        # Evaluar resultados de entrenamiento
│       ├── per_frame_analysis.py   # Análisis frame-by-frame
│       ├── streaming_qoe.py        # Dashboard de QoE de streaming
│       └── compression_detail.py   # Detalle de una compresión específica
│
├── assets/                         # CSS, JS estáticos para Dash
│   └── styles.css
│
└── tests/                          # Unit tests
    ├── __init__.py
    ├── test_models.py
    ├── test_services.py
    ├── test_readers.py
    └── fixtures/                   # Datos de prueba
        ├── sample_benchmark.json
        ├── sample_summary.csv
        └── sample_training.json
```

---

## 4. Modelos de Dominio (Entities & Value Objects)

### 4.1. `MetricRecord` — Entidad principal

```python
@dataclass
class MetricRecord:
    """Un resultado completo de una evaluación/benchmark."""
    id: str                                    # Identificador único
    source: ResultSource                       # Enum: BENCHMARK, TRAINING, VMAF
    category: ResultCategory                   # Enum: COMPRESSION, DECOMPRESSION, TRAINING
    name: str                                  # e.g. "hexplane_downsample"
    scene: str                                 # e.g. "coffee_martini"
    timestamp: Optional[datetime]              # Cuándo se generó
    tags: Dict[str, str]                       # Metadata libre: {"iteration": "4000", "version": "antes"}

    # Métricas de calidad de imagen (medias)
    quality_metrics: QualityMetrics

    # Métricas por frame (opcionales)
    per_frame_metrics: Optional[PerFrameMetrics]

    # Métricas de compresión (opcionales — solo para benchmarks)
    compression_metrics: Optional[CompressionMetrics]

    # Métricas de streaming QoE (opcionales)
    streaming_metrics: Optional[StreamingMetrics]

    # Métricas de timing
    timing_metrics: Optional[TimingMetrics]

    # Info del modelo
    model_info: Optional[ModelInfo]

    # Pipeline stats (para benchmarks con múltiples etapas)
    pipeline_stats: Optional[List[PipelineStageStats]]
```

### 4.2. Value Objects

```python
@dataclass(frozen=True)
class QualityMetrics:
    """Métricas de calidad de imagen (medias)."""
    psnr: Optional[float] = None        # dB
    ssim: Optional[float] = None
    lpips_vgg: Optional[float] = None
    lpips_alex: Optional[float] = None
    ms_ssim: Optional[float] = None
    d_ssim: Optional[float] = None
    vmaf: Optional[float] = None

@dataclass(frozen=True)
class PerFrameMetrics:
    """Series temporales de métricas por frame."""
    frame_indices: List[int]
    psnr: Optional[List[float]] = None
    ssim: Optional[List[float]] = None
    lpips: Optional[List[float]] = None
    vmaf: Optional[List[float]] = None
    # Extensible: pueden agregarse más listas

@dataclass(frozen=True)
class CompressionMetrics:
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    savings_pct: float
    num_chunks: int

@dataclass(frozen=True)
class StreamingMetrics:
    total_payload_bytes: int
    bandwidth_mbps: float
    startup_delay_s: float
    rebuffer_events: int
    total_stall_duration_s: float
    e2e_latency_s: float
    effective_throughput_mbps: float
    qoe_score: float
    target_fps: float

@dataclass(frozen=True)
class TimingMetrics:
    compress_time_s: Optional[float] = None
    decompress_time_s: Optional[float] = None
    train_time_s: Optional[float] = None
    render_time_s: Optional[float] = None
    export_time_per_frame_s: Optional[float] = None

@dataclass(frozen=True)
class ModelInfo:
    num_gaussians_original: Optional[int] = None
    num_gaussians_compressed: Optional[int] = None
    sh_degree_original: Optional[int] = None
    sh_degree_compressed: Optional[int] = None
    iteration: Optional[int] = None

@dataclass(frozen=True)
class PipelineStageStats:
    strategy: str
    ratio: float
    savings_pct: float
    compress_time_s: float
    decompress_time_s: float
    extra: Dict[str, Any] = field(default_factory=dict)
```

### 4.3. Enums

```python
class ResultSource(Enum):
    BENCHMARK_JSON = "benchmark_json"
    BENCHMARK_CSV = "benchmark_csv"
    TRAINING_JSON = "training_json"
    VMAF_JSON = "vmaf_json"

class ResultCategory(Enum):
    COMPRESSION = "compression"
    DECOMPRESSION = "decompression"
    TRAINING = "training"
    END_TO_END = "end_to_end"

class MetricType(Enum):
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS_VGG = "lpips_vgg"
    LPIPS_ALEX = "lpips_alex"
    MS_SSIM = "ms_ssim"
    D_SSIM = "d_ssim"
    VMAF = "vmaf"
    COMPRESSION_RATIO = "compression_ratio"
    SAVINGS_PCT = "savings_pct"
    QOE_SCORE = "qoe_score"

class ComparisonAxis(Enum):
    """Los tres ejes de comparación del benchmark."""
    COMPRESSION_FIDELITY = "compression_fidelity"     # decompressed vs original model
    END_TO_END = "end_to_end"                         # decompressed vs ground truth
    TRAINING_BASELINE = "training_baseline"           # original vs ground truth
```

---

## 5. Ports (Interfaces)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class IDataSourceReader(Protocol):
    """Puerto driven: lee datos de una fuente y devuelve MetricRecords."""

    def supports(self, path: Path) -> bool:
        """¿Este reader puede leer el archivo/directorio dado?"""
        ...

    def read(self, path: Path) -> List[MetricRecord]:
        """Lee y parsea la fuente, retorna lista de MetricRecords."""
        ...

    @property
    def source_type(self) -> ResultSource:
        """Tipo de fuente que maneja este reader."""
        ...


@runtime_checkable
class IResultExporter(Protocol):
    """Puerto driven: exporta datos en distintos formatos."""

    def export(self, records: List[MetricRecord], config: ExportConfig) -> bytes:
        ...

    @property
    def supported_formats(self) -> List[str]:
        ...


@runtime_checkable
class IMetricsService(Protocol):
    """Puerto driving: API que el frontend consume."""

    def get_all_records(self) -> List[MetricRecord]:
        ...

    def get_records_by_category(self, category: ResultCategory) -> List[MetricRecord]:
        ...

    def compare(self, record_ids: List[str], metrics: List[MetricType]) -> ComparisonResult:
        ...

    def get_per_frame_data(self, record_id: str, metric: MetricType) -> PerFrameMetrics:
        ...

    def get_available_filters(self) -> FilterOptions:
        ...
```

---

## 6. Servicio de Dominio

```python
class MetricsService:
    """Implementación del servicio central. No depende de I/O directo."""

    def __init__(self, readers: List[IDataSourceReader]):
        self._readers = readers
        self._records: Dict[str, MetricRecord] = {}
        self._registry = ReaderRegistry(readers)

    def load_from_path(self, path: Path) -> int:
        """Carga datos usando el reader apropiado. Retorna # de records agregados."""
        reader = self._registry.get_reader_for(path)
        new_records = reader.read(path)
        for r in new_records:
            self._records[r.id] = r
        return len(new_records)

    def auto_discover(self, base_dir: Path) -> int:
        """Escanea directorio base y carga todos los resultados encontrados."""
        ...

    def compare(self, record_ids, metrics) -> ComparisonResult:
        """Genera tabla comparativa de las métricas seleccionadas."""
        ...

    def get_per_frame_data(self, record_id, metric) -> TimeSeriesData:
        """Extrae serie temporal de un métrica específica."""
        ...

    def aggregate_by(self, group_by: str) -> Dict[str, AggregatedResult]:
        """Agrupa records por scene, strategy, category, etc."""
        ...
```

---

## 7. Adapters — Readers

### 7.1 `BenchmarkJsonReader`

Lee `benchmark_results/benchmark_results.json` y variantes `_antes`.

- Cada entrada del array JSON → 1-3 `MetricRecord`:
  - Uno con `ComparisonAxis.COMPRESSION_FIDELITY`
  - Uno con `ComparisonAxis.END_TO_END`
  - Uno con `ComparisonAxis.TRAINING_BASELINE`
- O alternativamente, un solo `MetricRecord` con los tres conjuntos de QualityMetrics como campos separados.
- Populará `compression_metrics`, `streaming_metrics`, `timing_metrics`, `model_info`, `pipeline_stats`.
- Incluye `per_frame_metrics` (psnr_per_frame, etc.).
- Tag `version`: detecta si el archivo tiene sufijo `_antes`.

### 7.2 `BenchmarkCsvReader`

Lee `benchmark_results/benchmark_summary.csv` (y `_antes`).

- Solo métricas medias (sin per-frame).
- Útil como fallback o vista rápida.
- Detecta `_antes` para tagging.

### 7.3 `TrainingJsonReader`

Lee `results_json/*.json`.

- Extrae métricas de `metrics_full.ours_<iter>`.
- Populará `timing_metrics` (train, render, metrics, ply export).
- Populará `model_info` (iteration, num_ply_files, total_size).
- VMAF incluido.

### 7.4 `VmafJsonReader`

Lee `benchmark_results/<strategy>/vmaf.json` y `vmaf_vs_gt.json`.

- Extrae per-frame VMAF + sub-métricas (ADM, VIF, motion).
- Genera `PerFrameMetrics` detallado.

### 7.5 `DirectoryScanner`

No es un reader per se, sino un utility que:
1. Escanea `benchmark_results/`, `results_json/`, `compressed_output/`, `decompressed_output/`
2. Devuelve lista de `(path, suggested_reader_type)` para auto-discovery.

---

## 8. Frontend — Páginas del Dashboard

### Tecnología: **Dash (Plotly)** con `dash-bootstrap-components`

Justificación:
- Python nativo (no necesita JS separado)
- Gráficos interactivos con zoom, hover, export
- Bootstrap para layout responsive
- Callbacks para interactividad
- Fácil de deployar

### 8.1 **Overview** (`/`)
- Cards con totales: # estrategias evaluadas, # escenas, mejor PSNR, mejor ratio
- Tabla resumen con todas las estrategias y sus métricas principales
- Código de colores: verde (mejor), rojo (peor) por columna
- Filtro por: escena, categoría, versión

### 8.2 **Benchmark Compare** (`/benchmark`)
- Selección múltiple de estrategias a comparar
- **Bar chart agrupado**: PSNR, SSIM, LPIPS lado a lado por estrategia
- **Scatter plot**: compression_ratio (x) vs PSNR (y) — **Pareto frontier**
- **Radar chart**: Normalizar 5-6 métricas y plotear por estrategia
- **Tabla detallada** con deltas (Δ respecto al baseline/referencia)
- Toggle entre los 3 ejes de comparación (fidelity / e2e / baseline)

### 8.3 **Training Evaluation** (`/training`)
- Comparar diferentes iteraciones de entrenamiento
- Line chart: métricas vs iteraciones
- Bar chart: tiempos (train, render, metrics, export)
- Tabla con métricas completas (SSIM, PSNR, LPIPS-vgg, LPIPS-alex, MS-SSIM, D-SSIM, VMAF)
- Size info: total PLY size, num files

### 8.4 **Per-Frame Analysis** (`/per-frame`)
- Selección de estrategia(s) y métrica
- **Line chart**: métrica vs frame_number con una línea por estrategia
- **Area chart**: banda de confianza (min/max) si hay múltiples runs
- Hover: ver valor exacto + frame thumbnail (si existe PNG)
- **Heatmap**: frames × strategies, coloreado por PSNR/SSIM/VMAF
- Detección de frames "problemáticos" (outliers, drops)

### 8.5 **Streaming QoE** (`/streaming`)
- Métricas de calidad de experiencia de streaming
- Gauge charts: QoE score (1-5), throughput
- Timeline: buffer level, rebuffer events
- Comparación bandwidth vs quality tradeoff
- Tabla: startup delay, stall time, latency por estrategia

### 8.6 **Compression Detail** (`/compression/<strategy_name>`)
- Detalle drill-down de una estrategia específica
- Pipeline stages breakdown (si hay múltiples etapas)
- Distribución de tamaños por chunk
- Si hay video: embed MP4 player comparativo (reference vs compressed)

---

## 9. Componentes UI Reutilizables

| Componente | Descripción | Usado en |
|---|---|---|
| `MetricCard` | Card con icono, valor, delta vs referencia, sparkline | Overview, Detail |
| `ComparisonTable` | Tabla con highlights de mejor/peor por columna | Overview, Benchmark |
| `PerFrameChart` | Line chart interactivo Plotly multi-serie con zoom | Per-Frame, Detail |
| `BarChart` | Barras agrupadas/stacked con colores por estrategia | Benchmark, Training |
| `RadarChart` | Spider chart multi-métrica normalizada | Benchmark |
| `ScatterPlot` | Scatter con Pareto frontier overlay | Benchmark |
| `Heatmap` | Heatmap frames × strategies | Per-Frame |
| `FilterPanel` | Dropdowns + checkboxes + date range | Todas las páginas |
| `ExportButton` | Descarga datos como CSV/PNG/HTML | Todas las páginas |

---

## 10. Flujo de Datos (Secuencia)

```
1. app.py arranca
2. config.py lee rutas base (benchmark_results/, results_json/, etc.)
3. DirectoryScanner descubre archivos
4. MetricsService.auto_discover() carga todo via readers apropiados
5. Se construyen MetricRecords en memoria (Dict[id, MetricRecord])
6. Dash layout se construye con datos iniciales
7. Usuario interactúa → callbacks → MetricsService queries → Plotly figures update
```

---

## 11. Plan de Implementación por Fases

### Fase 1: Core Domain + Readers (Día 1)
| # | Tarea | Archivos |
|---|-------|----------|
| 1.1 | Crear modelos de dominio | `domain/models.py`, `domain/enums.py` |
| 1.2 | Definir interfaces/ports | `domain/ports.py` |
| 1.3 | Implementar `BenchmarkJsonReader` | `adapters/readers/benchmark_json_reader.py` |
| 1.4 | Implementar `BenchmarkCsvReader` | `adapters/readers/benchmark_csv_reader.py` |
| 1.5 | Implementar `TrainingJsonReader` | `adapters/readers/training_json_reader.py` |
| 1.6 | Implementar `VmafJsonReader` | `adapters/readers/vmaf_json_reader.py` |
| 1.7 | Implementar `DirectoryScanner` | `adapters/readers/directory_scanner.py` |
| 1.8 | Implementar `MetricsService` | `domain/services.py` |
| 1.9 | Tests unitarios para readers y service | `tests/` |

### Fase 2: UI Base + Overview (Día 2)
| # | Tarea | Archivos |
|---|-------|----------|
| 2.1 | Setup Dash app + layout base con tabs | `app.py`, `ui/layout.py` |
| 2.2 | Componentes: `MetricCard`, `ComparisonTable` | `ui/components/` |
| 2.3 | Página Overview | `ui/pages/overview.py` |
| 2.4 | `FilterPanel` con dropdowns | `ui/components/filters.py` |
| 2.5 | Callbacks iniciales | `ui/callbacks.py` |
| 2.6 | CSS base (dark theme) | `assets/styles.css` |

### Fase 3: Comparación de Benchmarks (Día 3)
| # | Tarea | Archivos |
|---|-------|----------|
| 3.1 | `BarChart` component | `ui/components/bar_chart.py` |
| 3.2 | `RadarChart` component | `ui/components/radar_chart.py` |
| 3.3 | `ScatterPlot` con Pareto | `ui/components/scatter_plot.py` |
| 3.4 | Página Benchmark Compare | `ui/pages/benchmark_compare.py` |
| 3.5 | Callbacks de comparación | `ui/callbacks.py` (ampliar) |

### Fase 4: Per-Frame + Training (Día 4)
| # | Tarea | Archivos |
|---|-------|----------|
| 4.1 | `PerFrameChart` component | `ui/components/per_frame_chart.py` |
| 4.2 | `Heatmap` component | `ui/components/heatmap.py` |
| 4.3 | Página Per-Frame Analysis | `ui/pages/per_frame_analysis.py` |
| 4.4 | Página Training Evaluation | `ui/pages/training_eval.py` |

### Fase 5: Streaming QoE + Detail + Export (Día 5)
| # | Tarea | Archivos |
|---|-------|----------|
| 5.1 | Página Streaming QoE | `ui/pages/streaming_qoe.py` |
| 5.2 | Página Compression Detail | `ui/pages/compression_detail.py` |
| 5.3 | Exporters (CSV, PNG) | `adapters/exporters/` |
| 5.4 | `ExportButton` component | `ui/components/` |
| 5.5 | Tests de integración | `tests/` |

### Fase 6: Polish + Deploy (Día 6)
| # | Tarea | Archivos |
|---|-------|----------|
| 6.1 | Responsive design, tooltips, loading states | UI files |
| 6.2 | Auto-refresh cuando cambian archivos | `app.py` |
| 6.3 | CLI arguments (puerto, directorio base) | `app.py`, `config.py` |
| 6.4 | README de uso del viewer | `viewers/README.md` |
| 6.5 | Script de arranque rápido | `run_viewer.py` |

---

## 12. Dependencias

```
# Agregar a requirements.txt o crear viewers/requirements.txt
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.18.0
pandas>=2.0.0
# Ya existentes en el proyecto:
# numpy, torch (no hacer dependencia de torch en el viewer)
```

---

## 13. Cómo Ejecutar

```bash
# Desde la raíz del proyecto
python -m viewers.app

# Con opciones
python -m viewers.app --port 8050 --data-dir ./benchmark_results --results-dir ./results_json
```

Abre `http://localhost:8050` en el navegador.

---

## 14. Extensibilidad — Cómo Agregar Nuevas Cosas

### Nuevo formato de datos:
1. Crear un nuevo reader en `adapters/readers/` que implemente `IDataSourceReader`
2. Registrarlo en `DirectoryScanner` y en la config
3. El dominio y la UI no cambian

### Nueva métrica:
1. Agregar campo a `QualityMetrics` o crear nuevo value object
2. Agregar a `MetricType` enum
3. Los componentes UI que usan `MetricType` la mostrarán automáticamente

### Nueva página/vista:
1. Crear archivo en `ui/pages/`
2. Registrar tab en `ui/layout.py`
3. Agregar callbacks en `ui/callbacks.py`
4. Reutilizar componentes existentes de `ui/components/`

### Nueva estrategia de compresión:
No se necesita código nuevo — el `DirectoryScanner` y `BenchmarkJsonReader` la descubrirán automáticamente si sigue el formato existente.

---

## 15. Decisiones de Diseño

| Decisión | Justificación |
|----------|---------------|
| **Dash sobre Streamlit** | Más control sobre layout, mejor para dashboards complejos multi-página, callbacks más flexibles |
| **Hexagonal sobre MVC** | Desacopla completamente I/O del dominio, facilita testing y cambio de tecnología |
| **Dataclasses sobre Pydantic** | Menos dependencias, suficiente para este caso, solo serialización simple |
| **Protocol sobre ABC** | Más pythonico, structural typing, no requiere herencia explícita |
| **Readers como lista inyectada** | Permite agregar/quitar readers sin tocar el servicio |
| **Auto-discovery** | No hardcodear rutas, escalar a nuevas escenas/estrategias sin config manual |
| **Per-frame como lista opcional** | No todos los sources tienen datos per-frame, no forzar |

---

## 16. Visualizaciones Clave por Tipo de Comparación

### Compresión vs Compresión:
```
┌───────────────────────────┐  ┌────────────────────────────┐
│ Bar Chart: PSNR by        │  │ Scatter: Ratio vs PSNR     │
│ Strategy                  │  │ (con Pareto frontier)      │
│ ████ hex_ds  28.3 dB      │  │     *hex_svd               │
│ █████ hex_svd 29.1 dB     │  │   *hex_ds                  │
│ ██ aggressive 25.2 dB     │  │         *balanced           │
└───────────────────────────┘  └────────────────────────────┘
┌───────────────────────────┐  ┌────────────────────────────┐
│ Radar Chart               │  │ Per-Frame PSNR             │
│    PSNR                   │  │ ─── hex_ds                 │
│   /    \                  │  │ ─── hex_svd                │
│  VMAF    SSIM             │  │ ─── aggressive             │
│   \    /                  │  │  ↓ frame 23: drop          │
│    Ratio                  │  └────────────────────────────┘
└───────────────────────────┘
```

### Training (iteraciones):
```
┌──────────────────────────────────────────┐
│  Line: Metrics vs Iteration              │
│  PSNR ↑  ──────────────*                 │
│          ────────*                        │
│  it2000         it4000       it8000       │
└──────────────────────────────────────────┘
```

### Streaming QoE:
```
┌──────────┐ ┌──────────┐ ┌──────────┐
│ QoE: 3.2 │ │ Startup  │ │ Stalls   │
│ ██████░░ │ │  0.31s   │ │   50     │
│  /5.0    │ │          │ │          │
└──────────┘ └──────────┘ └──────────┘
```

---

## Siguiente Paso

Confirmar el plan y proceder con **Fase 1**: implementar domain core + readers.
