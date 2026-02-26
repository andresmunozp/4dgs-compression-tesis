"""Port interfaces (driven & driving) for the metrics viewer.

All ports use ``typing.Protocol`` for structural subtyping — adapters
do *not* need to inherit from these classes, they just need to match
the method signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .enums import ComparisonAxis, MetricType, ResultCategory, ResultSource
from .models import ComparisonResult, FilterOptions, MetricRecord, PerFrameMetrics


# ── Driven ports (infrastructure side) ─────────────────────────


@runtime_checkable
class IDataSourceReader(Protocol):
    """Reads metric data from a specific file/directory format."""

    def supports(self, path: Path) -> bool:
        """Return True if this reader can handle the given path."""
        ...

    def read(self, path: Path) -> List[MetricRecord]:
        """Parse the source and return a list of MetricRecords."""
        ...

    @property
    def source_type(self) -> ResultSource:
        """The type of source this reader handles."""
        ...


@dataclass
class ExportConfig:
    """Configuration passed to exporters."""
    format: str = "csv"                 # csv | png | html
    title: str = ""
    width: int = 1200
    height: int = 800
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IResultExporter(Protocol):
    """Exports metric data to a file format."""

    def export(self, records: List[MetricRecord], config: ExportConfig) -> bytes:
        """Serialize records into the specified format, return raw bytes."""
        ...

    @property
    def supported_formats(self) -> List[str]:
        """List of format strings this exporter can produce."""
        ...


# ── Driving port (application / UI side) ───────────────────────


@runtime_checkable
class IMetricsService(Protocol):
    """API consumed by the UI layer to query metric data."""

    def get_all_records(self) -> List[MetricRecord]:
        ...

    def get_record(self, record_id: str) -> Optional[MetricRecord]:
        ...

    def get_records_by_category(self, category: ResultCategory) -> List[MetricRecord]:
        ...

    def get_records_by_source(self, source: ResultSource) -> List[MetricRecord]:
        ...

    def search_records(
        self,
        *,
        category: Optional[ResultCategory] = None,
        source: Optional[ResultSource] = None,
        scene: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[MetricRecord]:
        ...

    def compare(
        self,
        record_ids: List[str],
        metrics: List[MetricType],
        axis: Optional[ComparisonAxis] = None,
    ) -> ComparisonResult:
        ...

    def get_per_frame_data(
        self,
        record_id: str,
        axis: Optional[ComparisonAxis] = None,
    ) -> Optional[PerFrameMetrics]:
        ...

    def get_available_filters(self) -> FilterOptions:
        ...

    def load_from_path(self, path: Path) -> int:
        """Load data from a file/dir; return number of records added."""
        ...

    def auto_discover(self, base_dir: Path) -> int:
        """Scan base_dir for known data files; return total records loaded."""
        ...
