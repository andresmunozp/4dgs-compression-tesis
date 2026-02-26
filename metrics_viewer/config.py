"""Configuration and settings for the metrics viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ViewerConfig:
    """Application-level configuration.

    The ``project_root`` is auto-detected or passed via CLI.
    All other paths are resolved relative to it.
    """

    project_root: Path = Path(".")

    # Directories to scan for data (relative to project_root)
    scan_dirs: List[str] = field(default_factory=lambda: [
        "benchmark_results",
        "results_json",
        "compressed_output",
        "decompressed_output",
    ])

    # Dash server settings
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = True

    # Auto-refresh interval (seconds). 0 = disabled.
    auto_refresh_interval_s: int = 30

    # UI defaults
    default_theme: str = "darkly"   # dash-bootstrap-components theme name

    def resolve(self) -> "ViewerConfig":
        """Resolve project_root to an absolute path."""
        self.project_root = self.project_root.resolve()
        return self

    @classmethod
    def from_args(cls, args) -> "ViewerConfig":
        """Build config from argparse namespace."""
        cfg = cls(
            project_root=Path(args.data_dir) if hasattr(args, "data_dir") else Path("."),
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 8050),
            debug=getattr(args, "debug", True),
            auto_refresh_interval_s=getattr(args, "refresh_interval", 30),
        )
        return cfg.resolve()
