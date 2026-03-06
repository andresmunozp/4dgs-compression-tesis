"""Application entry point — wires together domain, adapters, and UI.

Usage::

    python -m viewers.metrics_viewer.app --data-dir . --port 8050
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .adapters.readers import (
    BenchmarkCsvReader,
    BenchmarkJsonReader,
    CompressionReportReader,
    DecompressionReportReader,
    TrainingJsonReader,
    VmafJsonReader,
)
from .config import ViewerConfig
from .domain.services import MetricsService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def build_service(config: ViewerConfig) -> MetricsService:
    """Construct the MetricsService with all readers wired in."""
    readers = [
        BenchmarkJsonReader(),
        BenchmarkCsvReader(),
        TrainingJsonReader(),
        VmafJsonReader(),        CompressionReportReader(),
        DecompressionReportReader(),    ]
    service = MetricsService(readers=readers)
    total = service.auto_discover(config.project_root)
    logger.info("Service ready \u2014 %d records loaded", total)
    return service


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4DGaussians Metrics Viewer \u2014 interactive dashboard for benchmarks, training, and compression metrics.",
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Project root directory (default: current dir)",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050,
                        help="Server port (default: 8050)")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                        help="Disable Dash debug/hot-reload mode")
    parser.add_argument("--refresh-interval", type=int, default=30,
                        metavar="SEC",
                        help="Auto-refresh check interval in seconds. 0 = disabled (default: 30)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ViewerConfig.from_args(args)
    service = build_service(config)

    # Print summary to verify data loading
    records = service.get_all_records()
    filters = service.get_available_filters()
    print(f"\n{'='*60}")
    print(f"  4DGaussians Metrics Viewer v0.2.0")
    print(f"  Project root: {config.project_root}")
    print(f"  Records loaded: {len(records)}")
    print(f"  Sources: {[s.value for s in filters.sources]}")
    print(f"  Categories: {[c.value for c in filters.categories]}")
    print(f"  Scenes: {filters.scenes}")
    refresh_label = f"{config.auto_refresh_interval_s}s" if config.auto_refresh_interval_s > 0 else "disabled"
    print(f"  Auto-refresh: {refresh_label}")
    print(f"{'='*60}\n")

    for r in records:
        q = r.get_quality()
        q_str = ""
        if q:
            parts = []
            if q.psnr is not None:
                parts.append(f"PSNR={q.psnr:.2f}")
            if q.ssim is not None:
                parts.append(f"SSIM={q.ssim:.4f}")
            if q.vmaf is not None:
                parts.append(f"VMAF={q.vmaf:.2f}")
            q_str = ", ".join(parts)

        comp_str = ""
        if r.compression_metrics:
            cm = r.compression_metrics
            comp_str = f"  ratio={cm.compression_ratio:.2f} savings={cm.savings_pct:.1f}%"

        print(f"  [{r.source.value:16s}] {r.display_name:40s}  {q_str}{comp_str}")

    print()

    # Launch Dash UI
    from .ui.layout import create_app

    app = create_app(service, refresh_interval_s=config.auto_refresh_interval_s)
    print(f"Starting Dash server at http://{config.host}:{config.port}")
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
