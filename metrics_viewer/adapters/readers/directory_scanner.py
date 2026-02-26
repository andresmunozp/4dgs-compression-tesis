"""Auto-discovery of result files in the workspace."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type

from ...domain.ports import IDataSourceReader

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredFile:
    """A file found during scanning, paired with the reader that can handle it."""
    path: Path
    reader: IDataSourceReader


class DirectoryScanner:
    """Walks known directories and matches files to the appropriate reader.

    Usage::

        scanner = DirectoryScanner(readers=[
            BenchmarkJsonReader(),
            BenchmarkCsvReader(),
            TrainingJsonReader(),
            VmafJsonReader(),
        ])
        found = scanner.scan(project_root)
    """

    # Directories to scan (relative to project root)
    DEFAULT_SCAN_DIRS = [
        "benchmark_results",
        "results_json",
        "compressed_output",
        "decompressed_output",
    ]

    # File extensions to consider
    EXTENSIONS = {".json", ".csv"}

    def __init__(self, readers: List[IDataSourceReader]) -> None:
        self._readers = readers

    def scan(
        self,
        base_dir: Path,
        scan_dirs: List[str] | None = None,
        max_depth: int = 3,
    ) -> List[DiscoveredFile]:
        """Recursively scan directories and match files to readers.

        Args:
            base_dir: Project root directory.
            scan_dirs: Subdirectories to scan (relative).  Defaults to
                ``DEFAULT_SCAN_DIRS``.
            max_depth: Maximum recursion depth.

        Returns:
            List of ``DiscoveredFile`` instances.
        """
        dirs = scan_dirs or self.DEFAULT_SCAN_DIRS
        discovered: List[DiscoveredFile] = []
        seen_paths: set[str] = set()

        for rel_dir in dirs:
            target = base_dir / rel_dir
            if not target.exists():
                logger.debug("Scan dir not found: %s", target)
                continue
            self._walk(target, discovered, seen_paths, depth=0, max_depth=max_depth)

        logger.info("Discovered %d data files", len(discovered))
        return discovered

    def _walk(
        self,
        directory: Path,
        out: List[DiscoveredFile],
        seen: set[str],
        depth: int,
        max_depth: int,
    ) -> None:
        if depth > max_depth:
            return

        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            logger.warning("Permission denied: %s", directory)
            return

        for entry in entries:
            if entry.is_dir():
                self._walk(entry, out, seen, depth + 1, max_depth)
            elif entry.is_file() and entry.suffix.lower() in self.EXTENSIONS:
                canonical = str(entry.resolve())
                if canonical in seen:
                    continue
                reader = self._find_reader(entry)
                if reader is not None:
                    out.append(DiscoveredFile(path=entry, reader=reader))
                    seen.add(canonical)
                    logger.debug("Matched %s → %s", entry.name, type(reader).__name__)

    def _find_reader(self, path: Path) -> IDataSourceReader | None:
        """Return the first reader that supports the given path."""
        for reader in self._readers:
            if reader.supports(path):
                return reader
        return None
