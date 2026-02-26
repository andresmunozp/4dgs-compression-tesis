"""
Chunker / Assembler for network transmission via Mininet.

Splits a compressed .4dgs archive into fixed-size chunks, each with its
own header, sequence number, and CRC32 checksum.  On the receiver side
the assembler validates and reassembles the chunks.

Chunk wire format
-----------------
  [magic       4 bytes]  b'4DGC'  (4DGS Chunk)
  [chunk_id    2 bytes]  uint16   (0-based sequence number)
  [total       2 bytes]  uint16   (total number of chunks)
  [chunk_type  1 byte ]  uint8    (0x01=header, 0x02=data)
  [payload_len 4 bytes]  uint32   (length of payload)
  [payload     N bytes]
  [crc32       4 bytes]  uint32   (CRC of header + payload)

Files are written as ``chunk_NNNNN_of_TTTTT.4dgsc``.
"""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CHUNK_MAGIC = b"4DGC"
CHUNK_HEADER_FMT = "<4sHHBI"  # magic(4) + id(2) + total(2) + type(1) + payload_len(4) = 13 bytes
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FMT)
CRC_SIZE = 4  # uint32

CHUNK_TYPE_HEADER = 0x01
CHUNK_TYPE_DATA = 0x02


# ── Chunker ───────────────────────────────────────────────────────────────

class ModelChunker:
    """Split a binary archive into network-friendly chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum payload size per chunk (bytes).  Default 1 MB.
    """

    def __init__(self, chunk_size: int = 1_048_576):
        self.chunk_size = chunk_size

    def split(self, archive: bytes) -> List[bytes]:
        """Split *archive* into chunks.  Returns list of chunk blobs."""
        payloads: List[bytes] = []
        offset = 0
        while offset < len(archive):
            end = min(offset + self.chunk_size, len(archive))
            payloads.append(archive[offset:end])
            offset = end

        total = len(payloads)
        chunks: List[bytes] = []
        for idx, payload in enumerate(payloads):
            ctype = CHUNK_TYPE_HEADER if idx == 0 else CHUNK_TYPE_DATA
            header = struct.pack(
                CHUNK_HEADER_FMT,
                CHUNK_MAGIC,
                idx,
                total,
                ctype,
                len(payload),
            )
            body = header + payload
            crc = zlib.crc32(body) & 0xFFFFFFFF
            chunks.append(body + struct.pack("<I", crc))

        return chunks

    def write_chunks(self, chunks: List[bytes], output_dir: str) -> List[str]:
        """Write chunks to individual files in *output_dir*.

        Returns list of file paths written.
        """
        os.makedirs(output_dir, exist_ok=True)
        total = len(chunks)
        paths: List[str] = []
        for idx, chunk in enumerate(chunks):
            fname = f"chunk_{idx:05d}_of_{total:05d}.4dgsc"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "wb") as f:
                f.write(chunk)
            paths.append(fpath)
        return paths

    def split_and_write(self, archive: bytes, output_dir: str) -> List[str]:
        """Convenience: split + write in one call."""
        return self.write_chunks(self.split(archive), output_dir)


# ── Assembler ─────────────────────────────────────────────────────────────

class ModelAssembler:
    """Reassemble chunks into the original archive.

    Validates magic bytes, CRC32 checksums, and sequential ordering.
    """

    @staticmethod
    def read_chunks(input_dir: str) -> List[bytes]:
        """Read all ``.4dgsc`` chunk files from *input_dir*, sorted by name."""
        p = Path(input_dir)
        files = sorted(p.glob("*.4dgsc"))
        if not files:
            raise FileNotFoundError(f"No .4dgsc files found in {input_dir}")
        return [f.read_bytes() for f in files]

    @staticmethod
    def validate_chunk(chunk: bytes) -> Tuple[int, int, int, bytes]:
        """Validate a single chunk and return (chunk_id, total, chunk_type, payload).

        Raises ValueError on any integrity issue.
        """
        if len(chunk) < CHUNK_HEADER_SIZE + CRC_SIZE:
            raise ValueError(f"Chunk too small ({len(chunk)} bytes)")

        # Parse header
        magic, chunk_id, total, ctype, payload_len = struct.unpack_from(
            CHUNK_HEADER_FMT, chunk, 0
        )
        if magic != CHUNK_MAGIC:
            raise ValueError(f"Bad chunk magic: {magic!r}")

        body_end = CHUNK_HEADER_SIZE + payload_len
        if body_end + CRC_SIZE != len(chunk):
            raise ValueError(
                f"Chunk size mismatch: expected {body_end + CRC_SIZE}, got {len(chunk)}"
            )

        # CRC
        body = chunk[:body_end]
        stored_crc = struct.unpack_from("<I", chunk, body_end)[0]
        actual_crc = zlib.crc32(body) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError(
                f"CRC mismatch for chunk {chunk_id}: "
                f"stored {stored_crc:#010x}, computed {actual_crc:#010x}"
            )

        payload = chunk[CHUNK_HEADER_SIZE:body_end]
        return chunk_id, total, ctype, payload

    @classmethod
    def assemble(cls, chunks: List[bytes], verify: bool = True) -> bytes:
        """Reassemble ordered chunks into the original archive.

        Parameters
        ----------
        chunks : list[bytes]
            Chunk blobs (as returned by ``ModelChunker.split()`` or read from files).
        verify : bool
            If True (default), validate CRC and ordering.
        """
        parsed = []
        for chunk in chunks:
            cid, total, ctype, payload = cls.validate_chunk(chunk)
            parsed.append((cid, total, ctype, payload))

        # Sort by chunk_id
        parsed.sort(key=lambda x: x[0])

        # Validate completeness
        expected_total = parsed[0][1]
        ids = [p[0] for p in parsed]
        if len(parsed) != expected_total:
            raise ValueError(
                f"Expected {expected_total} chunks, got {len(parsed)}.  "
                f"Missing: {set(range(expected_total)) - set(ids)}"
            )
        if ids != list(range(expected_total)):
            raise ValueError(f"Non-contiguous chunk IDs: {ids}")

        return b"".join(p[3] for p in parsed)

    @classmethod
    def assemble_from_dir(cls, input_dir: str, verify: bool = True) -> bytes:
        """Read chunks from directory and assemble."""
        raw_chunks = cls.read_chunks(input_dir)
        return cls.assemble(raw_chunks, verify=verify)
