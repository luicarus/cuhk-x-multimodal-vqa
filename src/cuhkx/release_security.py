"""Bounded, manifest-verified extraction for local release builders."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile


@dataclass(frozen=True)
class ArchiveLimits:
    max_archive_bytes: int = 512 * 1024**2
    max_entries: int = 20_000
    max_expanded_bytes: int = 512 * 1024**2
    max_member_bytes: int = 16 * 1024**2
    max_compression_ratio: float = 50.0
    free_space_reserve: int = 256 * 1024**2


DEFAULT_ARCHIVE_LIMITS = ArchiveLimits()
_SHA256 = re.compile(r"[0-9a-f]{64}")


def safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or path.as_posix() != name or any(ord(character) < 32 for character in name)):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def _zip_index(archive: Path, zipped: zipfile.ZipFile, limits: ArchiveLimits):
    if archive.stat().st_size > limits.max_archive_bytes:
        raise ValueError("archive exceeds compressed-size budget")
    infos = zipped.infolist()
    if not infos or len(infos) > limits.max_entries:
        raise ValueError("archive entry count is outside the allowed budget")
    index, folded, expanded = {}, set(), 0
    for info in infos:
        safe_member_name(info.filename)
        folded_name = info.filename.casefold()
        if info.filename in index or folded_name in folded:
            raise ValueError("archive contains duplicate or case-colliding paths")
        if info.is_dir() or info.flag_bits & 1:
            raise ValueError("archive directories and encrypted entries are unsupported")
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type not in (0, stat.S_IFREG):
            raise ValueError("archive links and special files are unsupported")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError("unsupported archive compression")
        if info.file_size < 0 or info.file_size > limits.max_member_bytes:
            raise ValueError("archive member exceeds size budget")
        if info.file_size and (not info.compress_size or
                info.file_size / info.compress_size > limits.max_compression_ratio):
            raise ValueError("archive member exceeds compression-ratio budget")
        expanded += info.file_size
        if expanded > limits.max_expanded_bytes:
            raise ValueError("archive exceeds expanded-size budget")
        index[info.filename] = info
        folded.add(folded_name)
    return index, expanded


def _read_member(zipped, info, limits):
    with zipped.open(info) as handle:
        content = handle.read(limits.max_member_bytes + 1)
    if len(content) != info.file_size or len(content) > limits.max_member_bytes:
        raise ValueError("archive member size changed while reading")
    return content


def _hash_member(zipped, info, limits):
    digest, size = hashlib.sha256(), 0
    with zipped.open(info) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            if size > info.file_size or size > limits.max_member_bytes:
                raise ValueError("archive member exceeded verified size")
            digest.update(block)
    if size != info.file_size:
        raise ValueError("archive member size differs from central directory")
    return size, digest.hexdigest()


def _validated_entries(manifest, index, marker, limits, validate_name):
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > limits.max_entries - 1:
        raise ValueError("invalid manifest file list")
    expected, folded, total = {}, set(), 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid manifest entry")
        name, size, digest = entry["path"], entry["bytes"], entry["sha256"]
        validate_name(name)
        if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                or size > limits.max_member_bytes or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None):
            raise ValueError("invalid manifest size or digest")
        if name in expected or name.casefold() in folded:
            raise ValueError("manifest contains duplicate or case-colliding paths")
        if name not in index or index[name].file_size != size:
            raise ValueError("manifest size/hash differs from archive metadata")
        expected[name] = entry
        folded.add(name.casefold())
        total += size
        if total > limits.max_expanded_bytes:
            raise ValueError("manifest exceeds expanded-size budget")
    if set(index) != set(expected) | {marker}:
        raise ValueError("archive file set differs from manifest")
    return expected


def extract_verified_archive(archive, destination, marker, validate_manifest,
                             *, validate_name=safe_member_name,
                             limits=DEFAULT_ARCHIVE_LIMITS):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if not archive.is_file():
        raise ValueError("release archive is missing")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("extraction target must be empty")
    with zipfile.ZipFile(archive) as zipped:
        index, expanded = _zip_index(archive, zipped, limits)
        if marker not in index:
            raise ValueError("release manifest is missing")
        raw_manifest = _read_member(zipped, index[marker], limits)
        manifest = json.loads(raw_manifest)
        if not validate_manifest(manifest):
            raise ValueError("wrong release identity")
        expected = _validated_entries(manifest, index, marker, limits, validate_name)
        for name, entry in expected.items():
            size, digest = _hash_member(zipped, index[name], limits)
            if size != entry["bytes"] or digest != entry["sha256"]:
                raise ValueError(f"archive content hash mismatch: {name}")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < expanded + limits.free_space_reserve:
            raise ValueError("insufficient free space for bounded extraction")
        destination.mkdir(parents=True, exist_ok=True)
        try:
            for name, entry in expected.items():
                target = (destination / name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("archive target escapes extraction root")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                digest, size = hashlib.sha256(), 0
                with zipped.open(index[name]) as source, partial.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(block)
                        if size > entry["bytes"] or size > limits.max_member_bytes:
                            raise ValueError("archive member exceeded manifest size")
                        digest.update(block)
                        output.write(block)
                if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError("archive content changed during extraction")
                os.replace(partial, target)
            (destination / marker).write_bytes(raw_manifest)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    return manifest, hashlib.sha256(raw_manifest).hexdigest()
