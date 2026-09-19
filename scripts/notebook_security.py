"""Shared stdlib-only archive guard embedded in generated Kaggle notebooks."""


NOTEBOOK_ARCHIVE_GUARD = r'''
import hmac, stat

MAX_ARCHIVE_BYTES = 512 * 1024**2
MAX_ARCHIVE_FILES = 20_000
MAX_EXPANDED_BYTES = 512 * 1024**2
MAX_MEMBER_BYTES = 16 * 1024**2
MAX_COMPRESSION_RATIO = 50.0
FREE_SPACE_RESERVE = 256 * 1024**2

if not isinstance(EXPECTED_MANIFEST_SHA256, str) or re.fullmatch(r"[0-9a-f]{64}", EXPECTED_MANIFEST_SHA256) is None:
    raise RuntimeError("set EXPECTED_MANIFEST_SHA256 from the trusted local package command")


def _safe_bundle_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or path.as_posix() != name or any(ord(character) < 32 for character in name)):
        raise RuntimeError("unsafe package path: " + repr(name))
    return path


def _open_bounded_archive(bundle):
    if not bundle.is_file():
        return None, None, 0
    if bundle.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError("package exceeds compressed-size budget")
    archive = zipfile.ZipFile(bundle)
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ARCHIVE_FILES:
        raise RuntimeError("package entry count is outside the allowed budget")
    index, folded, expanded = {}, set(), 0
    for info in infos:
        _safe_bundle_name(info.filename)
        folded_name = info.filename.casefold()
        if info.filename in index or folded_name in folded:
            raise RuntimeError("package contains duplicate or case-colliding paths")
        mode = (info.external_attr >> 16) & 0xFFFF
        if (info.is_dir() or info.flag_bits & 1 or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise RuntimeError("package contains an unsupported entry")
        if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise RuntimeError("package member exceeds size budget")
        if info.file_size and (not info.compress_size
                or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO):
            raise RuntimeError("package member exceeds compression-ratio budget")
        expanded += info.file_size
        if expanded > MAX_EXPANDED_BYTES:
            raise RuntimeError("package exceeds expanded-size budget")
        index[info.filename] = info
        folded.add(folded_name)
    return archive, index, expanded


def _directory_member(bundle, name):
    source = bundle / name
    if source.is_symlink():
        raise RuntimeError("package symlinks are unsupported: " + name)
    path = source.resolve()
    if not path.is_relative_to(bundle.resolve()) or not path.is_file():
        raise RuntimeError("unsafe package file: " + name)
    if path.stat().st_size > MAX_MEMBER_BYTES:
        raise RuntimeError("package member exceeds size budget")
    return path


def _member_bytes(bundle, archive, index, name):
    if archive:
        info = index.get(name)
        if info is None:
            raise RuntimeError("package member is missing: " + name)
        with archive.open(info) as handle:
            content = handle.read(MAX_MEMBER_BYTES + 1)
        if len(content) != info.file_size or len(content) > MAX_MEMBER_BYTES:
            raise RuntimeError("package member size changed while reading")
        return content
    return _directory_member(bundle, name).read_bytes()


def _trusted_manifest(bundle, archive, index, marker):
    raw = _member_bytes(bundle, archive, index, marker)
    digest = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(digest, EXPECTED_MANIFEST_SHA256):
        raise RuntimeError("package manifest is not the trusted release")
    return raw, json.loads(raw)


def _validated_bundle_entries(bundle, archive, index, marker, manifest, prefix):
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > MAX_ARCHIVE_FILES - 1:
        raise RuntimeError("invalid package file list")
    expected, folded, total = {}, set(), 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise RuntimeError("invalid package entry")
        name, size, digest = entry["path"], entry["bytes"], entry["sha256"]
        path = _safe_bundle_name(name)
        if (not name.startswith(prefix) or not isinstance(size, int) or isinstance(size, bool)
                or size < 0 or size > MAX_MEMBER_BYTES or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise RuntimeError("invalid package path, size, or digest")
        if name in expected or name.casefold() in folded:
            raise RuntimeError("package manifest contains duplicate paths")
        if archive:
            if name not in index or index[name].file_size != size:
                raise RuntimeError("package manifest differs from ZIP metadata")
        else:
            if _directory_member(bundle, name).stat().st_size != size:
                raise RuntimeError("package manifest differs from directory metadata")
        expected[name] = entry
        folded.add(name.casefold())
        total += size
        if total > MAX_EXPANDED_BYTES:
            raise RuntimeError("package manifest exceeds expanded-size budget")
    actual = set(index) if archive else {marker, *expected}
    if actual != set(expected) | {marker}:
        raise RuntimeError("package file set differs from manifest")
    disk_root = WORK
    while not disk_root.exists():
        disk_root = disk_root.parent
    if shutil.disk_usage(disk_root).free < total + FREE_SPACE_RESERVE:
        raise RuntimeError("insufficient free space for bounded extraction")
    return list(expected.values())


def _verify_entry(bundle, archive, index, entry):
    digest, size = hashlib.sha256(), 0
    source = archive.open(index[entry["path"]]) if archive else _directory_member(bundle, entry["path"]).open("rb")
    with source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            if size > entry["bytes"] or size > MAX_MEMBER_BYTES:
                raise RuntimeError("package member exceeded manifest size")
            digest.update(block)
    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise RuntimeError("package content hash mismatch: " + entry["path"])


def _copy_entry(bundle, archive, index, entry, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        partial.unlink()
    digest, size = hashlib.sha256(), 0
    source = archive.open(index[entry["path"]]) if archive else _directory_member(bundle, entry["path"]).open("rb")
    try:
        with source, partial.open("xb") as output:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                size += len(block)
                if size > entry["bytes"] or size > MAX_MEMBER_BYTES:
                    raise RuntimeError("package member exceeded manifest size")
                digest.update(block)
                output.write(block)
        if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise RuntimeError("package content changed while copying")
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
'''


def secure_loader_source(*, package_id, identity_key, marker, prefix, zip_name,
                         runtime_prefix, repository_name, experiment=False,
                         extra_validation="", extra_prints=""):
    experiment_check = r'''
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", EXPERIMENT):
    raise RuntimeError("invalid experiment name")
''' if experiment else ""
    runtime = (f'RUNTIME = WORK / ("{runtime_prefix}" + MANIFEST_SHA256[:12]) / EXPERIMENT'
               if experiment else
               f'RUNTIME = WORK / ("{runtime_prefix}" + MANIFEST_SHA256[:12])')
    template = r'''
PACKAGE_ID = __PACKAGE_ID__
MARKER = __MARKER__
__EXPERIMENT_CHECK__
if BUNDLE_INPUT is None:
    candidates = []
    for candidate_marker in INPUT.rglob(MARKER):
        try:
            if candidate_marker.stat().st_size <= MAX_MEMBER_BYTES:
                value = json.loads(candidate_marker.read_text(encoding="utf-8"))
                if value.get(__IDENTITY_KEY__) == PACKAGE_ID:
                    candidates.append(candidate_marker.parent)
        except (OSError, ValueError):
            pass
    if not candidates:
        candidates = list(INPUT.rglob(__ZIP_NAME__))
    if len(candidates) != 1:
        raise RuntimeError(f"found {len(candidates)} matching packages; set BUNDLE_INPUT")
    BUNDLE_INPUT = candidates[0]
BUNDLE_INPUT = Path(BUNDLE_INPUT).resolve()
archive, archive_index, expanded_bytes = _open_bounded_archive(BUNDLE_INPUT)
try:
    raw_manifest, manifest = _trusted_manifest(
        BUNDLE_INPUT, archive, archive_index, MARKER)
    if manifest.get("schema_version") != 1 or manifest.get(__IDENTITY_KEY__) != PACKAGE_ID:
        raise RuntimeError("wrong package identity")
    __EXTRA_VALIDATION__
    entries = _validated_bundle_entries(
        BUNDLE_INPUT, archive, archive_index, MARKER, manifest, __PREFIX__)
    for entry in entries:
        _verify_entry(BUNDLE_INPUT, archive, archive_index, entry)
    MANIFEST_SHA256 = hashlib.sha256(raw_manifest).hexdigest()
    WORK_ROOT = WORK.resolve()
    RUNTIME = WORK_ROOT / (__RUNTIME_PREFIX__ + MANIFEST_SHA256[:12])__RUNTIME_SUFFIX__
    if RUNTIME.is_symlink():
        raise RuntimeError("runtime root must not be a symlink")
    RUNTIME_ROOT = RUNTIME.resolve()
    if not RUNTIME_ROOT.is_relative_to(WORK_ROOT):
        raise RuntimeError("runtime root escapes working directory")
    for entry in entries:
        candidate = RUNTIME_ROOT / entry["path"]
        target = candidate.resolve()
        if candidate.is_symlink() or not target.is_relative_to(RUNTIME_ROOT):
            raise RuntimeError("runtime path escapes package root")
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError("runtime contains an unsafe existing path")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if target.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
                raise RuntimeError("runtime copy was modified; use a new experiment/runtime")
    for entry in entries:
        target = (RUNTIME_ROOT / entry["path"]).resolve()
        if not target.exists():
            _copy_entry(BUNDLE_INPUT, archive, archive_index, entry, target)
    REPO = RUNTIME_ROOT / __REPOSITORY_NAME__
    (RUNTIME_ROOT / MARKER).write_bytes(raw_manifest)
finally:
    if archive:
        archive.close()
print("Verified repository:", REPO)
print("Trusted manifest SHA256:", MANIFEST_SHA256)
__EXTRA_PRINTS__
'''
    values = {
        "__PACKAGE_ID__": repr(package_id),
        "__IDENTITY_KEY__": repr(identity_key),
        "__MARKER__": repr(marker),
        "__PREFIX__": repr(prefix),
        "__ZIP_NAME__": repr(zip_name),
        "__RUNTIME__": runtime,
        "__RUNTIME_PREFIX__": repr(runtime_prefix),
        "__RUNTIME_SUFFIX__": " / EXPERIMENT" if experiment else "",
        "__REPOSITORY_NAME__": repr(repository_name),
        "__EXPERIMENT_CHECK__": experiment_check.strip(),
        "__EXTRA_VALIDATION__": (extra_validation.strip().replace("\n", "\n    ") or "pass"),
        "__EXTRA_PRINTS__": extra_prints.strip(),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    return NOTEBOOK_ARCHIVE_GUARD.strip() + "\n\n" + template.strip() + "\n"
