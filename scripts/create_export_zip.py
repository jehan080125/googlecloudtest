"""Create a distributable zip of the 역전재판 workspace."""
from __future__ import annotations

import os
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_NAME = f"역전재판_export_{date.today().strftime('%Y%m%d')}.zip"
OUT_PATH = ROOT / OUT_NAME

EXCLUDE_DIR_NAMES = {
    "node_modules",
    "__pycache__",
    ".git",
    ".pytest_cache",
    "dist",
    ".cursor",
    ".vite",
}
EXCLUDE_FILE_NAMES = {
    OUT_NAME,
    ".DS_Store",
    "Thumbs.db",
}
EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo"}


def should_skip(path: Path, rel: Path) -> bool:
    parts = rel.parts
    if any(part in EXCLUDE_DIR_NAMES for part in parts):
        return True
    if path.name in EXCLUDE_FILE_NAMES:
        return True
    if path.suffix in EXCLUDE_FILE_SUFFIXES:
        return True
    # Exclude secrets; keep examples
    if path.name == ".env":
        return True
    return False


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        current = Path(dirpath)
        rel_dir = current.relative_to(ROOT)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in EXCLUDE_DIR_NAMES and d != OUT_NAME.replace(".zip", "")
        ]
        for name in filenames:
            file_path = current / name
            rel = file_path.relative_to(ROOT)
            if should_skip(file_path, rel):
                continue
            yield file_path, rel.as_posix()


def main() -> None:
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    count = 0
    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path, arcname in iter_files():
            zf.write(file_path, arcname)
            count += 1
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"ZIP: {OUT_PATH}")
    print(f"Files: {count}")
    print(f"Size: {OUT_PATH.stat().st_size:,} bytes ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
