"""Build investigation -Scene cutouts from inventory sources (black or white bg removal)."""
from __future__ import annotations

import shutil
import sys
from collections import deque
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("PIL not installed", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "frontend" / "public"
DIST = ROOT / "frontend" / "dist"


def _is_black(r: int, g: int, b: int, tolerance: int) -> bool:
    return r <= tolerance and g <= tolerance and b <= tolerance


def _is_white(r: int, g: int, b: int, tolerance: int) -> bool:
    floor = 255 - tolerance
    return r >= floor and g >= floor and b >= floor


def flood_fill_corners(
    im: Image.Image,
    *,
    mode: str = "black",
    tolerance: int = 0,
) -> int:
    rgba = im.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size
    removed = 0
    visited = bytearray(w * h)
    seeds: deque[tuple[int, int]] = deque()

    for x in range(w):
        seeds.append((x, 0))
        seeds.append((x, h - 1))
    for y in range(h):
        seeds.append((0, y))
        seeds.append((w - 1, y))

    match = _is_black if mode == "black" else _is_white

    while seeds:
        x, y = seeds.popleft()
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        i = y * w + x
        if visited[i]:
            continue
        visited[i] = 1
        r, g, b, a = px[x, y]
        if a == 0:
            continue
        if match(r, g, b, tolerance):
            px[x, y] = (r, g, b, 0)
            removed += 1
            seeds.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    im.paste(rgba)
    return removed


def expand_transparency(
    im: Image.Image,
    *,
    mode: str = "black",
    tolerance: int = 40,
) -> int:
    rgba = im.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size
    removed = 0
    seeds: deque[tuple[int, int]] = deque()
    match = _is_black if mode == "black" else _is_white

    for y in range(h):
        for x in range(w):
            if px[x, y][3] == 0:
                seeds.append((x, y))

    visited = bytearray(w * h)
    while seeds:
        x, y = seeds.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            i = ny * w + nx
            if visited[i]:
                continue
            visited[i] = 1
            r, g, b, a = px[nx, ny]
            if a == 0:
                seeds.append((nx, ny))
                continue
            if match(r, g, b, tolerance):
                px[nx, ny] = (r, g, b, 0)
                removed += 1
                seeds.append((nx, ny))

    im.paste(rgba)
    return removed


def process_sprite(
    path: Path,
    *,
    mode: str = "black",
    corner_tolerance: int = 0,
    fringe_tolerance: int = 40,
) -> None:
    im = Image.open(path).convert("RGBA")
    corner_removed = flood_fill_corners(
        im, mode=mode, tolerance=corner_tolerance
    )
    fringe_removed = expand_transparency(
        im, mode=mode, tolerance=fringe_tolerance
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG")

    alpha = im.split()[-1]
    transparent = sum(1 for a in alpha.getdata() if a == 0)
    total = im.size[0] * im.size[1]
    print(
        f"{path.name}: mode={mode} corner={corner_removed} fringe={fringe_removed} "
        f"transparent={transparent}/{total} ({100 * transparent / total:.1f}%)"
    )


def build_scene(src_rel: str, dst_rel: str, *, mode: str, fringe: int = 40) -> None:
    src = PUBLIC / src_rel.lstrip("/")
    dst = PUBLIC / dst_rel.lstrip("/")
    if not src.exists():
        print(f"MISSING source: {src}")
        return
    shutil.copy2(src, dst)
    process_sprite(dst, mode=mode, fringe_tolerance=fringe)

    dist_dst = DIST / dst_rel.lstrip("/")
    dist_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dst, dist_dst)


def main() -> None:
    jobs = [
        # turnabout_clock — transparent PhotoRoom cutouts (no PIL bg removal)
        # 가방: investigation/가방-Photoroom.png -> 가방-Scene.png, 가방.png
        # 돌하르방: investigation/돌하르방-Photoroom.png -> 돌하르방-Scene.png, 돌하르방.png
        # turnabout_clock — white-bg court evidence
        ("court-assets/evidence/im2.png", "court-assets/evidence/im2-Scene.png", "white", 18),
        ("court-assets/evidence/im3.png", "court-assets/evidence/im3-Scene.png", "white", 18),
        # disaster_epitaph: use originals from repo-root 증거사진/ (no auto-processing)
        # vx2설명-Scene.png: white-bg removal only (증거사진/vx2설명.png -> PIL white flood-fill); never black mode
    ]

    for src, dst, mode, fringe in jobs:
        if src == dst:
            path = PUBLIC / src
            if not path.exists():
                print(f"MISSING in-place: {path}")
                continue
            process_sprite(path, mode=mode, fringe_tolerance=fringe)
            dist_dst = DIST / src
            dist_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dist_dst)
        else:
            build_scene(src, dst, mode=mode, fringe=fringe)


if __name__ == "__main__":
    main()
