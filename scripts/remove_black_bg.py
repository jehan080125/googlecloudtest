"""Remove black sprite backgrounds via corner flood-fill + fringe expansion."""
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


def flood_fill_corners(im: Image.Image, tolerance: int = 0) -> int:
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
        if r <= tolerance and g <= tolerance and b <= tolerance:
            px[x, y] = (r, g, b, 0)
            removed += 1
            seeds.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    im.paste(rgba)
    return removed


def expand_transparency(im: Image.Image, tolerance: int = 40) -> int:
    rgba = im.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size
    removed = 0
    seeds: deque[tuple[int, int]] = deque()

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
            if r <= tolerance and g <= tolerance and b <= tolerance:
                px[nx, ny] = (r, g, b, 0)
                removed += 1
                seeds.append((nx, ny))

    im.paste(rgba)
    return removed


def process_sprite(path: Path, fringe_tolerance: int = 40) -> None:
    im = Image.open(path).convert("RGBA")
    corner_removed = flood_fill_corners(im, tolerance=0)
    fringe_removed = expand_transparency(im, tolerance=fringe_tolerance)
    im.save(path, "PNG")

    alpha = im.split()[-1]
    transparent = sum(1 for a in alpha.getdata() if a == 0)
    total = im.size[0] * im.size[1]
    print(
        f"{path.name}: corner_removed={corner_removed}, fringe_removed={fringe_removed}, "
        f"transparent={transparent}/{total} ({100 * transparent / total:.1f}%)"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    public = root / "frontend/public/epitaph-characters"
    dist = root / "frontend/dist/epitaph-characters"

    # Minsoo needs higher fringe tolerance due to thick anti-aliased halo (RGB 16-40).
    configs = [("minsoo.png", 42), ("yang-jinhuk.png", 45)]
    for name, tol in configs:
        src = public / name
        if not src.exists():
            print(f"MISSING: {src}")
            continue
        process_sprite(src, fringe_tolerance=tol)
        dist.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dist / name)


if __name__ == "__main__":
    main()
