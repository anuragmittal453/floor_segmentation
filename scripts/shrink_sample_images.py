#!/usr/bin/env python3
"""
scripts/shrink_sample_images.py

Resizes and recompresses the overlay/sample images in overlays/ and
assets/ down to a repo-friendly size. These are large PNG screenshots of
segmentation output composited at the ORIGINAL stock-photo resolution
(often 5000-6000px on the long edge) — fine for local inspection, but
20-58MB per file is far too large to commit to git (a 24-image folder at
that size is well over 1GB).

Converts to JPEG (photographic content, lossy compression is appropriate
here — these are illustrative samples, not ground-truth data) and resizes
so the long edge is at most --max-size pixels (default 1600, comfortably
readable in a README/GitHub preview without being huge).

By default this OVERWRITES files in place after resizing (with a .png
extension changed to .jpg) and removes the oversized original — pass
--dry-run first to see what it would do without touching anything.

Usage:
    # Preview what would happen, no changes made
    python scripts/shrink_sample_images.py --dry-run

    # Actually shrink everything under overlays/ and assets/
    python scripts/shrink_sample_images.py

    # Custom directories / max size
    python scripts/shrink_sample_images.py --dirs overlays assets --max-size 1200
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency. Install with:\n    pip install opencv-python-headless\n")


def find_large_images(directories: list[Path], min_size_mb: float) -> list[Path]:
    min_bytes = min_size_mb * 1024 * 1024
    found = []
    for directory in directories:
        if not directory.is_dir():
            print(f"  (skipping, not a directory: {directory})")
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if path.stat().st_size >= min_bytes:
                    found.append(path)
    return sorted(found)


def shrink_image(path: Path, max_size: int, jpeg_quality: int, dry_run: bool) -> tuple[Path, int, int]:
    """Returns (output_path, original_bytes, new_bytes). new_bytes is 0 in dry-run mode."""
    original_bytes = path.stat().st_size

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"  WARNING: could not read {path}, skipping")
        return path, original_bytes, original_bytes

    height, width = image.shape[:2]
    long_edge = max(height, width)

    if long_edge > max_size:
        scale = max_size / long_edge
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        new_width, new_height = width, height

    output_path = path.with_suffix(".jpg")

    if dry_run:
        print(f"  {path.name}: {width}x{height} ({original_bytes / 1e6:.1f} MB) "
              f"-> {new_width}x{new_height} JPEG q={jpeg_quality} -> {output_path.name}")
        return output_path, original_bytes, 0

    success = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not success:
        print(f"  WARNING: failed to write {output_path}, leaving original untouched")
        return path, original_bytes, original_bytes

    new_bytes = output_path.stat().st_size

    if output_path != path:
        path.unlink()

    return output_path, original_bytes, new_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", default=["overlays", "assets"], help="Directories to scan (default: overlays assets)")
    parser.add_argument("--max-size", type=int, default=1600, help="Max long-edge size in pixels (default: 1600)")
    parser.add_argument("--jpeg-quality", type=int, default=88, help="JPEG quality 1-100 (default: 88)")
    parser.add_argument("--min-size-mb", type=float, default=1.0, help="Only touch files at least this large (default: 1.0 MB) — leaves already-small files alone")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without changing any files")

    args = parser.parse_args()

    directories = [Path(d) for d in args.dirs]

    print(f"Scanning: {[str(d) for d in directories]}")
    print(f"Threshold: files >= {args.min_size_mb} MB")
    print(f"Target: max {args.max_size}px long edge, JPEG quality {args.jpeg_quality}")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will modify files)'}")
    print()

    targets = find_large_images(directories, args.min_size_mb)

    if not targets:
        print("No files found above the size threshold. Nothing to do.")
        return

    print(f"Found {len(targets)} file(s) to process:\n")

    total_original = 0
    total_new = 0

    for path in targets:
        output_path, original_bytes, new_bytes = shrink_image(
            path, args.max_size, args.jpeg_quality, args.dry_run,
        )
        total_original += original_bytes
        total_new += new_bytes

        if not args.dry_run:
            print(f"  {path.name}: {original_bytes / 1e6:.1f} MB -> "
                  f"{output_path.name}: {new_bytes / 1e6:.1f} MB "
                  f"({100 * (1 - new_bytes / original_bytes):.0f}% smaller)")

    print()
    print(f"Total before: {total_original / 1e6:.1f} MB")
    if not args.dry_run:
        print(f"Total after:  {total_new / 1e6:.1f} MB")
        print(f"Saved:        {(total_original - total_new) / 1e6:.1f} MB "
              f"({100 * (1 - total_new / total_original):.0f}%)")
    else:
        print("(Dry run — no files were changed. Re-run without --dry-run to apply.)")


if __name__ == "__main__":
    main()