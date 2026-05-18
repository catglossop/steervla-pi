#!/usr/bin/env python3
"""Recursively extract every rgb.zip under a simlingo_rgb_zips directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract_rgb_archives(root: Path, overwrite: bool) -> tuple[int, int]:
    """Extract all rgb.zip archives under root into each archive's parent."""
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    total = 0
    extracted = 0

    for archive in root.rglob("rgb.zip"):
        if not archive.is_file():
            continue

        total += 1
        dest = archive.parent

        if not overwrite:
            try:
                with zipfile.ZipFile(archive) as zf:
                    members = [m for m in zf.namelist() if m and not m.endswith("/")]
                if members and all((dest / member).exists() for member in members):
                    print(f"[SKIP] {archive} (appears already extracted)")
                    continue
            except zipfile.BadZipFile:
                print(f"[ERROR] Bad zip file: {archive}")
                continue

        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            extracted += 1
            print(f"[OK] Extracted {archive} -> {dest}")
        except zipfile.BadZipFile:
            print(f"[ERROR] Bad zip file: {archive}")

    return total, extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find and extract all rgb.zip archives under simlingo_rgb_zips."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="simlingo_rgb_zips",
        help="Root directory to scan (default: simlingo_rgb_zips)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Extract even when files appear to already exist",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    total, extracted = extract_rgb_archives(root, overwrite=args.overwrite)
    print(f"\nProcessed {total} rgb.zip archive(s), extracted {extracted}.")


if __name__ == "__main__":
    main()
