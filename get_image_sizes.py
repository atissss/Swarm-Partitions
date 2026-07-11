#!/usr/bin/env python3
"""
Scans a folder (recursively) for images and writes a small CSV of
filename,width,height,extension

No dependencies required (pure Python - reads JPEG/PNG/BMP/GIF headers directly,
so it works even without Pillow installed).

Usage:
    python3 get_image_sizes.py /path/to/images/folder
    python3 get_image_sizes.py /path/to/images/folder -o image_sizes.csv
"""

import os
import struct
import sys
import argparse


def get_image_size(filepath):
    """Return (width, height) for jpg/png/bmp/gif, or None if unreadable."""
    with open(filepath, 'rb') as f:
        head = f.read(32)

        # PNG
        if head[:8] == b'\x89PNG\r\n\x1a\n':
            w, h = struct.unpack('>ii', head[16:24])
            return w, h

        # GIF
        if head[:6] in (b'GIF87a', b'GIF89a'):
            w, h = struct.unpack('<HH', head[6:10])
            return w, h

        # BMP
        if head[:2] == b'BM':
            w, h = struct.unpack('<ii', head[18:26])
            return w, abs(h)

        # JPEG
        if head[:2] == b'\xff\xd8':
            f.seek(2)
            while True:
                marker_bytes = f.read(2)
                if len(marker_bytes) < 2:
                    return None
                marker = struct.unpack('>H', marker_bytes)[0]
                if marker == 0xFFD9 or marker == 0xFFDA:  # EOI / SOS
                    return None
                seg_len = struct.unpack('>H', f.read(2))[0]
                # SOF markers (start of frame) hold the dimensions
                if 0xFFC0 <= marker <= 0xFFCF and marker not in (0xFFC4, 0xFFC8, 0xFFCC):
                    f.read(1)  # precision
                    h, w = struct.unpack('>HH', f.read(4))
                    return w, h
                else:
                    f.seek(seg_len - 2, 1)

    return None


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def main():
    parser = argparse.ArgumentParser(description="Extract image dimensions into a CSV")
    parser.add_argument('folder', help='Path to folder containing images (searched recursively)')
    parser.add_argument('-o', '--output', default='image_sizes.csv', help='Output CSV path')
    args = parser.parse_args()

    rows = []
    errors = []
    for root, _, files in os.walk(args.folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            fpath = os.path.join(root, fname)
            try:
                size = get_image_size(fpath)
                if size is None:
                    errors.append(fname)
                    continue
                w, h = size
                # store without extension so it can be matched to the .txt annotation file
                base = os.path.splitext(fname)[0]
                rows.append((base, w, h))
            except Exception as e:
                errors.append(f"{fname} ({e})")

    with open(args.output, 'w') as f:
        f.write("filename,width,height\n")
        for base, w, h in rows:
            f.write(f"{base},{w},{h}\n")

    print(f"Wrote {len(rows)} entries to {args.output}")
    if errors:
        print(f"Could not read {len(errors)} files:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == '__main__':
    main()