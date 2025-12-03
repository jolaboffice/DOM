#!/usr/bin/env python3
"""
DOM_make_info.py

Generates metadata *.info files for each TIF (molecule length in bp and BPP)
to speed up DOM_match.py by avoiding repeated TIF reads.
"""

import os
import sys
import glob
import argparse
import DOM_lib as lib
import DOM_constants


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DOM *.info metadata files for TIFFs.")
    parser.add_argument('data_folder',
                        help="Folder containing tif data files.")
    parser.add_argument('--bpp', type=int, default=DOM_constants.DEFAULT_BPP,
                        help=f"Base pairs per pixel (default: {DOM_constants.DEFAULT_BPP}).")
    return parser.parse_args()


def main():
    args = parse_args()
    data_folder = os.path.abspath(args.data_folder)
    if not os.path.isdir(data_folder):
        print(f"Error: data folder not found: {data_folder}")
        sys.exit(1)

    tif_files = []
    for ext in ("*.tif", "*.tiff"):
        tif_files.extend(glob.glob(os.path.join(data_folder, ext)))
    tif_files = sorted(tif_files)
    if not tif_files:
        print("No TIFF files found.")
        return

    print(f"Generating .info metadata for {len(tif_files)} TIFF file(s) with BPP={args.bpp}...\n")
    for idx, tif_file in enumerate(tif_files, 1):
        print(f"[{idx}/{len(tif_files)}] {os.path.basename(tif_file)}")
        try:
            _, w2, _ = lib.tif2RGimage(tif_file)
        except Exception as exc:
            print(f"  ⚠  Failed to read TIFF data: {exc}")
            continue

        info_data = {
            "filename": os.path.basename(tif_file),
            "bpp": args.bpp,
            "mol_length_bp": w2 * args.bpp
        }
        info_file = tif_file.replace(".tif", ".info")
        info_file = info_file.replace(".tiff", ".info")
        try:
            with open(info_file, "w") as f:
                for key, val in info_data.items():
                    f.write(f"{key}:{val}\n")
            print(f"  ✓ Saved {os.path.basename(info_file)}")
        except Exception as exc:
            print(f"  ⚠  Failed to write {info_file}: {exc}")
    print("\nDone.")


if __name__ == '__main__':
    main()

