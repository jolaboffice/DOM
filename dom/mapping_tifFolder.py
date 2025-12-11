#!/usr/bin/env python
"""
mapping_tifFolder.py

Generates maligner logs for TIF folder.
Processes all TIF files in the specified folder and generates _mal.log files.
"""

import os
import sys
import shutil
import argparse
import subprocess
import pandas as pd
from scipy import signal
import numpy as np
import DOM_lib as lib
import DOM_constants


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate maligner logs for TIFF folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('data_folder', help='Folder containing TIFF data files')
    parser.add_argument('--map', dest='map', default=DOM_constants.DEFAULT_MAP, help=f'Reference map (.tif or .maps file). Default: {DOM_constants.DEFAULT_MAP}')
    parser.add_argument('--bpp', dest='bpp', type=int, default=DOM_constants.DEFAULT_BPP,
                       help=f'Base pairs per pixel (default: {DOM_constants.DEFAULT_BPP})')
    parser.add_argument('--maligner-path', dest='maligner_path', default=DOM_constants.DEFAULT_MALIGNER_PATH,
                       help=f'Path to maligner executable (default: {DOM_constants.DEFAULT_MALIGNER_PATH})')
    
    return parser.parse_args()

def resolve_maligner_path(path_hint):
    """Resolve maligner executable path."""
    mal_path = shutil.which(path_hint)
    if mal_path is None:
        print(f"Error: maligner executable '{path_hint}' not found in PATH", file=sys.stderr)
        sys.exit(1)
    return mal_path

def maligner_call_and_parse(data_file, map_r, map_g, ref_map_maps, bpp, mal_path):
    """Generate data maps, run maligner, parse output and generate _mal.log"""
    data_maps = data_file.replace(".tif", f"_{bpp}.maps")
    _, w2, _ = lib.tif2RGimage(data_file)
    data_r, data_g, peaks = lib.tif2RGprofile(data_file)
    lib.peaks2maps(data_file, w2 * bpp, peaks * bpp, data_maps)

    txt_out = data_file.replace(".tif", f"_{bpp}.txt")
    try:
        with open(txt_out, 'w') as f:
            result = subprocess.run(
                [mal_path, '--reference-is-circular', data_maps, ref_map_maps],
                stdout=f,
                stderr=subprocess.PIPE,
                check=False
            )
        if result.returncode != 0:
            print(f"Warning: maligner returned non-zero exit code ({result.returncode})", file=sys.stderr)
            if result.stderr:
                print(f"Error message: {result.stderr.decode()}", file=sys.stderr)
    except Exception as e:
        print(f"Error running maligner: {e}", file=sys.stderr)
        return

    mal_log = data_file.replace(".tif", "_mal.log")
    try:
        df = pd.read_csv(txt_out, sep='\t', header=0)
    except Exception as e:
        print(f"Error reading maligner output {txt_out}: {e}")
        with open(mal_log, 'w') as log:
            print("rank,shift,direction,scale,score,cc_rg2,cc_rg,cc_r,cc_g", file=log)
        return

    w1 = len(map_g)

    dir_dict = {'F': 1, 'R': -1}
    results = []
    for i, row in df.iterrows():
        ref_start_bp = row.get('ref_start_bp', None)
        is_forward = row.get('is_forward', None)
        query_scaling_factor = row.get('query_scaling_factor', row.get('scale', 1.0))
        total_rescaled_score = row.get('total_rescaled_score', row.get('score', 0.0))

        if ref_start_bp is None or is_forward is None:
            print(f"Skipping row {i}: missing expected fields.")
            continue

        shift = int(ref_start_bp / bpp)
        direction = dir_dict.get(is_forward, 1)
        scale_val = float(query_scaling_factor)
        score = float(total_rescaled_score)
        image_width = int(w2 * scale_val)

        clip1r = lib.normalize(map_r)
        clip1r = np.concatenate((clip1r, clip1r[:int(image_width*1.5)]), axis=0)
        clip1g = lib.normalize(map_g)
        clip1g = np.concatenate((clip1g, clip1g[:int(image_width*1.5)]), axis=0)
        clip2r = lib.normalize(signal.resample(data_r, image_width))
        clip2g = lib.normalize(signal.resample(data_g, image_width))
        mol_r = clip2r[::direction]
        mol_g = clip2g[::direction]

        if shift <= 0:
            shift += w1
        s = lib.compute_shift(clip1g[shift:shift + image_width], mol_g)
        shift -= s
        if shift <= 0:
            shift += w1
        if shift > w1:
            shift -= w1

        ref_r = clip1r[shift:shift + image_width]
        ref_g = clip1g[shift:shift + image_width]
        cc_r = lib.corrcoef(mol_r, ref_r)
        cc_g = lib.dcc(mol_g, ref_g)
        cc2 = cc_r * (cc_g)**2
        cc = cc_r * cc_g
    
        results.append((i, shift, direction, scale_val, score, cc2, cc, cc_r, cc_g))

    with open(mal_log, "w") as log:
        print("rank,shift,direction,scale,score,cc_rg2,cc_rg,cc_r,cc_g", file=log)
        for rank, res in enumerate(results):
            i, shift, direction, scale_val, score, cc2, cc, cc_r, cc_g = res
            print(f"{rank},{shift},{direction},{scale_val:.3f},{score:.6f},{cc2:.6f},{cc:.6f},{cc_r:.6f},{cc_g:.6f}", file=log)

def find_reference_map(data_folder):
    """Find reference map file, checking common locations."""
    candidates = [
        os.path.join(data_folder, DOM_constants.DEFAULT_MAP),
        os.path.join(os.path.dirname(data_folder), DOM_constants.DEFAULT_MAP),
        os.path.join(DOM_constants.DEFAULT_MAP),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def prepare_reference_maps(ref_path, bpp):
    """
    Prepare reference map files (.tif and .maps) from user input.
    map_g and map_r are the R and G channel profiles of the reference map (tif file).
    """
    ref_path_lower = ref_path.lower()
    
    if ref_path_lower.endswith(".maps"):
        # Case 1: User provided .maps file
        ref_map_maps = ref_path
        ref_map_tif = os.path.splitext(ref_map_maps)[0] + ".tif"
        if os.path.exists(ref_map_tif):
            map_r, map_g, map_peaks = lib.tif2RGprofile(ref_map_tif)
        else:
            print(f"Warning: Corresponding .tif file not found: {ref_map_tif}", file=sys.stderr)
            print("Some features may not work correctly without the .tif file.", file=sys.stderr)
            map_r = np.array([])
            map_g = np.array([])
    
    elif ref_path_lower.endswith((".tif", ".tiff")):
        # Case 2: User provided .tif file
        ref_map_tif = ref_path
        map_r, map_g, map_peaks = lib.tif2RGprofile(ref_map_tif)
        map_peaks = np.concatenate(([0], map_peaks, [len(map_g)]))
        ref_map_maps = os.path.splitext(ref_map_tif)[0] + ".maps"
        lib.peaks2maps(ref_map_tif, len(map_g) * bpp, map_peaks * bpp, ref_map_maps)
        
        # Optionally create smoothed maps file
        ref_map_maps_smoothed = os.path.splitext(ref_map_tif)[0] + "_smoothed.maps"
        smooth_cmd = shutil.which("smooth_maps_file")
        if smooth_cmd:
            #print(f"Smoothing maps file...")
            try:
                with open(ref_map_maps_smoothed, 'w') as f:
                    subprocess.run([smooth_cmd, ref_map_maps], stdout=f, check=True)
                ref_map_maps = ref_map_maps_smoothed
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to smooth maps file, using original", file=sys.stderr)
        else:
            print(f"Warning: smooth_maps_file not found, using unsmoothed maps", file=sys.stderr)
    
    else:
        print(f"Error: Reference map must be a .tif or .maps file: {ref_path}", file=sys.stderr)
        sys.exit(1)
    
    return ref_map_maps, map_r, map_g


def process_tiff_files(tiff_files, map_r, map_g, ref_map_maps, bpp, mal_path):
    """
    Process all TIFF files in the folder.
    
    Args:
        tiff_files: List of TIFF file paths
        map_r: Reference R channel profile
        map_g: Reference G channel profile
        ref_map_maps: Path to reference .maps file
        bpp: Base pairs per pixel
        mal_path: Path to maligner executable
    """
    tiff_files_to_process = [f for f in tiff_files if f.endswith(".tif")]
    total_files = len(tiff_files_to_process)
    
    if total_files == 0:
        print("No TIFF files to process.")
        return
    
    print(f"Processing {total_files} TIFF file(s)...")
    for idx, data_file in enumerate(tiff_files_to_process, 1):
        try:
            print(f"[{idx}/{total_files}] Processing {os.path.basename(data_file)}...")
            maligner_call_and_parse(data_file, map_r, map_g, ref_map_maps, bpp, mal_path)
        except Exception as e:
            print(f"Error processing {os.path.basename(data_file)}: {e}", file=sys.stderr)
    
    print(f"All processing complete. ({total_files}/{total_files} files processed)")


def main():
    """Main function to process TIFF folder and generate maligner logs."""
    args = parse_args()
    
    # Validate input reference map
    ref_path = args.map
    if not os.path.exists(ref_path):
        print(f"Error: Reference map not found: {ref_path}", file=sys.stderr)
        sys.exit(1)
    
    # Resolve maligner executable
    bpp = args.bpp
    mal_path = resolve_maligner_path(args.maligner_path)
    
    # Get TIFF files from folder
    tiff_files = lib.Read_TIFF_folder(args.data_folder)
    if not tiff_files:
        print(f"Warning: No TIFF files found in {args.data_folder}", file=sys.stderr)
        return
    
    # Prepare reference maps
    ref_map_maps, map_r, map_g = prepare_reference_maps(ref_path, bpp)
    
    # Process all TIFF files
    process_tiff_files(tiff_files, map_r, map_g, ref_map_maps, bpp, mal_path)

if __name__ == '__main__':
    main()