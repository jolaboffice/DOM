#!/usr/bin/env python3
"""
DOM_match.py

Generates DOM matching results by running PCC and maligner alignment methods.
Reads or generates _pcc.log and _mal.log files, filters results to find best matches,
and writes per-TIF CSV summaries (_matching_results.csv).
"""

import os
import sys
import glob
import argparse
import subprocess
import pandas as pd
import DOM_lib as lib
import DOM_constants

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Column names for output CSV
OUTPUT_COLUMNS = ['method', 'start_bp', 'direction', 'scale', 'cc_r', 'cc_g', 'cc_rg', 'cc_rg2', 'rank', 'misc']


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate DOM matching results (CSV only, no GUI)."
    )
    parser.add_argument('data_target',
                        help="Folder or file containing tif data.")
    parser.add_argument('--map', dest='map', default=DOM_constants.DEFAULT_MAP,
                        help="Reference map (TIF file). Required if not using default.")
    parser.add_argument('--file', dest='file_opt', default=None,
                        help="Single tif file to process (overrides folder).")
    parser.add_argument('--bpp', dest='bpp', type=int, default=DOM_constants.DEFAULT_BPP,
                        help=f"Base pairs per pixel (default: {DOM_constants.DEFAULT_BPP}).")
    parser.add_argument('--start-bpp', type=int, default=DOM_constants.DEFAULT_START_BPP,
                        dest='start_bpp',
                        help=f'Start BPP for CC calculation range (default: {DOM_constants.DEFAULT_START_BPP})')
    parser.add_argument('--end-bpp', type=int, default=DOM_constants.DEFAULT_END_BPP,
                        dest='end_bpp',
                        help=f'End BPP for CC calculation range (default: {DOM_constants.DEFAULT_END_BPP})')
    parser.add_argument('--shift-window', type=int, default=DOM_constants.DEFAULT_SHIFT_WINDOW,
                        dest='shift_window',
                        help=f'Window size in pixels for grouping similar shifts (default: {DOM_constants.DEFAULT_SHIFT_WINDOW})')
    return parser.parse_args()


def resolve_paths(args):
    """Resolve and validate map and data paths."""
    map_path = os.path.abspath(args.map)
    if not os.path.exists(map_path):
        print(f"Error: map path not found: {map_path}", file=sys.stderr)
        sys.exit(1)

    if args.file_opt:
        data_target = os.path.abspath(args.file_opt)
        if not os.path.isfile(data_target):
            print(f"Error: file not found: {data_target}", file=sys.stderr)
            sys.exit(1)
        return map_path, [data_target], False

    data_path = os.path.abspath(args.data_target)
    if os.path.isdir(data_path):
        return map_path, [data_path], True
    elif os.path.isfile(data_path):
        return map_path, [data_path], False
    else:
        print(f"Error: data target not found: {data_path}", file=sys.stderr)
        sys.exit(1)


def _read_log_file(data_file, suffix, log_type):
    """Helper function to read log files (PCC or Maligner)."""
    log_file = data_file.replace(".tif", suffix)
    if not os.path.exists(log_file):
        print(f"  ⚠  Missing {log_type} log: {log_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(log_file)
    except pd.errors.EmptyDataError:
        print(f"  ⚠  Empty {log_type} log: {log_file}")
        return pd.DataFrame()
    except Exception as exc:
        print(f"  ⚠  Failed to read {log_file}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()
    return df


def read_pcc_results(data_file):
    """Read and process PCC log file."""
    df = _read_log_file(data_file, "_pcc.log", "PCC")
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    df['method'] = 'pcc'
    df['g_pos'] = 0.0
    df['r_pos'] = 0.0
    df['cc_rg2'] = df['score']
    df['rank'] = df.index
    return df[['method','shift','direction','scale','score','g_pos','r_pos','cc_rg2','cc_r','cc_g','cc_rg','rank']]


def read_maligner_results(data_file):
    """Read and process maligner log file."""
    df = _read_log_file(data_file, "_mal.log", "maligner")
    if df.empty:
        return pd.DataFrame()

    if 'method' not in df.columns:
        df['method'] = 'maligner'
    else:
        df['method'] = df['method'].fillna('maligner')

    # Some maligner logs may not have g_pos/r_pos columns
    if 'g_pos' not in df.columns:
        df['g_pos'] = 0.0
    if 'r_pos' not in df.columns:
        df['r_pos'] = 0.0

    return df[['method','shift','direction','scale','score','g_pos','r_pos','cc_rg2','cc_r','cc_g','cc_rg','rank']]


def filter_results(df, bpp, mol_length_bp, shift_window):
    """Filter matching results by grouping similar shifts and removing overlapping ranges.

    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['start_bp'] = df['shift'] * bpp
    df['end_bp'] = df['start_bp'] + (mol_length_bp * df['scale'])
    df = df.sort_values(by='cc_rg2', ascending=False).reset_index(drop=True)

    pcc_df = df[df['method'] == 'pcc'].copy().reset_index(drop=True)
    mal_df = df[df['method'] == 'maligner'].copy().reset_index(drop=True)

    def _group_by_shift(df_subset, shift_window_bp, use_count_as_score=False):
        """Group results by start_bp within shift_window. Returns filtered DataFrame."""
        if df_subset.empty:
            return pd.DataFrame(columns=df_subset.columns)
        
        groups = []
        for idx, row in df_subset.iterrows():
            start_bp = row['start_bp']
            matched = False
            # Check if this start_bp matches any existing group
            for grp in groups:
                if abs(start_bp - grp['shift']) <= shift_window_bp:
                    if use_count_as_score:
                        grp['count'] += 1  # Increment count for PCC
                    matched = True
                    break
            if matched:
                continue
            # Create new group if we haven't reached the limit
            if len(groups) < 10:
                groups.append({
                    'shift': start_bp,
                    'count': 1 if use_count_as_score else 0,
                    'record': row.copy()
                })
            else:
                break  # Early exit when we have 10 groups
        
        if not groups:
            return pd.DataFrame(columns=df_subset.columns)
        
        filtered = pd.DataFrame([grp['record'] for grp in groups])
        if use_count_as_score:
            # Set score to the count of duplicate candidates in each group (PCC)
            filtered['score'] = [grp['count'] for grp in groups]
            filtered['rank'] = range(len(filtered))
        # For maligner, keep original score and rank (already in record)
        return filtered

    shift_window_bp = shift_window * bpp
    pcc_filtered = _group_by_shift(pcc_df, shift_window_bp, use_count_as_score=True)
    mal_filtered = _group_by_shift(mal_df, shift_window_bp, use_count_as_score=False)

    combined = pd.concat([pcc_filtered, mal_filtered], ignore_index=True)
    combined = combined.sort_values(by='cc_rg2', ascending=False).reset_index(drop=True)
    # Rename 'score' to 'misc' for output
    combined = combined.rename(columns={'score': 'misc'})

    return combined


def parse_info_file(info_path):
    """Parse .info file into dict."""
    data = {}
    try:
        with open(info_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip()
    except Exception:
        return None
    return data


def get_info_path(tif_path):
    base, _ = os.path.splitext(tif_path)
    return base + ".info"


def run_make_info(target_folder, bpp):
    maker_path = os.path.join(SCRIPT_DIR, "DOM_make_info.py")
    cmd = [sys.executable, maker_path, target_folder, "--bpp", str(bpp)]
    subprocess.run(cmd, check=True)


def _validate_info_data(info_data, bpp):
    """Validate info file data. Returns True if valid."""
    if not info_data:
        return False
    try:
        mol_len = float(info_data.get("mol_length_bp", 0))
        bpp_val = float(info_data.get("bpp", bpp))
        return mol_len > 0 and bpp_val > 0 and int(bpp_val) == bpp
    except (TypeError, ValueError):
        return False

def ensure_single_info(tif_path, bpp):
    """Ensure a single TIFF has a valid .info file."""
    info_path = get_info_path(tif_path)
    info_data = parse_info_file(info_path)
    if _validate_info_data(info_data, bpp):
        return info_data
    
    # Generate .info file directly for this single file (don't process entire folder)
    print(f"Info missing/outdated for {os.path.basename(tif_path)}. Generating .info file...")
    try:
        _, w2, _ = lib.tif2RGimage(tif_path)
    except Exception as exc:
        print(f"ERROR: Failed to read TIFF data from {tif_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    
    info_data = {
        "filename": os.path.basename(tif_path),
        "bpp": bpp,
        "mol_length_bp": w2 * bpp
    }
    try:
        with open(info_path, "w") as f:
            for key, val in info_data.items():
                f.write(f"{key}:{val}\n")
    except Exception as exc:
        print(f"ERROR: Failed to write {info_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    
    return info_data


def ensure_info_files(tif_files, data_folder, bpp):
    """Ensure all TIFFs have .info metadata; run DOM_make_info.py if needed."""
    missing = []
    for tif in tif_files:
        info_path = get_info_path(tif)
        info_data = parse_info_file(info_path) if os.path.exists(info_path) else None
        if not _validate_info_data(info_data, bpp):
            missing.append(tif)

    if not missing:
        return

    print(f"Info files missing/outdated for {len(missing)} file(s). Running DOM_make_info.py ...")
    try:
        run_make_info(data_folder, bpp)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: DOM_make_info.py failed ({exc.returncode}).", file=sys.stderr)
        sys.exit(1)


def run_helper_script(script_name, extra_args):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"Warning: {script_name} not found at {script_path}")
        return False
    cmd = [sys.executable, script_path] + extra_args
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Warning: {script_name} failed (exit code {exc.returncode}).")
        return False


def ensure_log_files(map_path, data_folder, tif_files, bpp, start_bpp=None, end_bpp=None):
    """Ensure PCC and maligner logs exist; run helper scripts when necessary."""
    missing_pcc = [t for t in tif_files if not os.path.exists(t.replace(".tif", "_pcc.log"))]
    missing_mal = [t for t in tif_files if not os.path.exists(t.replace(".tif", "_mal.log"))]

    if missing_pcc:
        print(f"PCC logs missing for {len(missing_pcc)} file(s). Running pcc_tifFolder.py ...")
        pcc_args = [
            data_folder,  # Positional argument
            "--map", map_path,
            "--bpp", str(bpp),
        ]
        if start_bpp is not None:
            pcc_args.extend(["--start-bpp", str(start_bpp)])
        if end_bpp is not None:
            pcc_args.extend(["--end-bpp", str(end_bpp)])
        run_helper_script("pcc_tifFolder.py", pcc_args)

    if missing_mal:
        print(f"Maligner logs missing for {len(missing_mal)} file(s). Running mapping_tifFolder.py ...")
        run_helper_script("mapping_tifFolder.py", [
            data_folder,  # Positional argument
            "--map", map_path,
            "--bpp", str(bpp),
        ])


def process_tif(data_file, bpp, shift_window):
    pcc_df = read_pcc_results(data_file)
    mal_df = read_maligner_results(data_file)

    if pcc_df.empty and mal_df.empty:
        print("  ⚠  Skipping: no log data available.")
        return
    
    # Debug: print counts
    if not pcc_df.empty:
        print(f"  ✓ Found {len(pcc_df)} PCC result(s)")
    if not mal_df.empty:
        print(f"  ✓ Found {len(mal_df)} maligner result(s)")

    # Get molecule length from .info file or calculate from TIFF
    info_path = get_info_path(data_file)
    info_data = parse_info_file(info_path)
    mol_length_bp = None
    if info_data and "mol_length_bp" in info_data:
        try:
            mol_length_bp = float(info_data["mol_length_bp"])
        except (TypeError, ValueError):
            pass

    if mol_length_bp is None or mol_length_bp <= 0:
        try:
            _, w2, _ = lib.tif2RGimage(data_file)
            mol_length_bp = w2 * bpp
        except Exception as exc:
            print(f"  ⚠  Failed to read TIFF data: {exc}")
            return
    
    # Combine and filter results
    combined_df = pd.concat([pcc_df, mal_df], ignore_index=True)
    combined = filter_results(combined_df, bpp, mol_length_bp, shift_window)
    if combined.empty:
        print("  ⚠  No matching results after filtering.")
        return
    
    # Debug: print filtered counts
    pcc_count = len(combined[combined['method'] == 'pcc'])
    mal_count = len(combined[combined['method'] == 'maligner'])
    if pcc_count > 0:
        print(f"  ✓ After filtering: {pcc_count} PCC result(s)")
    if mal_count > 0:
        print(f"  ✓ After filtering: {mal_count} maligner result(s)")
    elif not mal_df.empty:
        print(f"  ⚠  Warning: {len(mal_df)} maligner result(s) were filtered out")

    # Prepare output DataFrame
    df_display = combined[['method','shift','direction','scale','cc_r','cc_g','cc_rg','cc_rg2','rank','misc']].copy()
    df_display['start_bp'] = df_display['shift'] * bpp
    df_display = df_display.drop(columns=['shift'])
    df_display['rank'] = df_display['rank'] + 1  # Convert 0-based to 1-based rank
    
    out_file = data_file.replace(".tif", "_matching_results.csv")
    df_display[OUTPUT_COLUMNS].to_csv(out_file, index=False)
    print(f"  ✓ Saved matching results to {os.path.basename(out_file)}")

def _get_tif_files(data_folder):
    """Get sorted list of TIFF files from folder."""
    tif_files = []
    for ext in ("*.tif", "*.tiff"):
        tif_files.extend(glob.glob(os.path.join(data_folder, ext)))
    return sorted(tif_files)


def _update_bpp_from_map_info(map_path, bpp):
    """Update BPP from map info file if available."""
    map_info = ensure_single_info(map_path, bpp)
    if map_info and "bpp" in map_info:
        try:
            map_bpp = int(float(map_info["bpp"]))
            if map_bpp != bpp:
                print(f"Using BPP from map info ({map_bpp}) instead of CLI value {bpp}.")
                return map_bpp
        except (TypeError, ValueError):
            pass
    return bpp


def main():
    """Main entry point."""
    args = parse_args()
    map_path, data_targets, is_folder = resolve_paths(args)

    # Update BPP from map info if available
    bpp = _update_bpp_from_map_info(map_path, args.bpp)

    if is_folder:
        data_folder = data_targets[0]
        print(f"Map: {map_path}")
        print(f"Data folder: {data_folder}\n")

        tif_files = _get_tif_files(data_folder)
        if not tif_files:
            print("No TIFF files found.")
            return

        print(f"Found {len(tif_files)} TIFF file(s).")
        ensure_log_files(map_path, data_folder, tif_files, bpp, args.start_bpp, args.end_bpp)
        ensure_info_files(tif_files, data_folder, bpp)
        print("Generating matching result CSVs...\n")

        for idx, tif_file in enumerate(tif_files, 1):
            print(f"[{idx}/{len(tif_files)}] {os.path.basename(tif_file)}")
            process_tif(tif_file, bpp, args.shift_window)
            print()
    else:
        tif_file = data_targets[0]
        print(f"Map: {map_path}")
        print(f"File: {tif_file}\n")

        single_folder = os.path.dirname(tif_file) or "."
        ensure_log_files(map_path, single_folder, [tif_file], bpp, args.start_bpp, args.end_bpp)
        ensure_single_info(tif_file, bpp)
        print("Generating matching result CSV...\n")
        process_tif(tif_file, bpp, args.shift_window)

    print("Done.")


if __name__ == '__main__':
    main()
