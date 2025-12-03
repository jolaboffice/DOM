#!/usr/bin/env python
"""
DOM.py

Main execution script for DOM matching tool with GUI.
"""

import os,sys
import subprocess
import time
import numpy as np
import pandas as pd
import DOM_lib as lib
import DOM_ui
import DOM_init
import DOM_match

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

runtime_config = DOM_init.init_config()

map_path = runtime_config['map_path']
tif_data_folder = runtime_config['tif_data_folder']
BPP = runtime_config['BPP']
SHIFT_WINDOW = runtime_config['SHIFT_WINDOW']
START_BPP = runtime_config['START_BPP']
END_BPP = runtime_config['END_BPP']



control_state = {'action': None}
file_list_state = {'tif_files': [], 'current_index': 0, 'ax_list': None, 'clicked_index': None, 'file_rows': [], 
                   'ax_table': None, 'clicked_row_idx': None, 'table_rows': [], 'current_table_row_idx': None,
                   'view_start_idx': None, 'view_end_idx': None,
                   'ax_header': None, 'map_path': None, 'data_folder_path': None, 'need_restart': False,
                   'updating_scrollbar': False}
total_files = 0

def _read_matching_csv(data_file):
    csv_path = data_file.replace(".tif", "_matching_results.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Warning: Failed to read precomputed matching results {csv_path}: {exc}")
        return None
    required = {'method','shift_bp','scale','cc_rg','cc_r','cc_g','cc_rg2','rank','misc'}
    if not required.issubset(df.columns):
        return None
    if 'direction' not in df.columns:
        df['direction'] = 1
    matching_methods = []
    for _, row in df.iterrows():
        method = str(row['method'])
        shift_bp = row['shift_bp']
        shift = shift_bp / BPP
        direction = int(row['direction']) if 'direction' in row and not pd.isna(row['direction']) else 1
        scale = row['scale']
        misc = row['misc'] if pd.notna(row['misc']) else 0.0
        g_pos = row['g_pos'] if 'g_pos' in row else 0.0
        r_pos = row['r_pos'] if 'r_pos' in row else 0.0
        cc_rg = row['cc_rg']
        cc_r = row['cc_r']
        cc_g = row['cc_g']
        cc_rg2 = row['cc_rg2']
        rank = row['rank'] if pd.notna(row['rank']) else 0
        matching_methods.append([method, shift, direction, scale, misc, g_pos, r_pos, cc_rg, cc_r, cc_g, cc_rg2, rank])
    return matching_methods

def _run_dom_match(data_target, is_file=True):
    """Run DOM_match.py for a file or folder."""
    dom_match_path = os.path.join(SCRIPT_DIR, "DOM_match.py")
    if not os.path.exists(dom_match_path):
        print(f"Warning: DOM_match.py not found; cannot generate matching results.")
        return False
    try:
        cmd = [sys.executable, dom_match_path, data_target,
               "--map", map_path,
               "--bpp", str(BPP),
               "--start-bpp", str(START_BPP),
               "--end-bpp", str(END_BPP)]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Warning: DOM_match.py failed ({exc}).")
        return False

def _ensure_matching_csv(data_file):
    csv_path = data_file.replace(".tif", "_matching_results.csv")
    if os.path.exists(csv_path):
        return True
    print(f"Precomputed matching results not found. Running DOM_match.py for {data_file} ...")
    return _run_dom_match(data_file, is_file=True)

def load_matching_results(data_file):
    """Load matching results, generating CSV if needed."""
    matching_methods = _read_matching_csv(data_file)
    if matching_methods:
        return matching_methods
    if _ensure_matching_csv(data_file):
        return _read_matching_csv(data_file) or []
    return []


def _get_molecule_dimensions(data_file):
    _, w2, _ = lib.tif2RGimage(data_file)
    return w2, w2 * BPP

def _prepare_matches(mat_df):
    """Prepare matches from CSV (already filtered by DOM_match.py, no re-filtering needed)."""
    df = mat_df.copy()
    # Ensure all required columns exist
    if 'cc_rg2' not in df.columns:
        df['cc_rg2'] = df.get('cc_rg', 0) * (df.get('cc_g', 0) ** 2) if 'cc_r' in df.columns and 'cc_g' in df.columns else df.get('misc', 0)
    # Ensure g_pos and r_pos exist
    if 'g_pos' not in df.columns:
        df['g_pos'] = 0.0
    if 'r_pos' not in df.columns:
        df['r_pos'] = 0.0
    return df



def _load_profile_data(RGmap, data_file):
    RGmap_image,w1,h1=lib.tif2RGimage(RGmap)
    data_image,w2,h2=lib.tif2RGimage(data_file)
    map_r,map_g,_=lib.tif2RGprofile(RGmap)
    data_r,data_g,_=lib.tif2RGprofile(data_file)
    clip1r=lib.normalize(map_r)
    clip1g=lib.normalize(map_g)
    clip2r=lib.normalize(data_r)
    clip2g=lib.normalize(data_g)
    clip1r=np.concatenate((clip1r,clip1r[0:w2]), axis=0)
    clip1g=np.concatenate((clip1g,clip1g[0:w2]), axis=0)
    return {
        'RGmap_image': RGmap_image,
        'data_image': data_image,
        'w1': w1,
        'h1': h1,
        'w2': w2,
        'clip1r': clip1r,
        'clip1g': clip1g,
        'clip2r': clip2r,
        'clip2g': clip2g
    }




def _prepare_alignment_arrays(row, w2, clip1r, clip1g, clip2r, clip2g):
    shift_px = int(row['shift'])
    scale = row['scale']
    direction = int(row['direction'])
    image_width = int(w2 * scale)
    x2 = np.arange(shift_px, shift_px + w2) * BPP / 1e6
    d_clip2g = clip2g[::direction]
    d_clip2r = clip2r[::direction]
    s_clip1g = lib.normalize(clip1g[shift_px:shift_px + image_width])
    s_clip1r = lib.normalize(clip1r[shift_px:shift_px + image_width])
    return {
        'shift_px': shift_px,
        'scale': scale,
        'direction': direction,
        'image_width': image_width,
        'x2': x2,
        'd_clip2g': d_clip2g,
        'd_clip2r': d_clip2r,
        's_clip1g': s_clip1g,
        's_clip1r': s_clip1r
    }



def match_display(mat_df, RGmap, data_file):
    DOM_ui.update_current_file_list_display(data_file)
    w2, mol_length_bp = _get_molecule_dimensions(data_file)

    df = _prepare_matches(mat_df)
    df_display = DOM_ui.build_display_table(df, BPP)

    DOM_ui.render_results_panel(df_display, current_row_idx=None)
    print(df_display[['rank','method','shift_bp','scale','cc_rg','cc_r','cc_g','cc_rg2','misc']])
    print('\n')
    profiles = _load_profile_data(RGmap, data_file)
    DOM_ui.clear_alignment_lines()
    df_indices = df.index.tolist()
    i = 0
    clicked_idx = DOM_ui.find_clicked_row_index(df)
    if clicked_idx is not None:
        i = df_indices.index(clicked_idx)
    
    while i < len(df_indices):
        idx = df_indices[i]
        row = df.loc[idx]
        method = row['method']
        shift_px = int(row['shift'])
        shift_bp = shift_px * BPP
        direction = int(row['direction'])
        scale = row['scale']
        misc = row['misc']
        g_pos = row['g_pos']
        r_pos = row['r_pos']
        cc_rg = row['cc_rg']
        cc_r = row['cc_r']
        cc_g = row['cc_g']

        DOM_ui.update_table_highlight(df_display, method, shift_bp, scale)
        alignment = _prepare_alignment_arrays(row, profiles['w2'], profiles['clip1r'], profiles['clip1g'],
                                              profiles['clip2r'], profiles['clip2g'])
        DOM_ui.plot_alignment(
            alignment['x2'],
            alignment['d_clip2g'],
            alignment['d_clip2r'],
            alignment['direction'],
            profiles['data_image'],
            alignment['shift_px'],
            alignment['image_width'],
            profiles['h1'],
            profiles['RGmap_image'],
            alignment['s_clip1r'],
            alignment['s_clip1g'],
            alignment['d_clip2r'],
            alignment['d_clip2g'],
            profiles['w2'],
            alignment['scale']
        )

        result = DOM_ui.control_RGmap_viewer(
            alignment['shift_px'],
            profiles['w2'],
            direction,
            data_file,
            method,
            scale,
            misc,
            alignment['x2'],
            r_pos,
            g_pos,
            alignment['d_clip2r'],
            alignment['d_clip2g'],
            cc_rg,
            cc_r,
            cc_g,
            shift_bp=shift_bp
        )

        if result == 'prev':
            if i > 0:
                i -= 1
            continue
        if result == 'exit':
            return 'exit'
        if result == 'file_clicked':
            return 'file_clicked'
        if result == 'row_clicked':
            i, _ = DOM_ui.maybe_jump_to_clicked_row(df, df_indices, i)
            continue
        i += 1
    
    # Return None if all items processed normally
    return None

def Compare_Ref_and_Data(RGmap, data_file):
    matching_methods = load_matching_results(data_file)
    if not matching_methods:
        print(f"No matching results available for {data_file}.")
        return 'continue'
    col=['method','shift','direction','scale','misc','g_pos','r_pos','cc_rg','cc_r','cc_g','cc_rg2','rank']
    mat_df=pd.DataFrame(matching_methods, columns=col)
    result = match_display(mat_df, RGmap, data_file)
    if result == 'file_clicked':
        return 'file_clicked'
    return result if result else 'continue'

def load_reference_and_data():
    """Load RG map and discover TIFF files."""
    global RGmap, tif_files, total_files
    try:
        RGmap = lib.validate_tif_path(map_path)
    except (ValueError, FileNotFoundError):
        # Error message already printed by validate_tif_path
        sys.exit(1)
    
    tif_files = lib.Read_TIFF_folder(tif_data_folder)
    tif_files = sorted(tif_files, key=lambda x: os.path.basename(x).lower())
    total_files = len(tif_files)
    print(f"Found {total_files} TIFF file(s).\n")
    ensure_matching_csvs()

def ensure_matching_csvs():
    """Run DOM_match.py once if any matching_result CSV is missing."""
    if not tif_files:
        return
    
    missing_count = sum(1 for tif_file in tif_files 
                       if not os.path.exists(tif_file.replace(".tif", "_matching_results.csv")))
    
    if missing_count == 0:
        return
    
    print(f"\n{'='*60}")
    print(f"Missing matching results for {missing_count}/{len(tif_files)} file(s).")
    print(f"This may take a while if log files (_pcc.log, _mal.log) are also missing.")
    print(f"Running DOM_match.py for entire folder...")
    print(f"{'='*60}\n")
    
    if _run_dom_match(tif_data_folder, is_file=False):
        print(f"\n{'='*60}")
        print(f"DOM_match.py completed. Starting GUI...")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"Warning: DOM_match.py failed. Some matching results may still be missing.")
        print(f"{'='*60}\n")


def setup_main_figure():
    """Initialize the matplotlib layout, event handlers, and global panes."""
    return DOM_ui.setup_main_figure(tif_files, map_path, tif_data_folder, RGmap, BPP, file_list_state, control_state)

def run_main_loop(fig):
    """Iterate through files once, then keep the window interactive."""
    def process_file_callback(tif_file):
        return Compare_Ref_and_Data(RGmap, tif_file)
    DOM_ui.run_main_loop(fig, tif_files, process_file_callback)

def main():
    if tif_data_folder is None:
        print("Usage: DOM.py <data_folder> [--map MAP_FILE] [--bpp BPP]")
        print("  Default map: ref_map/RG.tif")
        sys.exit(1)
    load_reference_and_data()
    fig = setup_main_figure()
    run_main_loop(fig)

if __name__ == '__main__':
    main()