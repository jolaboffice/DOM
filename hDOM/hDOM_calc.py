#!/usr/bin/env python
"""
hDOM_calc.py
Calculate matching results CSV files using FFT-accelerated PCC on GPU.
Reads reference and data files, then generates matching results CSV.
Calls hDOM_mal.py and hDOM_pcc_ft.py as external commands.
"""

import os
import sys
import argparse
import subprocess
import concurrent.futures
import time
from datetime import datetime
import threading
import pandas as pd
import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import signal, stats
import math
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

DEFAULTS = {
    'BPP': 200,
}

# mal multiprocessing (CPU-bound)
USE_MALIGNER_MULTIPROC = True
MALIGNER_WORKERS = None  # None -> auto (half of CPU cores)
# PCC script name to call
PCC_SCRIPT_NAME = "hDOM_pcc_ft.py"
# Matching results multiprocessing (CPU-bound)
USE_MATCH_MULTIPROC = True
MATCH_WORKERS = None  # None -> auto (half of CPU cores)

def corrcoef(x, y):
    if HAS_GPU:
        # GPU calculation using CuPy (CUDA)
        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)
        x_max = x_gpu.max()
        x_min = x_gpu.min()
        x_gap = x_max - x_min
        if x_gap == 0:
            x_gap = 1
        x_norm = (x_gpu - x_min) / x_gap
        
        y_max = y_gpu.max()
        y_min = y_gpu.min()
        y_gap = y_max - y_min
        if y_gap == 0:
            y_gap = 1
        y_norm = (y_gpu - y_min) / y_gap
        
        n = len(y_norm)
        xy = cp.dot(x_norm, y_norm)
        xx = cp.dot(x_norm, x_norm)
        yy = cp.dot(y_norm, y_norm)
        s_x = cp.sum(x_norm)
        s_y = cp.sum(y_norm)
        
        denominator = cp.sqrt((n * xx - s_x * s_x) * (n * yy - s_y * s_y))
        if denominator == 0:
            return 0.0
        cc = (n * xy - s_x * s_y) / denominator
        return float(cc)
    else:
        # Fallback to numpy if GPU not available
        x_max=np.max(x)
        x_min=np.min(x)
        x_gap=x_max-x_min
        if (x_gap==0):
            x_gap=1
        x=(x-x_min)/x_gap
        y_max=np.max(y)
        y_min=np.min(y)
        y_gap=y_max-y_min
        if (y_gap==0):
            y_gap=1
        y=(y-y_min)/y_gap
        
        n=len(y)
        xy=np.dot(x,y)
        xx=np.dot(x,x)
        yy=np.dot(y,y)
        s_x=np.sum(x)
        s_y=np.sum(y)
        denominator = math.sqrt((n*xx-s_x*s_x)*(n*yy-s_y*s_y))
        if denominator == 0:
            return 0
        cc=(n*xy-s_x*s_y)/denominator
        return cc 

def dcc(x, y):   #cross correlation without mean subtration suggested by Ambjorsson
    if HAS_GPU:
        # GPU calculation using CuPy (CUDA)
        x_gpu = cp.asarray(np.ascontiguousarray(x))
        y_gpu = cp.asarray(np.ascontiguousarray(y))
        x_dot = cp.dot(x_gpu, x_gpu)
        y_dot = cp.dot(y_gpu, y_gpu)
        denominator = cp.sqrt(x_dot) * cp.sqrt(y_dot)
        if denominator == 0:
            return 0.0
        cc = cp.dot(x_gpu, y_gpu) / denominator
        return float(cc)
    else:
        # Fallback to numpy
        x_contig = np.ascontiguousarray(x)
        y_contig = np.ascontiguousarray(y)
        x_dot = np.dot(x_contig,x_contig)
        y_dot = np.dot(y_contig,y_contig)
        denominator = math.sqrt(x_dot) * math.sqrt(y_dot)
        if denominator == 0:
            return 0.0
        cc=np.dot(x_contig,y_contig)/denominator
        return cc

def normalize(ar):
    if (len(ar)==0):
        return ar
    min_val=ar.min()
    max_val=ar.max()
    if (max_val==min_val):
        max_val=max_val+1
    rt=(ar-min_val)/(max_val-min_val)
    return rt

def tif2RGimage(file):
    image=Image.open(file)
    w,h=image.size
    if (image.mode=='I;16B'):
        data=tiff.imread(file)
        r=data[0].flatten()
        r=r-r.min()
        r=normalize(r)
        r=r*280
        g=data[1].flatten()
        g=g-g.min()
        g=normalize(g)
        thres=g.mean()+(g.max()-g.mean())*0.1
        g[g<thres]=0 
        g=g*280       
        b=np.zeros(h*w)
        r[r>255]=255
        g[g>255]=255
        pixel=np.column_stack((r,g,b)).astype('uint8')
        image=Image.frombuffer('RGB', (w, h), pixel, 'raw', 'RGB', 0, 0)
    return image,w,h    

def tif2RGprofile(file):
    img=Image.open(file)
    if (img.mode=='I;16B'): #Experimental Tiff files
        data=tiff.imread(file)
        r=np.max(data[0], axis=0)
        g=np.max(data[1], axis=0)
        thres=g.mean()+(g.max()-g.mean())*0.1
        g[g<thres]=0
    elif (img.mode=='RGB'): #Image from FASTA file
        w,h=img.size
        color=img.split()
        r_data=np.array(color[0].getdata())
        r_data=r_data.reshape(h,w)
        r=np.max(r_data,axis=0)
        g_data=np.array(color[1].getdata())
        g_data=g_data.reshape(h,w)
        g=np.max(g_data,axis=0)
        thres=g.mean()+(g.max()-g.mean())*0.1
        g[g<thres]=0
    peaks,_= signal.find_peaks(g)
    return r,g, peaks

def Read_TIFF_folder(path):
    if (path[-1]!='/'):
        path += "/"
    folder=sorted(os.listdir(path))
    folder.sort(key=lambda f: os.stat(path+f).st_size, reverse=True)
    tif_files=[]
    for filename in folder:
        if(filename[-4:]=='.tif'):
            tif_files.append(path+filename)
    return tif_files

def parse_args():
    """Parse command-line arguments for hDOM_calc.py"""
    parser = argparse.ArgumentParser(
        description='hDOM_calc: Calculate matching results CSV files'
    )
    parser.add_argument('data_folder', nargs='?', default=None,
                       help='Folder containing tif data files. Positional argument or use --data-folder.')
    parser.add_argument('--data-folder', dest='data_folder_opt', default=None,
                       help='Folder containing tif data files (alternative to positional argument)')
    parser.add_argument('--ref-map', dest='ref_map_folder', default=None,
                       help='Folder containing reference map files (default: ref_map folder relative to data folder)')
    parser.add_argument('--bpp', type=int, default=DEFAULTS['BPP'],
                       help='Base pairs per pixel (default: 200)')
    parser.add_argument('--recalculate', action='store_true',
                       help='Force recalculation of matching results (ignores existing CSV files)')
    parser.add_argument('--circular', action='store_true',
                       help='Treat reference genomes as circular (default: linear)')
    return parser.parse_args()

def get_data_folder(parsed_args):
    """Get data folder from optional or positional argument"""
    data_folder = parsed_args.data_folder_opt if parsed_args.data_folder_opt is not None else parsed_args.data_folder
    
    if data_folder is None:
        print("Error: Data folder is required.")
        print("Usage: hDOM_calc.py <data_folder> [options]")
        sys.exit(1)
    
    if not os.path.isdir(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        sys.exit(1)
    
    return os.path.abspath(data_folder)

def get_reference_files(ref_map_folder):
    """Get sorted list of reference TIF files from folder."""
    if not os.path.isdir(ref_map_folder):
        return []
    import glob
    reference_files = []
    for ext in ("*.tif", "*.tiff"):
        reference_files.extend(glob.glob(os.path.join(ref_map_folder, ext)))
    return sorted([f for f in reference_files if f.endswith((".tif", ".tiff"))], 
                  key=lambda x: os.path.basename(x).lower())

def get_ref_map_folder(data_folder, user_specified=None):
    """Get reference map folder path"""
    if user_specified:
        ref_map_folder = os.path.abspath(user_specified)
        if not os.path.isdir(ref_map_folder):
            print(f"Warning: Specified ref_map folder not found: {ref_map_folder}")
            return None
        return ref_map_folder
    
    # Default: check inside data folder, then sibling of data folder
    data_abs = os.path.abspath(data_folder)
    parent_dir = os.path.dirname(data_abs)
    for candidate in [
        os.path.join(data_abs, "ref_map"),
        os.path.join(parent_dir, "ref_map"),
    ]:
        if os.path.isdir(candidate):
            return candidate

    print(f"Warning: ref_map folder not found in {data_abs} or {parent_dir}")
    return None

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

def read_pcc_results(data_file, ref_name=None):
    """Read PCC results. If ref_name is provided, look for reference-specific log file."""
    if ref_name:
        data_name = os.path.splitext(os.path.basename(data_file))[0]
        data_dir = os.path.dirname(data_file)
        log_file = os.path.join(data_dir, f"{data_name}_{ref_name}_pcc.log")
        # Fallback to generic log if reference-specific doesn't exist
        if not os.path.exists(log_file):
            log_file = data_file.replace(".tif", "_pcc.log")
    else:
        log_file = data_file.replace(".tif", "_pcc.log")
    
    if not os.path.exists(log_file):
        return pd.DataFrame()

    try:
        df = pd.read_csv(log_file)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    df['method'] = 'pcc'
    df['g_pos'] = 0.0
    df['r_pos'] = 0.0
    # Calculate cc_rg from cc_r and cc_g (not from score)
    # If cc_g is 0.0, change it to 0.001 to avoid zero division
    if 'cc_g' in df.columns:
        df.loc[df['cc_g'] == 0.0, 'cc_g'] = 0.001
    if 'cc_r' in df.columns and 'cc_g' in df.columns:
        df['cc_rg'] = df['cc_r'] * df['cc_g']
        df['cc_rg2'] = df['cc_r'] * (df['cc_g'] ** 2)
    else:
        df['cc_rg'] = df['score']  # Fallback to score if cc_r or cc_g missing
        df['cc_rg2'] = df['score']
    df['rank'] = df.index
    return df[['method','shift','direction','scale','score','g_pos','r_pos','cc_rg','cc_r','cc_g','rank', 'cc_rg2']]

def filter_results(df, bpp, mol_length_bp):
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    if 'cc_rg2' not in df.columns:
        df['cc_rg2'] = df['cc_r'] * (df['cc_g'] ** 2)
    
    df['shift_bp'] = df['shift'] * bpp
    df['start_bp'] = df['shift_bp']
    df['end_bp'] = df['shift_bp'] + (mol_length_bp * df['scale'])
    df = df.sort_values(by='cc_rg2', ascending=False).reset_index(drop=True)

    pcc_df = df[df['method'] == 'pcc'].copy()

    pcc_groups = []
    for _, row in pcc_df.iterrows():
        shift_bp = row['shift_bp']
        matched = False
        for grp in pcc_groups:
            if abs(shift_bp - grp['shift']) <= 5000:
                grp['count'] += 1
                current_rank = row['rank'] if pd.notna(row['rank']) else np.inf
                if current_rank < grp['best_rank']:
                    grp['best_rank'] = current_rank
                    grp['record'] = row.copy()
                matched = True
                break
        if matched:
            continue
        if len(pcc_groups) < 10:
            grp_rank = row['rank'] if pd.notna(row['rank']) else np.inf
            pcc_groups.append({
                'shift': shift_bp,
                'count': 1,
                'best_rank': grp_rank,
                'record': row.copy()
            })
    if pcc_groups:
        pcc_filtered = pd.DataFrame([grp['record'] for grp in pcc_groups])
        pcc_filtered['score'] = [grp['count'] for grp in pcc_groups]
    else:
        pcc_filtered = pd.DataFrame(columns=pcc_df.columns)

    combined = pcc_filtered
    combined = combined.sort_values(by='cc_rg2', ascending=False).reset_index(drop=True)
    return combined

def build_matching_df_for_tif_local(data_file, bpp, map_path=None):
    """Build matching results DataFrame for a single TIF file."""
    # Extract ref_name from map_path
    ref_name = None
    if map_path and os.path.exists(map_path):
        ref_name = os.path.splitext(os.path.basename(map_path))[0]
    
    pcc_df = read_pcc_results(data_file, ref_name=ref_name)

    if pcc_df.empty:
        return pd.DataFrame()
    
    info_path = get_info_path(data_file)
    info_data = parse_info_file(info_path)
    if info_data and "mol_length_bp" in info_data:
        try:
            mol_length_bp = float(info_data["mol_length_bp"])
        except (TypeError, ValueError):
            mol_length_bp = None
    else:
        mol_length_bp = None

    if mol_length_bp is None or mol_length_bp <= 0:
        try:
            _, w2, _ = tif2RGimage(data_file)
            mol_length_bp = w2 * bpp
        except Exception:
            return pd.DataFrame()

    combined = filter_results(pcc_df, bpp, mol_length_bp)
    if combined.empty:
        return pd.DataFrame()

    df_display = combined[['method','shift','direction','scale','cc_rg','cc_r','cc_g','cc_rg2','rank','score']].copy()
    
    # p_val will be calculated globally in ensure_matching_csvs
    df_display['p_val'] = np.nan

    df_display['shift_bp'] = df_display['shift'] * bpp
    df_display = df_display.drop(columns=['shift']).rename(columns={'shift_bp':'shift_bp'})
    if 'rank' in df_display.columns:
        df_display['rank'] = df_display['rank'].apply(lambda x: x + 1 if pd.notna(x) else x)
    
    # Add ref_name from map_path (extract filename without .tif extension)
    if map_path and os.path.exists(map_path):
        ref_name = os.path.splitext(os.path.basename(map_path))[0]
    else:
        ref_name = 'unknown'
    df_display['ref_name'] = ref_name

    # Rename score to misc if it exists
    if 'score' in df_display.columns and 'misc' not in df_display.columns:
        df_display = df_display.rename(columns={'score': 'misc'})

    return df_display

def load_maligner_dp_results(data_file, bpp):
    """Load raw maligner_dp results from the saved .txt file and convert to PCC format."""
    txt_path = data_file.replace(".tif", "_maligner_dp.txt")
    if not os.path.exists(txt_path):
        return pd.DataFrame()
    
    try:
        # Try reading as tab-separated with header
        df = pd.read_csv(txt_path, sep='\t', header=0)
        if df.empty:
            return pd.DataFrame()
    except Exception as e:
        # Try reading without header (maligner_dp might not have header)
        try:
            # Read first line to check format
            with open(txt_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return pd.DataFrame()
                # Check if it's tab-separated
                fields = first_line.split('\t')
                if len(fields) < 5:
                    # Try space-separated
                    fields = first_line.split()
                if len(fields) >= 5:
                    # Read as tab-separated without header
                    df = pd.read_csv(txt_path, sep='\t', header=None)
                    # Use expected column names based on maligner_dp output format
                    if len(df.columns) >= 42:  # maligner_dp has many columns
                        df.columns = [
                            'query_map', 'ref_map', 'is_forward', 'query_start', 'query_end',
                            'ref_start', 'ref_end', 'query_start_bp', 'query_end_bp',
                            'ref_start_bp', 'ref_end_bp', 'num_matched_chunks', 'query_misses',
                            'ref_misses', 'query_miss_rate', 'ref_miss_rate', 'total_score',
                            'total_rescaled_score', 'm_score', 'p_val', 'sizing_score',
                            'sizing_score_rescaled', 'query_scaling_factor', 'num_interior_chunks',
                            'score_per_inner_chunk', 'chunk_string', 'score_string'
                        ] + [f'col_{i}' for i in range(26, len(df.columns))]
                    elif len(df.columns) >= 5:
                        # Minimal columns: use positional
                        df.columns = ['query_map', 'ref_map', 'is_forward', 'query_start', 'query_end'] + \
                                    [f'col_{i}' for i in range(5, len(df.columns))]
                else:
                    return pd.DataFrame()
        except Exception as e2:
            print(f"Warning: Failed to parse maligner_dp results from {txt_path}: {e}, {e2}", file=sys.stderr)
            return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert maligner_dp results to PCC format
    mal_results = []
    for idx, row in df.iterrows():
        # Extract ref_name from ref_map (remove .maps extension if present)
        ref_map = str(row.get('ref_map', ''))
        ref_name = os.path.splitext(os.path.basename(ref_map))[0] if ref_map else 'unknown'
        
        # Get values with defaults
        ref_start_bp = row.get('ref_start_bp', 0)
        if pd.isna(ref_start_bp):
            ref_start_bp = row.get('ref_start', 0) * bpp if 'ref_start' in row else 0
        
        query_scaling_factor = row.get('query_scaling_factor', 1.0)
        if pd.isna(query_scaling_factor):
            query_scaling_factor = 1.0
        
        total_rescaled_score = row.get('total_rescaled_score', 0.0)
        if pd.isna(total_rescaled_score):
            total_rescaled_score = row.get('total_score', 0.0) if 'total_score' in row else 0.0
        
        # Determine direction from is_forward
        is_forward = str(row.get('is_forward', 'F')).upper()
        direction = 1 if is_forward == 'F' else -1
        
        # Convert to shift (in pixels) - ref_start_bp / bpp
        shift_bp = float(ref_start_bp) if not pd.isna(ref_start_bp) else 0.0
        shift = shift_bp / bpp if bpp else 0.0
        
        # Create result in PCC format
        mal_results.append({
            'method': 'mal',
            'shift': shift,
            'shift_bp': shift_bp,
            'direction': direction,
            'scale': float(query_scaling_factor) if not pd.isna(query_scaling_factor) else 1.0,
            'score': float(total_rescaled_score) if not pd.isna(total_rescaled_score) else 0.0,
            'misc': float(total_rescaled_score) if not pd.isna(total_rescaled_score) else 0.0,
            'log(P)': np.nan,
            'g_pos': np.nan,
            'r_pos': np.nan,
            'cc_rg': np.nan,  # mal doesn't have correlation values
            'cc_r': np.nan,
            'cc_g': np.nan,
            'cc_rg2': np.nan,
            'rank': idx,  # Use original index as rank
            'ref_name': ref_name,
            'reference_path': None  # We don't have the reference path
        })
    
    if not mal_results:
        return pd.DataFrame()
    
    return pd.DataFrame(mal_results)

def _run_maligner_for_file(args):
    """Helper for multiprocessing maligner runs."""
    tif_file, mal_script = args
    try:
        subprocess.run(
            [sys.executable, mal_script, tif_file],
            check=False,
            capture_output=False
        )
    except subprocess.CalledProcessError as exc:
        print(f"      Warning: hDOM_mal.py failed for {os.path.basename(tif_file)} "
              f"(exit code {exc.returncode}).")

def _build_matching_df(args):
    """Helper for multiprocessing matching results."""
    tif_file, bpp, ref_file = args
    ref_name = os.path.splitext(os.path.basename(ref_file))[0]
    try:
        # 1. Get ALL raw results for statistics
        df_raw = read_pcc_results(tif_file, ref_name=ref_name)
        raw_cc_rg2 = df_raw['cc_rg2'].tolist() if not df_raw.empty else []
        
        # 2. Get filtered top results for display
        df_ref = build_matching_df_for_tif_local(tif_file, bpp, map_path=ref_file)
        
        return ref_name, df_ref, raw_cc_rg2, None
    except Exception as exc:
        return ref_name, None, [], exc

def wait_for_maligner_txt(data_file, timeout_s=3600, poll_s=5):
    """Wait for maligner_dp output txt to appear (and be non-empty)."""
    txt_path = data_file.replace(".tif", "_maligner_dp.txt")
    start_time = time.time()
    while True:
        if os.path.exists(txt_path):
            try:
                if os.path.getsize(txt_path) > 0:
                    return True
            except OSError:
                pass
        if timeout_s is not None and (time.time() - start_time) >= timeout_s:
            return False
        time.sleep(poll_s)

def calculate_cc_for_maligner(data_file, ref_file, shift_bp, direction, scale, bpp, circular=False):
    """Calculate cc_r and cc_g for maligner result using shift_bp, direction, and scale."""
    try:
        # Load reference profiles
        map_r, map_g, _ = tif2RGprofile(ref_file)

        # Load data profiles
        data_r, data_g, _ = tif2RGprofile(data_file)

        # Normalize profiles
        clip1r = normalize(map_r)
        clip1g = normalize(map_g)
        clip2r = normalize(data_r)
        clip2g = normalize(data_g)

        # Circular genome: concatenate reference with itself to handle boundary-spanning shifts
        if circular:
            clip1r = np.concatenate([clip1r, clip1r])
            clip1g = np.concatenate([clip1g, clip1g])

        # Convert shift_bp to pixels
        shift_px = int(shift_bp / bpp) if bpp > 0 else 0

        # Apply direction to data (reverse if direction == -1)
        d_clip2r = clip2r[::direction]
        d_clip2g = clip2g[::direction]

        # Apply scale: resample data to match scaled length
        w2 = len(d_clip2r)
        image_width = int(w2 * scale)

        # Get reference window starting at shift_px
        if shift_px < 0 or shift_px + image_width > len(clip1r):
            return np.nan, np.nan
        
        s_clip1r = normalize(clip1r[shift_px:shift_px + image_width])
        s_clip1g = normalize(clip1g[shift_px:shift_px + image_width])
        
        # Resample data to match reference window length
        if scale != 1.0:
            # Interpolate data to match scaled length
            x_old = np.linspace(0, len(d_clip2r) - 1, len(d_clip2r))
            x_new = np.linspace(0, len(d_clip2r) - 1, image_width)
            d_clip2r = np.interp(x_new, x_old, d_clip2r).astype(np.float32)
            d_clip2g = np.interp(x_new, x_old, d_clip2g).astype(np.float32)
            d_clip2r = normalize(d_clip2r)
            d_clip2g = normalize(d_clip2g)
        
        # Ensure same length
        min_len = min(len(s_clip1r), len(d_clip2r))
        if min_len == 0:
            return np.nan, np.nan
        
        s_clip1r = s_clip1r[:min_len]
        s_clip1g = s_clip1g[:min_len]
        d_clip2r = d_clip2r[:min_len]
        d_clip2g = d_clip2g[:min_len]
        
        # Convert to float32
        s_clip1r = s_clip1r.astype(np.float32)
        s_clip1g = s_clip1g.astype(np.float32)
        d_clip2r = d_clip2r.astype(np.float32)
        d_clip2g = d_clip2g.astype(np.float32)
        
        # Calculate correlations
        cc_r = corrcoef(s_clip1r, d_clip2r)
        cc_g = dcc(s_clip1g, d_clip2g)
        
        return float(cc_r), float(cc_g)
    except Exception as e:
        print(f"    Warning: Failed to calculate cc_r/cc_g for maligner result: {e}", file=sys.stderr)
        return np.nan, np.nan

def ensure_matching_csvs(tif_files, tif_data_folder, reference_files, bpp, start_time=None, circular=False):
    """Run PCC and mal calculations, then combine all results into single CSV per data file."""
    if not tif_files or not reference_files:
        return None
    
    print("  - Starting matching results calculation...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mal_script = os.path.join(script_dir, "hDOM_mal.py")
    pcc_script = os.path.join(script_dir, PCC_SCRIPT_NAME)
    
    mal_end_time = None
    mal_thread = None
    # Step 0: Run hDOM_mal.py (maligner_dp) for ALL data files (optional diagnostic), in parallel with PCC
    if os.path.exists(mal_script):
        print(f"  - Step 0: Running hDOM_mal.py (maligner_dp) for all data files (in parallel)...")

        def _run_maligner_for_all():
            nonlocal mal_end_time
            mal_start_time = datetime.now()
            if USE_MALIGNER_MULTIPROC:
                cpu_count = os.cpu_count() or 1
                workers = MALIGNER_WORKERS or max(1, cpu_count // 2)
                workers = min(workers, len(tif_files))
                print(f"    - Running maligner_dp with {workers} worker(s)...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    list(executor.map(_run_maligner_for_file,
                                      [(tif_file, mal_script) for tif_file in tif_files]))
            else:
                for tif_file in tif_files:
                    print(f"    - Running maligner_dp for data file: {os.path.basename(tif_file)}...")
                    try:
                        subprocess.run(
                            [sys.executable, mal_script, tif_file],
                            check=False,
                            capture_output=False
                        )
                    except subprocess.CalledProcessError as exc:
                        print(f"      Warning: hDOM_mal.py failed for {os.path.basename(tif_file)} "
                              f"(exit code {exc.returncode}).")
            mal_end_time = datetime.now()
            # Print maligner_dp timing relative to overall start if provided
            if start_time is not None:
                elapsed_from_start = mal_end_time - start_time
                print(f"  - maligner_dp finished at {mal_end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"(elapsed from start: {elapsed_from_start})")
            else:
                print(f"  - maligner_dp finished at {mal_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        mal_thread = threading.Thread(target=_run_maligner_for_all, daemon=True)
        mal_thread.start()
    else:
        print("  - Step 0: hDOM_mal.py not found. Skipping maligner_dp step.")
    
    # Step 1: Run hDOM_pcc_ft.py once for all references (pass ref_map folder)
    print(f"  - Step 1: Running {PCC_SCRIPT_NAME} for all references (ref_map folder)...")
    if os.path.exists(pcc_script):
        try:
            pcc_cmd = [sys.executable, pcc_script,
                 "--map", os.path.dirname(reference_files[0]) if reference_files else tif_data_folder,
                 "--data-folder", tif_data_folder,
                 "--bpp", str(bpp)]
            if circular:
                pcc_cmd.append("--circular")
            subprocess.run(pcc_cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as exc:
            print(f"      Warning: {PCC_SCRIPT_NAME} failed for ref_map folder (exit code {exc.returncode}).")
    else:
        print(f"      Warning: {PCC_SCRIPT_NAME} not found.")
    
    # Ensure maligner_dp step is finished before combining results (we need its output files)
    if mal_thread is not None:
        mal_thread.join()
    
    print(f"\n  - Combining results into unified matching_results.csv files (in-memory)...")
    
    # For each data file, combine all reference results into one CSV
    for tif_file in tif_files:
        # Get image width for info.txt
        try:
            _, w2, _ = tif2RGimage(tif_file)
            mol_length_bp = w2 * bpp
        except Exception:
            w2 = 0
            mol_length_bp = 0

        unified_csv = tif_file.replace(".tif", "_matching_results.csv")
        # Always process, even if unified CSV exists
        
        # Build results from all references in memory
        all_results = []
        all_raw_cc_rg2 = []
        missing_refs = []
        if USE_MATCH_MULTIPROC:
            cpu_count = os.cpu_count() or 1
            workers = MATCH_WORKERS or max(1, cpu_count // 2)
            workers = min(workers, len(reference_files))
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for ref_name, df_ref, raw_cc_rg2, err in executor.map(
                    _build_matching_df,
                    [(tif_file, bpp, ref_file) for ref_file in reference_files]
                ):
                    if err is not None:
                        missing_refs.append(ref_name)
                        print(f"    Warning: Failed to build results for {ref_name}: {err}")
                    else:
                        if df_ref is not None and not df_ref.empty:
                            all_results.append(df_ref)
                        if raw_cc_rg2:
                            all_raw_cc_rg2.extend(raw_cc_rg2)
                        if (df_ref is None or df_ref.empty) and not raw_cc_rg2:
                            missing_refs.append(ref_name)
        else:
            for ref_file in reference_files:
                ref_name = os.path.splitext(os.path.basename(ref_file))[0]
                try:
                    df_raw = read_pcc_results(tif_file, ref_name=ref_name)
                    if not df_raw.empty:
                        all_raw_cc_rg2.extend(df_raw['cc_rg2'].tolist())
                        
                    df_ref = build_matching_df_for_tif_local(tif_file, bpp, map_path=ref_file)
                    if df_ref is not None and not df_ref.empty:
                        all_results.append(df_ref)
                    else:
                        missing_refs.append(ref_name)
                except Exception as exc:
                    missing_refs.append(ref_name)
                    print(f"    Warning: Failed to build results for {ref_name}: {exc}")
        
        # Calculate global statistics across ALL references for this tif_file
        global_avg = np.mean(all_raw_cc_rg2) if all_raw_cc_rg2 else 0.0
        global_std = np.std(all_raw_cc_rg2) if all_raw_cc_rg2 else 1.0
        if global_std == 0: global_std = 1.0

        if missing_refs:
            print(f"    Warning: Missing results for {len(missing_refs)} reference(s): {', '.join(missing_refs)}")
        
        # Load maligner_dp results
        maligner_df = load_maligner_dp_results(tif_file, bpp)
        if not maligner_df.empty:
            # Calculate cc_r and cc_g for each maligner result
            for idx, row in maligner_df.iterrows():
                ref_name = row.get('ref_name', 'unknown')
                # Find reference file by ref_name
                ref_file = None
                for rf in reference_files:
                    if os.path.splitext(os.path.basename(rf))[0] == ref_name:
                        ref_file = rf
                        break
                
                if ref_file and os.path.exists(ref_file):
                    shift_bp = row.get('shift_bp', 0)
                    direction = int(row.get('direction', 1))
                    scale = row.get('scale', 1.0)
                    cc_r, cc_g = calculate_cc_for_maligner(tif_file, ref_file, shift_bp, direction, scale, bpp, circular=circular)
                    maligner_df.at[idx, 'cc_r'] = cc_r
                    if pd.notna(cc_g) and cc_g == 0.0:
                        cc_g = 1.0
                    maligner_df.at[idx, 'cc_g'] = cc_g
                else:
                    maligner_df.at[idx, 'cc_r'] = np.nan
                    maligner_df.at[idx, 'cc_g'] = np.nan
            
            # Calculate cc_rg and cc_rg2
            mask = maligner_df['cc_r'].notna() & maligner_df['cc_g'].notna()
            maligner_df.loc[mask, 'cc_rg'] = maligner_df.loc[mask, 'cc_r'] * maligner_df.loc[mask, 'cc_g']
            maligner_df.loc[mask, 'cc_rg2'] = maligner_df.loc[mask, 'cc_r'] * (maligner_df.loc[mask, 'cc_g'] ** 2)
            
            all_results.append(maligner_df)

        if all_results:
            # Combine all results from all references and maligner
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Apply global statistics to each row
            combined_df['cc_rg2_avg'] = global_avg
            combined_df['cc_rg2_std'] = global_std
            
            # Recalculate p_val using global statistics (as log10(P))
            z_scores = (combined_df['cc_rg2'] - global_avg) / global_std
            # Use logsf for better precision with small p-values, then convert to log10
            combined_df['log(P)'] = stats.norm.logsf(z_scores) / np.log(10)

            # Ensure misc column exists (rename score if needed)
            if 'score' in combined_df.columns and 'misc' not in combined_df.columns:
                combined_df = combined_df.rename(columns={'score': 'misc'})
            
            # Sort PCC results by cc_rg2, keep maligner results in original order
            if 'method' in combined_df.columns and 'cc_rg2' in combined_df.columns:
                pcc_df = combined_df[combined_df['method'] == 'pcc'].copy()
                if not pcc_df.empty:
                    pcc_mask = pcc_df['cc_g'].notna() & (pcc_df['cc_g'] == 0.0)
                    pcc_df.loc[pcc_mask, 'cc_g'] = 1.0
                    if pcc_mask.any():
                        mask = pcc_df['cc_r'].notna() & pcc_df['cc_g'].notna()
                        pcc_df.loc[mask, 'cc_rg'] = pcc_df.loc[mask, 'cc_r'] * pcc_df.loc[mask, 'cc_g']
                        pcc_df.loc[mask, 'cc_rg2'] = pcc_df.loc[mask, 'cc_r'] * (pcc_df.loc[mask, 'cc_g'] ** 2)
                    pcc_df = pcc_df.sort_values(by='cc_rg2', ascending=False).reset_index(drop=True)
                    pcc_df = pcc_df.head(10)
                mal_df = combined_df[combined_df['method'] == 'mal'].copy()
                combined_df = pd.concat([pcc_df, mal_df], ignore_index=True)
            
            # Format and Save CSV
            analysis_df = combined_df.sort_values(by='cc_rg2', ascending=False).reset_index(drop=True)
            top1 = analysis_df.iloc[0] if not analysis_df.empty else None
            top2 = analysis_df.iloc[1] if len(analysis_df) > 1 else None

            pcc_part = combined_df[combined_df['method'] == 'pcc'].sort_values(by='cc_rg2', ascending=False).head(10)
            mal_part = combined_df[combined_df['method'] == 'mal']
            df_to_save = pd.concat([pcc_part, mal_part], ignore_index=True)
            
            # Add rank column
            df_to_save['rank'] = 0
            # PCC ranking: 1-10
            pcc_mask = df_to_save['method'] == 'pcc'
            if pcc_mask.any():
                df_to_save.loc[pcc_mask, 'rank'] = range(1, sum(pcc_mask) + 1)
            # mal ranking: 1, 2, ...
            mal_mask = df_to_save['method'] == 'mal'
            if mal_mask.any():
                df_to_save.loc[mal_mask, 'rank'] = range(1, sum(mal_mask) + 1)
            
            # Numeric Formatting for CSV/display
            if 'scale' in df_to_save.columns: df_to_save['scale'] = df_to_save['scale'].round(2)
            if 'shift_bp' in df_to_save.columns:
                df_to_save['aligned_bp'] = df_to_save['shift_bp'].round(0).astype('Int64')
            if 'direction' in df_to_save.columns:
                df_to_save['dir'] = df_to_save['direction']
            if 'cc_r' in df_to_save.columns: df_to_save['cc_r'] = df_to_save['cc_r'].round(2)
            if 'cc_g' in df_to_save.columns: df_to_save['cc_g'] = df_to_save['cc_g'].round(2)
            if 'cc_rg' in df_to_save.columns: df_to_save['cc_rg'] = df_to_save['cc_rg'].apply(lambda x: f"{x:.1e}" if pd.notna(x) else "")
            if 'cc_rg2' in df_to_save.columns: df_to_save['cc_rg2'] = df_to_save['cc_rg2'].apply(lambda x: f"{x:.1e}" if pd.notna(x) else "")
            if 'log(P)' in df_to_save.columns: df_to_save['log(P)'] = df_to_save['log(P)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            if 'misc' in df_to_save.columns: df_to_save['misc'] = df_to_save['misc'].round(0).astype('Int64')
            
            save_cols = ['rank', 'method','aligned_bp','dir','scale','cc_r','cc_g','cc_rg','cc_rg2','log(P)','misc']
            if 'ref_name' in df_to_save.columns: save_cols = ['rank', 'ref_name'] + save_cols[1:]
            available_cols = [col for col in save_cols if col in df_to_save.columns]
            df_to_save[available_cols].to_csv(unified_csv, index=False)

            # --- Comprehensive Analysis for _info.txt ---
            def is_homologous(r1, r2):
                r1, r2 = str(r1), str(r2)
                if r1 == r2: return False
                return r1[:-2] == r2[:-2] and r1[-2:] in ('_p', '_m') and r2[-2:] in ('_p', '_m')

            # mal ranking check from ANALYSIS_DF (which is NOT formatted)
            mal_hits = analysis_df[analysis_df['method'] == 'mal'].copy()
            mal1 = mal_hits.iloc[0] if not mal_hits.empty else None
            mal2 = mal_hits.iloc[1] if len(mal_hits) > 1 else None
            
            pcc_hits = analysis_df[analysis_df['method'] == 'pcc'].copy()
            pcc10 = pcc_hits.iloc[9] if len(pcc_hits) >= 10 else (pcc_hits.iloc[-1] if not pcc_hits.empty else None)
            
            mal_status = "No similar position found in top 2 mal hits"
            if top1 is not None:
                if mal1 is not None and abs(top1['shift_bp'] - mal1['shift_bp']) < 10000:
                    if top1['ref_name'] == mal1['ref_name']: mal_status = f"Matches mal Rank 1 position and chromosome ({top1['ref_name']})"
                    elif is_homologous(top1['ref_name'], mal1['ref_name']): mal_status = f"Matches mal Rank 1 position on homologous chromosome ({mal1['ref_name']})"
                    else: mal_status = f"Matches mal Rank 1 position but different chromosome ({mal1['ref_name']})"
                elif mal2 is not None and abs(top1['shift_bp'] - mal2['shift_bp']) < 10000:
                    if top1['ref_name'] == mal2['ref_name']: mal_status = f"Matches mal Rank 2 position and chromosome ({top1['ref_name']})"
                    elif is_homologous(top1['ref_name'], mal2['ref_name']): mal_status = f"Matches mal Rank 2 position on homologous chromosome ({mal2['ref_name']})"
                    else: mal_status = f"Matches mal Rank 2 position but different chromosome ({mal2['ref_name']})"

            # Save statistics and analysis to _info.txt
            info_txt = tif_file.replace(".tif", "_info.txt")
            with open(info_txt, 'w') as f_info:
                f_info.write(f"Image Width: {w2} px ({int(mol_length_bp):,} bp)\n")
                if top1 is not None:
                    shift_str = f"{int(top1['shift_bp']):,}"
                    scaled_len = mol_length_bp * top1['scale']
                    f_info.write(f"Top Hit (cc_rg2 Rank 1): {top1['ref_name']} at {shift_str} bp\n")
                    f_info.write(f"Scaled Length: {int(scaled_len):,} bp (scale: {top1['scale']:.2f})\n")
                    
                    # log(P) comparison with PCC Rank 10
                    p_val_str = f"{top1['log(P)']:.2f}"
                    if pcc10 is not None and abs(pcc10['log(P)']) > 0:
                        ratio = top1['log(P)'] / pcc10['log(P)']
                        p_val_str += f" ({ratio:.2f}x vs PCC Rank 10)"
                    f_info.write(f"log(P): {p_val_str}\n")
                    
                    f_info.write(f"mal Comparison: {mal_status}\n")
                
                if top1 is not None and top2 is not None:
                    if is_homologous(top1['ref_name'], top2['ref_name']):
                        f_info.write(f"Homologous Analysis: Rank 1 ({top1['ref_name']}) and Rank 2 ({top2['ref_name']}) are a homologous pair.\n")
                
                f_info.write(f"\nGlobal Statistics (cc_rg2):\n")
                f_info.write(f"  Average: {global_avg:.2e}\n")
                f_info.write(f"  Std Dev: {global_std:.2e}\n")
            
            # Clean up intermediate log/map files
            data_name = os.path.splitext(os.path.basename(tif_file))[0]
            data_dir = os.path.dirname(tif_file)
            for ref_file in reference_files:
                ref_name = os.path.splitext(os.path.basename(ref_file))[0]
                pcc_log = os.path.join(data_dir, f"{data_name}_{ref_name}_pcc.log")
                if os.path.exists(pcc_log):
                    try: os.remove(pcc_log)
                    except Exception: pass

            generic_pcc_log = os.path.join(data_dir, f"{data_name}_pcc.log")
            if os.path.exists(generic_pcc_log):
                try: os.remove(generic_pcc_log)
                except Exception: pass

            mal_log = os.path.join(data_dir, f"{data_name}_mal.log")
            if os.path.exists(mal_log):
                try: os.remove(mal_log)
                except Exception: pass
            
        else:
            print(f"    No results found for {os.path.basename(tif_file)}")
    
    print(f"\n  - All processing completed.\n")
    return mal_end_time

def main():
    """Main function to calculate matching results CSV files"""
    args = parse_args()
    
    # Get data folder
    tif_data_folder = get_data_folder(args)
    print(f"Data folder: {tif_data_folder}")
    
    # Get reference files
    ref_map_folder = get_ref_map_folder(tif_data_folder, args.ref_map_folder)
    if ref_map_folder is None:
        print("Error: Reference map folder not found.")
        sys.exit(1)
    
    reference_files = get_reference_files(ref_map_folder)
    if not reference_files:
        print(f"Error: No reference TIF files found in {ref_map_folder}")
        sys.exit(1)
    
    print(f"Found {len(reference_files)} reference file(s) in {ref_map_folder}.")
    print()
    
    # Get TIFF data files
    tif_files = Read_TIFF_folder(tif_data_folder)
    if not tif_files:
        print(f"Error: No TIFF files found in {tif_data_folder}")
        sys.exit(1)
    
    tif_files = sorted(tif_files, key=lambda x: os.path.basename(x).lower())
    print(f"Found {len(tif_files)} TIFF file(s) in {tif_data_folder}")
    print()
    
    # Calculate matching results CSV files
    start_time = datetime.now()
    print(f"Starting matching results calculation...")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"BPP: {args.bpp}")
    if args.circular:
        print(f"Genome type: circular")
    print()
    
    mal_end_time = ensure_matching_csvs(tif_files, tif_data_folder, reference_files, args.bpp, start_time=start_time, circular=args.circular)
    
    end_time = datetime.now()
    print("\nCalculation completed.")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    total_elapsed = end_time - start_time
    if mal_end_time is not None:
        mal_elapsed = mal_end_time - start_time
        post_mal_elapsed = end_time - mal_end_time
        print(f"Total elapsed time: {total_elapsed} (maligner_dp: {mal_elapsed}, pcc: {post_mal_elapsed})")
    else:
        print(f"Total elapsed time: {total_elapsed}")
    
    # Print created matching_results.csv files
    print("\nCreated matching_results.csv files:")
    csv_files = []
    for tif_file in tif_files:
        data_name = os.path.splitext(os.path.basename(tif_file))[0]
        data_dir = os.path.dirname(tif_file)
        unified_csv = os.path.join(data_dir, f"{data_name}_matching_results.csv")
        if os.path.exists(unified_csv):
            csv_files.append(unified_csv)
            print(f"  {os.path.basename(unified_csv)}")
    
    if not csv_files:
        print("  (No matching_results.csv files found)")
    else:
        print(f"\nTotal: {len(csv_files)} file(s)")
        print("\n" + "="*80)
        for csv_file in csv_files:
            print(f"\n{os.path.basename(csv_file)}:")
            print("-"*80)
            try:
                df = pd.read_csv(csv_file, dtype={'cc_rg': str, 'cc_rg2': str})
                if 'log(P)' in df.columns:
                    # Format log10(P) to 2 decimal places for display
                    df['log(P)'] = pd.to_numeric(df['log(P)'], errors='coerce').apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else ""
                    )
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', None)
                print(df.to_string(index=False))
                
                # Print _info.txt content below the table
                # Look for [data_name]_info.txt
                info_file = csv_file.replace("_matching_results.csv", "_info.txt")
                if not os.path.exists(info_file):
                    # Fallback: try replacing .csv with .txt if naming is different
                    info_file = csv_file.replace(".csv", ".txt").replace("_matching_results", "_info")
                
                if os.path.exists(info_file):
                    print("\n" + "-" * 40)
                    print(f"ANALYSIS INFO: {os.path.basename(info_file)}")
                    print("-" * 40)
                    with open(info_file, 'r') as f_info:
                        print(f_info.read().strip())
                else:
                    print(f"\n(Info file not found: {os.path.basename(info_file)})")
                print()
            except Exception as e:
                print(f"  Error reading {os.path.basename(csv_file)}: {e}")

if __name__ == '__main__':
    main()
