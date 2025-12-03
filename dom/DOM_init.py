#!/usr/bin/env python
"""
DOM_init.py

Configuration and argument parsing for DOM.py.
"""

import argparse
import os
import sys
import DOM_constants

def parse_args():
    """Parse command-line arguments for DOM.py"""
    parser = argparse.ArgumentParser(description='DOM: Dual-channel matching tool')
    parser.add_argument('data_folder', 
                       help='Folder containing tif data files.')
    parser.add_argument('--map', dest='map', default=DOM_constants.DEFAULT_MAP, 
                       help=f'Reference map (default: {DOM_constants.DEFAULT_MAP})')
    parser.add_argument('--bpp', type=int, default=DOM_constants.DEFAULT_BPP, 
                       help=f'Base pairs per pixel (default: {DOM_constants.DEFAULT_BPP})')
    parser.add_argument('--shift-window', type=int, default=DOM_constants.DEFAULT_SHIFT_WINDOW,
                       dest='shift_window', help=f'Window size in pixels for grouping similar shifts (default: {DOM_constants.DEFAULT_SHIFT_WINDOW})')
    parser.add_argument('--start-bpp', type=int, default=DOM_constants.DEFAULT_START_BPP,
                       dest='start_bpp', help=f'Start BPP for CC calculation range (default: {DOM_constants.DEFAULT_START_BPP})')
    parser.add_argument('--end-bpp', type=int, default=DOM_constants.DEFAULT_END_BPP,
                       dest='end_bpp', help=f'End BPP for CC calculation range (default: {DOM_constants.DEFAULT_END_BPP})')
    parser.add_argument('--maligner-path', type=str, default=DOM_constants.DEFAULT_MALIGNER_PATH,
                       dest='maligner_path', help=f'Path to maligner executable (default: {DOM_constants.DEFAULT_MALIGNER_PATH})')
    return parser.parse_args()

def get_runtime_config(parsed_args):
    """Extract runtime configuration from parsed arguments"""
    tif_data_folder = parsed_args.data_folder
    
    map_path = parsed_args.map
    if tif_data_folder and not os.path.exists(map_path):
        data_abs = os.path.abspath(tif_data_folder)
        if os.path.isdir(data_abs):
            parent_dir = os.path.dirname(data_abs)
            map_path = os.path.join(parent_dir, DOM_constants.DEFAULT_MAP)
        else:
            parent_dir = os.path.dirname(data_abs)
            map_path = os.path.join(parent_dir, DOM_constants.DEFAULT_MAP)
    
    return {
        'map_path': map_path,
        'tif_data_folder': tif_data_folder,
        'BPP': parsed_args.bpp,
        'SHIFT_WINDOW': parsed_args.shift_window,
        'START_BPP': parsed_args.start_bpp,
        'END_BPP': parsed_args.end_bpp,
    }

def init_config():
    """Parse arguments and return runtime configuration"""
    args = parse_args()
    config = get_runtime_config(args)
    
    if config['map_path'] and not os.path.exists(config['map_path']):
        print(f"Error: Reference map not found: {config['map_path']}", file=sys.stderr)
        print(f"Please prepare {os.path.abspath(DOM_constants.DEFAULT_MAP)} or specify with --map option.", file=sys.stderr)
        sys.exit(1)
    
    return config

