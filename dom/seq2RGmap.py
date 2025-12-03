#!/usr/bin/env python
"""
seq2RGmap.py

Generates RG map image (TIF file) from FASTA sequence file.
Searches for specific sequence patterns and creates a visual map.
"""

import os
import argparse
import numpy as np
from PIL import Image
import DOM_lib as lib
import DOM_constants


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate RG map from FASTA sequence file.'
    )
    parser.add_argument('fasta_file',
                        help='Input FASTA sequence file.')
    parser.add_argument('--seq-for-r', dest='seq_for_r',
                        default=DOM_constants.DEFAULT_SEQ_FOR_R,
                        help=f'Sequence pattern for R channel (default: {DOM_constants.DEFAULT_SEQ_FOR_R})')
    parser.add_argument('--seq-for-g', dest='seq_for_g',
                        default=DOM_constants.DEFAULT_SEQ_FOR_G,
                        help=f'Sequence pattern for G channel (default: {DOM_constants.DEFAULT_SEQ_FOR_G})')
    parser.add_argument('--bpp', type=int, default=DOM_constants.DEFAULT_BPP,
                        dest='bpp',
                        help=f'Base pairs per pixel (default: {DOM_constants.DEFAULT_BPP})')
    parser.add_argument('--output', dest='output_path', default=DOM_constants.DEFAULT_MAP,
                        help=f'Output RG map file path (default: {DOM_constants.DEFAULT_MAP})')
    return parser.parse_args()


def create_rg_map_image(r_positions, g_positions, bpp):
    """Convert position lists to RG map image."""
    seq_length = r_positions[-1]
    bins = np.arange(0, seq_length, bpp)
    
    r_counts, _ = np.histogram(r_positions, bins=bins)
    g_counts, _ = np.histogram(g_positions, bins=bins)
    
    r_filtered = lib.low_pass_filter(r_counts, 4, 0.4)
    g_filtered = lib.low_pass_filter(g_counts, 4, 0.4)

    # Normalize to 5-255 range. If max == min, set to 0 to avoid division by zero
    r_range = r_filtered.max() - r_filtered.min()
    if r_range == 0:
        r_normalized = np.full_like(r_filtered, 0)
    else:
        r_normalized = (r_filtered - r_filtered.min()) / r_range * 250 + 5

    g_range = g_filtered.max() - g_filtered.min()
    if g_range == 0:
        g_normalized = np.full_like(g_filtered, 0)
    else:
        g_normalized = (g_filtered - g_filtered.min()) / g_range * 250 + 5
    
    width = r_counts.size
    height = 10
    
    r_rows = np.tile(r_normalized, height)
    g_rows = np.tile(g_normalized, height)
    b_rows = np.zeros_like(r_rows)
    
    pixel_data = np.column_stack((r_rows, g_rows, b_rows)).astype('uint8')
    rg_map_image = Image.frombuffer('RGB', (width, height), pixel_data, 'raw', 'RGB', 0, 0)
    return rg_map_image


def main():
    """Main function to generate RG map from FASTA file."""
    args = parse_args()
    fasta_file = args.fasta_file.replace('\\', '/')
    seq_for_r = args.seq_for_r
    seq_for_g = args.seq_for_g
    bpp = args.bpp
    output_path = args.output_path

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find sequence positions for R and G channels
    r_positions = lib.get_sequence_positions(fasta_file, seq_for_r)
    g_positions = lib.get_sequence_positions(fasta_file, seq_for_g)
    
    # Create and save RG map image
    rg_map_image = create_rg_map_image(r_positions, g_positions, bpp)
    rg_map_image.save(output_path)
    print(f'{output_path} was created.')


if __name__ == '__main__':
    main()
