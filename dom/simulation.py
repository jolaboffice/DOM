#!/usr/bin/env python
"""
simulation.py

Simulates DNA binding sites and generates intensity profiles from FASTA sequences.
Creates in-silico RG map image based on binding affinity patterns.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from Bio import SeqIO
from PIL import Image
import DOM_lib as lib
import DOM_constants

# Constants
FILTER_ORDER = 4
FILTER_CUTOFF = 0.4
IMAGE_HEIGHT = 10
NORMALIZE_MIN = 5
NORMALIZE_MAX = 255
NORMALIZE_RANGE = 250
PSF_FWHM = 382
PSF_CONVERSION = 0.34
BINDING_LEN = 6
BLOCK_LEN = 10
BINDING_EFFICIENCY = 1
SIGMA_MULTIPLIER = 1.5


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Simulate DNA binding sites and generate intensity profiles from FASTA sequences.'
    )
    parser.add_argument('fasta_file',
                        help='Input FASTA sequence file.')
    parser.add_argument('--n-fragments', type=int, default=1,
                        dest='n_fragments',
                        help='Number of fragments to divide sequence (default: 10)')
    parser.add_argument('--sim-times', type=int, default=10,
                        dest='sim_times',
                        help='Number of simulation runs per fragment (default: 10)')
    parser.add_argument('--binding-len', type=int, default=BINDING_LEN,
                        dest='binding_len',
                        help=f'Binding length (default: {BINDING_LEN})')
    parser.add_argument('--block-len', type=int, default=BLOCK_LEN,
                        dest='block_len',
                        help=f'Block length (default: {BLOCK_LEN})')
    parser.add_argument('--binding-efficiency', type=float, default=BINDING_EFFICIENCY,
                        dest='binding_efficiency',
                        help=f'Binding efficiency (default: {BINDING_EFFICIENCY})')
    parser.add_argument('--seq-for-r', dest='seq_for_r',
                        default=DOM_constants.DEFAULT_SEQ_FOR_R,
                        help=f'Sequence pattern for R channel (default: {DOM_constants.DEFAULT_SEQ_FOR_R})')
    parser.add_argument('--seq-for-g', dest='seq_for_g',
                        default=DOM_constants.DEFAULT_SEQ_FOR_G,
                        help=f'Sequence pattern for G channel (default: {DOM_constants.DEFAULT_SEQ_FOR_G})')
    parser.add_argument('--bpp', type=int, default=DOM_constants.DEFAULT_BPP,
                        dest='bpp',
                        help=f'Base pairs per pixel for output map (default: {DOM_constants.DEFAULT_BPP})')
    parser.add_argument('--output', dest='output_path', default=DOM_constants.DEFAULT_MAP,
                        help=f'Output RG map file path (default: {DOM_constants.DEFAULT_MAP})')
    return parser.parse_args()


def search_pos_list2map_img(sim_r, r_data, g_data, bin_size):
    """Convert position lists to RG map image."""
    seq_len = r_data[-1]
    bins = np.arange(0, seq_len, bin_size)
    r_counts, _ = np.histogram(r_data, bins=bins)
    g_counts, _ = np.histogram(g_data, bins=bins)

    r_filtered = lib.low_pass_filter(r_counts, FILTER_ORDER, FILTER_CUTOFF)
    g_filtered = lib.low_pass_filter(g_counts, FILTER_ORDER, FILTER_CUTOFF)

    # Resize simulation result to match filtered data
    sim_r = cv2.resize(sim_r, (r_filtered.shape[0], 1), interpolation=cv2.INTER_AREA)[0]
    r_filtered = sim_r

    # Normalize to 5-255 range
    r_range = r_filtered.max() - r_filtered.min()
    if r_range == 0:
        r_normalized = np.full_like(r_filtered, 0)
    else:
        r_normalized = (r_filtered - r_filtered.min()) / r_range * NORMALIZE_RANGE + NORMALIZE_MIN

    g_range = g_filtered.max() - g_filtered.min()
    if g_range == 0:
        g_normalized = np.full_like(g_filtered, 0)
    else:
        g_normalized = (g_filtered - g_filtered.min()) / g_range * NORMALIZE_RANGE + NORMALIZE_MIN

    width = r_counts.size
    height = IMAGE_HEIGHT

    # Tile arrays to create image rows
    r_rows = np.tile(r_normalized, height)
    g_rows = np.tile(g_normalized, height)
    b_rows = np.zeros_like(r_rows)

    pixel_data = np.column_stack((r_rows, g_rows, b_rows)).astype('uint8')
    in_silico_image = Image.frombuffer('RGB', (width, height), pixel_data, 'raw', 'RGB', 0, 0)
    return in_silico_image


def make_ref_img(dna_path, strings, n_fold, n):
    """Read FASTA sequence and create binary reference image for specified fragment."""
    my_seq = SeqIO.read(dna_path, 'fasta')
    my_seq_len = int(len(my_seq) / n_fold)
    my_seq = my_seq[(n * my_seq_len):((n + 1) * my_seq_len + 1)]
    print(f'Fragment length: {my_seq_len}')
    my_seq = str(my_seq.seq)
    reference_img = np.zeros(len(my_seq))
    print(f'Reference image shape: {reference_img.shape}')

    for string in strings:
        for i in range(len(my_seq) - len(string) + 1):
            if my_seq[i:(i + len(string))] == string:
                reference_img[i:(i + len(string))] = 1
    return reference_img


def _get_affinity_for_pattern(search_region, at_count):
    """Get binding affinity based on AT count and pattern."""
    affinity_map = {
        1: 0.2,
        2: {tuple([1, 1, 0, 0, 0, 0]): 2.0,
            tuple([0, 1, 1, 0, 0, 0]): 2.0,
            tuple([0, 0, 1, 1, 0, 0]): 2.0,
            'default': 0.4},
        3: {tuple([1, 1, 1, 0, 0, 0]): 3.4,
            tuple([0, 1, 1, 1, 0, 0]): 3.4,
            tuple([1, 0, 1, 0, 1, 0]): 0.6,
            tuple([1, 0, 1, 0, 0, 1]): 0.6,
            'default': 2.2},
        4: {tuple([1, 1, 1, 1, 0, 0]): 18.1,
            tuple([0, 1, 1, 1, 1, 0]): 18.1,
            tuple([1, 1, 1, 0, 1, 0]): 9.8,
            tuple([1, 1, 1, 0, 0, 1]): 6.7,
            tuple([1, 1, 0, 1, 0, 1]): 5.4,
            tuple([1, 1, 0, 1, 1, 0]): 8.4,
            tuple([1, 1, 0, 0, 1, 1]): 6.2,
            tuple([1, 0, 1, 1, 0, 1]): 2.4,
            'default': None},
        5: {tuple([1, 1, 1, 1, 1, 0]): 43.9,
            tuple([1, 1, 1, 1, 0, 1]): 31.1,
            tuple([1, 1, 1, 0, 1, 1]): 26.9,
            'default': None},
        6: 100.0
    }

    if at_count not in affinity_map:
        return None

    affinity = affinity_map[at_count]

    if isinstance(affinity, dict):
        pattern_tuple = tuple(search_region.tolist())
        reverse_tuple = tuple(search_region[::-1].tolist())
        affinity = affinity.get(pattern_tuple) or affinity.get(reverse_tuple) or affinity.get('default')
        if affinity is None:
            return None

    return affinity


def find_binding_site(reference_img, binding_len, block_len, binding_efficiency):
    """Find binding sites based on AT patterns and affinity."""
    ref_img_len = reference_img.shape[0]
    binding_site_img = np.zeros(ref_img_len)

    # First pass: determine binding sites based on affinity
    for i in range(ref_img_len - binding_len):
        search_region = reference_img[i:i + binding_len]
        at_count = int(np.sum(search_region))

        affinity = _get_affinity_for_pattern(search_region, at_count)
        if affinity is None:
            continue

        random_val = np.random.rand()
        if random_val > (1 - affinity / 100):
            binding_site_img[i + int(binding_len / 2)] = 1

    # Second pass: apply binding efficiency filter
    for i in range(ref_img_len):
        if binding_site_img[i] == 1:
            random_val = np.random.rand()
            if random_val > binding_efficiency:
                binding_site_img[i] = 0

    # Third pass: apply block length effects
    index_list = [i for i in range(ref_img_len) if binding_site_img[i] == 1]
    index_np = np.array(index_list)
    np.random.shuffle(index_np)

    for index in index_np:
        if binding_site_img[index] == 1:
            mu, sigma = block_len, SIGMA_MULTIPLIER
            s = np.random.normal(mu, sigma, 1)[0]
            start_idx = max(0, index - int(s / 2))
            end_idx = min(ref_img_len, index + int(s / 2))
            binding_site_img[start_idx:end_idx] = 0
            binding_site_img[index] = 1

    return binding_site_img


def binary2img(img, thickness):
    """Convert binary array to image with specified thickness."""
    img_len = img.shape[0]
    reference_img = np.zeros(img_len).astype('uint8')
    for i in range(img_len):
        if img[i] == 1:
            reference_img[i:(i + thickness)] = 255

    reference_img_rows = np.array([reference_img])
    for _ in range(IMAGE_HEIGHT):
        reference_img_rows = np.concatenate((reference_img_rows, np.array([reference_img])), axis=0)

    reference_img_resized = cv2.resize(reference_img_rows, (0, 0), fx=1, fy=500,
                                       interpolation=cv2.INTER_LINEAR)
    return reference_img_resized


def gaussian(x, x0, sigma):
    """Calculate Gaussian function value."""
    A = 1 / (sigma * np.sqrt(2 * np.pi))
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def psf(binding_site_img, fwhm):
    """Apply point spread function to binding site image."""
    fwhm_converted = fwhm / PSF_CONVERSION
    sigma = fwhm_converted / (2 * np.sqrt(2 * np.log(2)))
    psf_img = np.zeros(binding_site_img.shape[0])

    for i in range(binding_site_img.shape[0]):
        if binding_site_img[i] == 1:
            for j in range(psf_img.shape[0]):
                psf_img[j] += gaussian(j, i, sigma)

    return psf_img


def simulate_fragment(path, binding_len, block_len, efficiency, file_name, n_fold, n):
    """Simulate a single fragment and save PSF image."""
    reference_img = make_ref_img(path, ['A', 'T'], n_fold, n)
    binding_site_img = find_binding_site(reference_img, binding_len, block_len, efficiency)
    psf_img = psf(binding_site_img, PSF_FWHM)

    folder_name = f'{n_fold}_{n}'
    os.makedirs(f'./{folder_name}', exist_ok=True)
    np.save(f'./{folder_name}/{file_name}', psf_img)
    return psf_img


def main():
    """Main function to run simulation."""
    args = parse_args()
    fasta_path = args.fasta_file
    n_fragments = args.n_fragments
    sim_times = args.sim_times
    binding_len = args.binding_len
    block_len = args.block_len
    binding_efficiency = args.binding_efficiency
    seq_for_r = args.seq_for_r
    seq_for_g = args.seq_for_g
    bpp = args.bpp
    output_path = args.output_path

    all_fragment_avg_img = np.zeros((1, 0))

    for i in range(n_fragments):
        # Simulate multiple times for each fragment
        psf_imgs_list = []
        for j in range(sim_times):
            psf_img = simulate_fragment(fasta_path, binding_len, block_len, binding_efficiency,
                                       str(j), n_fragments, i)
            psf_imgs_list.append(psf_img)

        # Average all simulations for this fragment
        img_width = psf_imgs_list[0].shape[0]
        total_img = np.zeros(img_width)
        for psf_img in psf_imgs_list:
            total_img += psf_img

        fragment_avg_img = total_img / len(psf_imgs_list)
        fragment_avg_img = np.array([fragment_avg_img])
        all_fragment_avg_img = np.concatenate((all_fragment_avg_img, fragment_avg_img), axis=1)

    # Save simulation result
    np.save('./sim_result', all_fragment_avg_img[0])

    # Generate final map image
    r_positions = lib.get_sequence_positions(fasta_path, seq_for_r)
    g_positions = lib.get_sequence_positions(fasta_path, seq_for_g)
    map_img = search_pos_list2map_img(all_fragment_avg_img[0], r_positions, g_positions, bpp)
    map_img.save(output_path)
    print(f'{output_path} was created.')


if __name__ == '__main__':
    main()
