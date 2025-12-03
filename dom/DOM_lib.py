#!/usr/bin/env python
"""
DOM_lib.py

Core utility functions for TIF processing, correlation calculations,
and image manipulation for DOM analysis.
"""

import os
import re
import sys
import numpy as np
from scipy import signal
from PIL import Image
import tifffile as tiff
from numpy.fft import fft, ifft, fftshift
import math
import numba
from Bio.Seq import Seq
import DOM_constants

IMAGE_SCALE_FACTOR = 280
GREEN_THRESHOLD_FACTOR = 0.1

@numba.njit
def corrcoef(x, y):
    """Calculate Pearson correlation coefficient between two arrays."""
    x_max = np.max(x)
    x_min = np.min(x)
    x_gap = x_max - x_min
    if x_gap == 0:
        x_gap = 1
    x = (x - x_min) / x_gap
    
    y_max = np.max(y)
    y_min = np.min(y)
    y_gap = y_max - y_min
    if y_gap == 0:
        y_gap = 1
    y = (y - y_min) / y_gap
    
    n = len(y)
    xy = np.dot(x, y)
    xx = np.dot(x, x)
    yy = np.dot(y, y)
    s_x = np.sum(x)
    s_y = np.sum(y)
    denominator = math.sqrt((n * xx - s_x * s_x) * (n * yy - s_y * s_y))
    if denominator == 0:
        return 0
    cc = (n * xy - s_x * s_y) / denominator
    return cc

@numba.njit
def dcc(x, y):
    """Calculate cross correlation without mean subtraction (Ambjorsson method)."""
    n = len(y)
    x_contig = np.ascontiguousarray(x)
    y_contig = np.ascontiguousarray(y)
    x_dot = np.dot(x_contig, x_contig)
    y_dot = np.dot(y_contig, y_contig)
    denominator = math.sqrt(x_dot) * math.sqrt(y_dot)
    if denominator == 0:
        return 0.0
    cc = np.dot(x_contig, y_contig) / denominator
    return cc

def cross_correlation_using_fft(x, y):
    """Calculate cross correlation using FFT."""
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

def compute_shift(x, y):
    """Compute optimal shift between two signals using cross correlation."""
    assert len(x) == len(y), "Signals must have the same length"
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x), "Correlation length mismatch"  
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift

def Read_TIFF_folder(folder_path):
    """Read all TIFF files from a folder, sorted by file size (largest first)."""
    folder_path = os.path.normpath(folder_path)
    file_list = sorted(os.listdir(folder_path))
    file_list.sort(key=lambda f: os.stat(os.path.join(folder_path, f)).st_size, reverse=True)
    tiff_files = []
    for filename in file_list:
        if filename.lower().endswith(('.tif', '.tiff')):
            tiff_files.append(os.path.join(folder_path, filename))
    return tiff_files

def tif2RGimage(file_path):
    """Convert TIFF file to RG (Red-Green) image format."""
    image = Image.open(file_path)
    width, height = image.size
    
    if image.mode == 'I;16B':
        image_data = tiff.imread(file_path)
        red_channel = _process_channel(image_data[0].flatten(), apply_threshold=False)
        green_channel = _process_channel(image_data[1].flatten(), apply_threshold=True)
        blue_channel = np.zeros(height * width)
        
        # Reshape to (height, width, 3) for Image.fromarray
        pixel_data = np.column_stack((red_channel, green_channel, blue_channel)).astype('uint8')
        pixel_data = pixel_data.reshape(height, width, 3)
        image = Image.fromarray(pixel_data, mode='RGB')
    
    return image, width, height    

def tif2RGprofile(file_path):
    """Extract R and G channel profiles and peaks from TIFF file."""
    image = Image.open(file_path)
    
    if image.mode == 'I;16B':
        # Experimental TIFF files
        image_data = tiff.imread(file_path)
        red_profile = np.max(image_data[0], axis=0)
        green_profile = np.max(image_data[1], axis=0)
    elif image.mode == 'RGB':
        # Image from FASTA file
        width, height = image.size
        color_channels = image.split()
        red_data = np.array(color_channels[0].getdata())
        red_data = red_data.reshape(height, width)
        red_profile = np.max(red_data, axis=0)
        green_data = np.array(color_channels[1].getdata())
        green_data = green_data.reshape(height, width)
        green_profile = np.max(green_data, axis=0)
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")
    
    # Apply threshold to green channel
    threshold = _calculate_green_threshold(green_profile)
    green_profile[green_profile < threshold] = 0
    
    peaks, _ = signal.find_peaks(green_profile)
    return red_profile, green_profile, peaks

def normalize(array):
    """Normalize array to [0, 1] range."""
    if len(array) == 0:
        return array
    min_val = array.min()
    max_val = array.max()
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(array)
    normalized = (array - min_val) / range_val
    return normalized


def _calculate_green_threshold(green_channel):
    """Calculate threshold for green channel filtering."""
    return green_channel.mean() + (green_channel.max() - green_channel.mean()) * GREEN_THRESHOLD_FACTOR


def _process_channel(channel_data, apply_threshold=False):
    """Process a single channel: normalize, optionally apply threshold, and scale."""
    channel = channel_data - channel_data.min()
    channel = normalize(channel)
    
    if apply_threshold:
        threshold = _calculate_green_threshold(channel)
        channel[channel < threshold] = 0
    
    channel = channel * IMAGE_SCALE_FACTOR
    channel = np.clip(channel, 0, 255)
    return channel

def peaks2maps(tiff_filename, dna_size_bp, peaks, output_filename):
    """Write peaks to maps file format."""
    with open(output_filename, "w", encoding='utf-8') as map_file:
        fragments = [tiff_filename, dna_size_bp, len(peaks) - 1]
        for i in range(len(peaks) - 1):
            fragments.append(peaks[i + 1] - peaks[i])
        line = '\t'.join(str(value) for value in fragments)
        print(line, file=map_file)

def validate_tif_path(map_path):
    """Validate and return TIF file path."""
    map_path_lower = map_path.lower()
    
    # Check if file exists first
    if not os.path.isfile(map_path):
        print(f"Error: Reference map file not found: {map_path}", file=sys.stderr)
        print(f"Please check the file path and try again.", file=sys.stderr)
        sys.exit(1)
    
    # Check file extension
    if map_path_lower.endswith(('.fasta', '.fa', '.fas')):
        print(f"Error: FASTA files are not supported as reference map.", file=sys.stderr)
        print(f"Please convert the FASTA file to TIF format first using seq2RGmap.py or simulation.py.", file=sys.stderr)
        sys.exit(1)
    
    if not (map_path_lower.endswith('.tif') or map_path_lower.endswith('.tiff')):
        print(f"Error: Reference map must be a TIF file (.tif or .tiff).", file=sys.stderr)
        print(f"Received: {map_path}", file=sys.stderr)
        sys.exit(1)
    
    return map_path


def low_pass_filter(signal_data, order, cutoff):
    """Apply low-pass filter to signal."""
    b, a = signal.butter(order, cutoff)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal


def read_fasta_sequence(fasta_file):
    """Read sequence from FASTA file, skipping header line."""
    with open(fasta_file, "r") as f:
        first_line = f.readline()
        if first_line.find(">") == 0:
            first_line = ""
        sequence = (first_line + f.read()).replace("\r", "").replace("\n", "")
    return sequence


def get_reverse_complement(sequence):
    """Get reverse complement of sequence using biopython."""
    seq = Seq(sequence)
    reverse_complement = seq.reverse_complement()
    return str(reverse_complement)


def convert_iupac_to_regex(pattern):
    """Convert IUPAC code to regex pattern for faster search."""
    iupac_map = {
        'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C',
        'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
        'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
        'H': '[ACT]', 'V': '[ACG]', 'N': '[ATGC]',
        'a': 'a', 't': 't', 'g': 'g', 'c': 'c',
        'r': '[ag]', 'y': '[ct]', 's': '[gc]', 'w': '[at]',
        'k': '[gt]', 'm': '[ac]', 'b': '[cgt]', 'd': '[agt]',
        'h': '[act]', 'v': '[acg]', 'n': '[atgc]'
    }
    regex_pattern = ''.join([iupac_map.get(char, char) for char in pattern])
    return regex_pattern


def search_sequence(sequence, search_pattern):
    """Search for pattern in sequence, handling IUPAC codes."""
    iupac_codes = set('RYSWKMBDHVNryswkmbdhvn')
    has_iupac = any(char in iupac_codes for char in search_pattern)
    
    if has_iupac:
        # Use regex for IUPAC codes (faster than nt_search)
        regex_pattern = convert_iupac_to_regex(search_pattern)
        positions = [m.start() for m in re.finditer(regex_pattern, sequence, re.IGNORECASE)]
    else:
        # Use simple string find for non-IUPAC (fastest)
        positions = []
        pos = 0
        while pos < len(sequence):
            pos = sequence.find(search_pattern, pos)
            if pos == -1:
                break
            positions.append(pos)
            pos += 1
    return positions


def get_sequence_positions(fasta_file, search_pattern):
    """Find all positions of search pattern in FASTA file (forward and reverse complement)."""
    sequence = read_fasta_sequence(fasta_file)
    positions = search_sequence(sequence, search_pattern)
    positions.extend(search_sequence(sequence, get_reverse_complement(search_pattern)))
    positions = list(set(positions))
    positions.sort()
    positions.append(len(sequence))  # Add sequence length as final position
    return positions
