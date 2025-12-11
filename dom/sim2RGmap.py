#!/usr/bin/env python
"""
sim2RGmap.py

Simulates DNA binding sites and generates intensity profile (RG map).
Generates TIF and NPY files from FASTA sequence files.
"""

import numpy as np
import cv2
from Bio import SeqIO
import os
import sys
import time
import argparse
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import DOM_lib as lib
import DOM_constants

# Import Numba (added prange for parallel processing)
try:
    from numba import njit, prange
except ImportError:
    print("Numba library is not installed. Please run 'pip install numba'.")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Simulate DNA binding sites and generate RG map from FASTA sequence file(s).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single FASTA file:
  sim2RGmap.py example.fasta 100
  sim2RGmap.py example.fasta 100 --n-fragments 10
  
  # Process all FASTA files in a folder:
  sim2RGmap.py input_folder 100
  
  # Overwrite existing files without prompting:
  sim2RGmap.py example.fasta 100 --overwrite

Output:
  For each FASTA file, creates in the same directory:
    - {basename}_simRG{sim_times}.tif (RG map image)
    - {basename}_simRG{sim_times}.npy (intensity array)
    - Intermediate simulation results in {basename}/{n_fold}_{fragment_idx}/ folders
        """
    )
    parser.add_argument('input_path',
                        help='Input FASTA file path or folder containing FASTA files.')
    parser.add_argument('sim_times', type=int,
                        help='Number of simulations per fragment.')
    parser.add_argument('--n-fragments', dest='n_fragments', type=int, default=None,
                        help='Number of fragments to split the sequence into (default: 1).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files without prompting.')
    parser.add_argument('--bpp', type=int, default=DOM_constants.DEFAULT_BPP,
                        help=f'Base pairs per pixel (default: {DOM_constants.DEFAULT_BPP})')
    parser.add_argument('--seq-for-r', dest='seq_for_r',
                        default=DOM_constants.DEFAULT_SEQ_FOR_R,
                        help=f'Sequence pattern for R channel (default: {DOM_constants.DEFAULT_SEQ_FOR_R})')
    parser.add_argument('--seq-for-g', dest='seq_for_g',
                        default=DOM_constants.DEFAULT_SEQ_FOR_G,
                        help=f'Sequence pattern for G channel (default: {DOM_constants.DEFAULT_SEQ_FOR_G})')
    
    return parser.parse_args()


def search_pos_list2map_img1(sim_r, r_data, g_data, bin_size):
    """Convert position lists to map image."""
    seq_len = r_data[-1]
    x = np.arange(0, seq_len, bin_size)

    r_y, _ = np.histogram(r_data, bins=x)
    g_y, _ = np.histogram(g_data, bins=x)

    # Green channel: presence-only (treat multiple occurrences in a bin as 1)
    g_y = (g_y > 0).astype(np.float64)

    r_f = lib.low_pass_filter(r_y, 4, 0.4)
    g_f = lib.low_pass_filter(g_y, 4, 0.4)

    # Ensure sim_r is in the correct format for cv2.resize
    if len(sim_r.shape) == 1:
        sim_r = sim_r.reshape(1, -1)
    elif sim_r.shape[0] != 1:
        sim_r = sim_r.flatten().reshape(1, -1)

    sim_r_resized = cv2.resize(sim_r, (r_f.shape[0], 1), interpolation=cv2.INTER_AREA)[0]
    r_f = sim_r_resized

    # Normalize and scale to 0-255 range
    r = (r_f - r_f.min()) / (r_f.max() - r_f.min() + 1e-10) * 250 + 5
    g = (g_f - g_f.min()) / (g_f.max() - g_f.min() + 1e-10) * 250 + 5

    w = r_y.size
    h = 10
    
    # Repeat rows to create image height
    r = np.tile(r, h)
    g = np.tile(g, h)
    b = np.zeros_like(r)

    pixel = np.column_stack((r, g, b)).astype('uint8')
    in_silico_image = Image.frombuffer('RGB', (w, h), pixel, 'raw', 'RGB', 0, 0)
    return in_silico_image


def make_ref_img(dna_path, strings, n_fold, n):
    """Extract fragment from FASTA and create reference image."""
    my_seq_record = SeqIO.read(dna_path, 'fasta')
    total_len = len(my_seq_record)
    my_seq_len = int(total_len / n_fold)

    start = n * my_seq_len
    end = (n + 1) * my_seq_len + 1
    sub_seq = my_seq_record.seq[start:end]

    seq_arr = np.array(list(str(sub_seq).upper()))
    reference_img = np.isin(seq_arr, strings).astype(np.float64)

    return reference_img


def build_affinity_table():
    """Build affinity table for 6-mer patterns."""
    table = np.zeros(64, dtype=np.float64)
    
    def check_pattern(pattern, search_region):
        """Check if pattern matches search_region or its reverse."""
        rev = search_region[::-1]
        return (search_region == pattern).all() or (rev == pattern).all()
    
    for i in range(64):
        pattern = np.array([(i >> (5 - k)) & 1 for k in range(6)])
        AT_count = np.sum(pattern)
        affinity = 0.0

        if AT_count == 1:
            affinity = 0.2
        elif AT_count == 2:
            high_affinity_patterns = [
                np.array([1, 1, 0, 0, 0, 0]),
                np.array([0, 1, 1, 0, 0, 0]),
                np.array([0, 0, 1, 1, 0, 0])
            ]
            if any(check_pattern(p, pattern) for p in high_affinity_patterns):
                affinity = 2.0
            else:
                affinity = 0.4
        elif AT_count == 3:
            high_affinity_patterns = [
                np.array([1, 1, 1, 0, 0, 0]),
                np.array([0, 1, 1, 1, 0, 0])
            ]
            low_affinity_patterns = [
                np.array([1, 0, 1, 0, 1, 0]),
                np.array([1, 0, 1, 0, 0, 1])
            ]
            if any(check_pattern(p, pattern) for p in high_affinity_patterns):
                affinity = 3.4
            elif any(check_pattern(p, pattern) for p in low_affinity_patterns):
                affinity = 0.6
            else:
                affinity = 2.2
        elif AT_count == 4:
            patterns = [
                (np.array([1, 1, 1, 1, 0, 0]), 18.1),
                (np.array([0, 1, 1, 1, 1, 0]), 18.1),
                (np.array([1, 1, 1, 0, 1, 0]), 9.8),
                (np.array([1, 1, 1, 0, 0, 1]), 6.7),
                (np.array([1, 1, 0, 1, 0, 1]), 5.4),
                (np.array([1, 1, 0, 1, 1, 0]), 8.4),
                (np.array([1, 1, 0, 0, 1, 1]), 6.2),
                (np.array([1, 0, 1, 1, 0, 1]), 2.4)
            ]
            for p, aff in patterns:
                if check_pattern(p, pattern):
                    affinity = aff
                    break
        elif AT_count == 5:
            patterns = [
                (np.array([1, 1, 1, 1, 1, 0]), 43.9),
                (np.array([1, 1, 1, 1, 0, 1]), 31.1),
                (np.array([1, 1, 1, 0, 1, 1]), 26.9)
            ]
            for p, aff in patterns:
                if check_pattern(p, pattern):
                    affinity = aff
                    break
        elif AT_count == 6:
            affinity = 100.0

        table[i] = affinity
    return table


AFFINITY_TABLE = build_affinity_table()


@njit(fastmath=True, cache=True, parallel=True)
def core_binding_simulation(reference_img, affinity_table, bl, Bl, be):
    n = len(reference_img)
    binding_site_img = np.zeros(n, dtype=np.float64)
    offset = bl // 2

    # 1. Pattern Matching & Affinity Check (Parallel processing)
    for i in prange(n - bl + 1):
        pattern_idx = 0
        for k in range(bl):
            if reference_img[i + k] > 0.5:
                pattern_idx += (1 << (bl - 1 - k))

        affinity = affinity_table[pattern_idx]

        if np.random.rand() > (1.0 - affinity / 100.0):
            binding_site_img[i + offset] = 1.0

    # 2. Binding Efficiency Check
    for i in range(n):
        if binding_site_img[i] == 1.0:
            if np.random.rand() > be:
                binding_site_img[i] = 0.0

    # 3. Steric Hindrance
    count = 0
    for i in range(n):
        if binding_site_img[i] == 1.0:
            count += 1

    if count > 0:
        bound_indices = np.zeros(count, dtype=np.int64)
        idx = 0
        for i in range(n):
            if binding_site_img[i] == 1.0:
                bound_indices[idx] = i
                idx += 1

        np.random.shuffle(bound_indices)

        for i in range(count):
            site_idx = bound_indices[i]
            if binding_site_img[site_idx] == 1.0:
                s = np.random.normal(Bl, 1.5)
                width_half = int(s / 2)

                start = max(0, site_idx - width_half)
                end = min(n, site_idx + width_half)

                for k in range(start, end):
                    binding_site_img[k] = 0.0

                binding_site_img[site_idx] = 1.0

    return binding_site_img


def find_binding_site(reference_img, binding_len, block_len, binding_efficiency):
    return core_binding_simulation(reference_img, AFFINITY_TABLE, binding_len, block_len, binding_efficiency)


def PSF(binding_site_img, fwhm):
    """Apply point spread function (Gaussian blur) to binding site image."""
    fwhm_nm = fwhm / 0.34
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    PSF_img = gaussian_filter1d(binding_site_img, sigma, mode='constant', cval=0.0)
    return PSF_img


def run_single_simulation(path, binding_len, block_len, efficiency, sim_id, n_fold, fragment_idx, base_output_dir):
    """
    Run a single simulation for one fragment.
    
    Args:
        path: FASTA file path
        binding_len: Binding length
        block_len: Block length
        efficiency: Binding efficiency
        sim_id: Simulation ID (for filename)
        n_fold: Number of fragments
        fragment_idx: Fragment index
        base_output_dir: Output directory for this FASTA
    
    Returns:
        PSF image array
    """
    reference_img = make_ref_img(path, ['A', 'T'], n_fold, fragment_idx)
    binding_site_img = find_binding_site(reference_img, binding_len, block_len, efficiency)
    PSF_img = PSF(binding_site_img, 382)

    folder_name = f"{n_fold}_{fragment_idx}"
    fragment_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(fragment_dir, exist_ok=True)

    save_path = os.path.join(fragment_dir, str(sim_id))
    np.save(save_path, PSF_img)

    return PSF_img




def read_sequence_info(fasta_path):
    """Read FASTA file and return sequence length."""
    print(f"  Reading sequence from {fasta_path}...")
    try:
        sequence = lib.read_fasta_sequence(fasta_path)
        seq_length = len(sequence)
        print(f"  Sequence length: {seq_length:,} bp")
        return seq_length
    except Exception as e:
        print(f"  Error reading FASTA file: {e}")
        sys.exit(1)


def determine_fragment_count(seq_length, n_fragments_arg):
    """Determine the number of fragments to use."""
    if n_fragments_arg is not None:
        n_fragments = n_fragments_arg
        print(f"  Using specified fragment count: {n_fragments}")
    else:
        n_fragments = 1
        print(f"  Using default fragment count: {n_fragments}")

    if n_fragments < 1:
        print("Error: n_fragments must be at least 1")
        sys.exit(1)

    return n_fragments


def determine_output_files(base_output_dir, fasta_basename, sim_times, overwrite):
    """Determine output filenames based on FASTA base name and simulation times.

    Args:
        base_output_dir: Output directory for this FASTA file
        fasta_basename: FASTA filename without extension
        sim_times: Number of simulations
        overwrite: If True, overwrite existing files without prompting

    Returns:
        tuple: (output_npy_file, output_tif_file)
    """
    os.makedirs(base_output_dir, exist_ok=True)

    output_tif_file = os.path.join(base_output_dir, f"{fasta_basename}_simRG{sim_times}.tif")
    output_npy_file = os.path.join(base_output_dir, f"{fasta_basename}_simRG{sim_times}.npy")

    existing_files = []
    if os.path.exists(output_npy_file):
        existing_files.append(output_npy_file)
    if os.path.exists(output_tif_file):
        existing_files.append(output_tif_file)

    if existing_files and not overwrite:
        print("\n⚠️  Warning: The following output files already exist:")
        for f in existing_files:
            print(f"   - {f}")
        print()

        while True:
            response = input("Do you want to overwrite these files? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("Continuing with overwrite...\n")
                break
            elif response in ['n', 'no']:
                print("Exiting without overwriting files for this FASTA.")
                return None, None
            else:
                print("Please enter 'y' or 'n'.")

    return output_npy_file, output_tif_file


def run_simulation(fasta_path, n_fragments, sim_times, base_output_dir):
    """Run the DNA binding site simulation for a single FASTA."""
    final_fragments_list = []

    print("  Simulation start with Numba (Parallel Acceleration)...")

    total_tasks = n_fragments * sim_times
    start_time = time.time()
    last_progress_time = start_time
    progress_interval = 30  # seconds

    for i in range(n_fragments):
        PSF_imgs_list = []
        for j in range(sim_times):
            current_time = time.time()

            if current_time - last_progress_time >= progress_interval:
                completed_tasks = i * sim_times + j
                progress_percent = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
                elapsed_time = current_time - start_time

                print(f"    Progress: {completed_tasks}/{total_tasks} "
                      f"({progress_percent:.1f}%) - Fragment {i + 1}/{n_fragments}, "
                      f"Simulation {j + 1}/{sim_times} - Elapsed: {elapsed_time:.1f}s")
                last_progress_time = current_time

            PSF_img = run_single_simulation(fasta_path, 6, 10, 1, j, n_fragments, i, base_output_dir)
            PSF_imgs_list.append(PSF_img)

        psf_stack = np.stack(PSF_imgs_list, axis=0)
        total_img = np.sum(psf_stack, axis=0)
        n_th_fragment_avg_img = total_img / sim_times
        final_fragments_list.append(n_th_fragment_avg_img)

    all_fragment_avg_img = np.concatenate(final_fragments_list)
    print("  Simulation completed!")
    return all_fragment_avg_img


def generate_map_image(fasta_path, all_fragment_avg_img, output_tif_file, seq_for_r, seq_for_g, bpp):
    """Generate and save the map image."""
    print("  Generating map image...")

    all_fragment_avg_img_2d = all_fragment_avg_img.reshape(1, -1)

    searched_position_list1 = lib.get_sequence_positions(fasta_path, seq_for_r)
    searched_position_list2 = lib.get_sequence_positions(fasta_path, seq_for_g)
    bin_size = bpp

    map_img = search_pos_list2map_img1(all_fragment_avg_img_2d,
                                       searched_position_list1,
                                       searched_position_list2,
                                       bin_size)
    map_img.save(output_tif_file)
    print(f"  Done. Saved map to {output_tif_file}")


def format_elapsed_time(elapsed_time):
    """Format elapsed time as hours, minutes, seconds."""
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return hours, minutes, seconds


def print_elapsed_time(hours, minutes, seconds, prefix=""):
    """Print formatted elapsed time."""
    print(f"{prefix}", end="")
    if hours > 0:
        print(f"{hours}h ", end="")
    if minutes > 0:
        print(f"{minutes}m ", end="")
    print(f"{seconds:.2f}s")


def process_single_fasta(fasta_path, sim_times, n_fragments_arg, overwrite, seq_for_r, seq_for_g, bpp):
    """Process a single FASTA file through the complete pipeline."""
    input_folder = os.path.dirname(fasta_path)
    fasta_basename = os.path.splitext(os.path.basename(fasta_path))[0]
    base_output_dir = os.path.join(input_folder, fasta_basename) 

    print(f"\n===== Processing FASTA: {fasta_basename} =====")
    print(f"  Output directory: {base_output_dir}")

    start_time = time.time()

    # 1. Read sequence length
    seq_length = read_sequence_info(fasta_path)

    # 2. Determine fragment count
    n_fragments = determine_fragment_count(seq_length, n_fragments_arg)

    # 3. Determine output files
    output_npy_file, output_tif_file = determine_output_files(
        base_output_dir, fasta_basename, sim_times, overwrite
    )

    # User declined to overwrite
    if output_npy_file is None or output_tif_file is None:
        print(f"  Skipping {fasta_basename} due to existing files.\n")
        return

    # 4. Run simulation
    all_fragment_avg_img = run_simulation(fasta_path, n_fragments, sim_times, base_output_dir)

    # 5. Save .npy file
    np.save(output_npy_file, all_fragment_avg_img)
    print(f"  Saved intensity array to {output_npy_file}")

    # 6. Generate map image
    generate_map_image(fasta_path, all_fragment_avg_img, output_tif_file, seq_for_r, seq_for_g, bpp)

    # 7. Print elapsed time
    elapsed_time = time.time() - start_time
    hours, minutes, seconds = format_elapsed_time(elapsed_time)
    print(f"  ⏱️  Execution time for {fasta_basename}: ", end="")
    print_elapsed_time(hours, minutes, seconds)
    print()


def find_fasta_files(folder_path):
    """Find all FASTA files in a folder."""
    fasta_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".fasta")
    ])
    return fasta_files


def main():
    """Main function to orchestrate the simulation workflow."""
    args = parse_args()
    input_path = args.input_path.replace('\\', '/')
    sim_times = args.sim_times
    n_fragments_arg = args.n_fragments
    overwrite = args.overwrite
    seq_for_r = args.seq_for_r
    seq_for_g = args.seq_for_g
    bpp = args.bpp

    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if os.path.isdir(input_path):
        # Process all .fasta files in folder
        fasta_files = find_fasta_files(input_path)

        if not fasta_files:
            print(f"Error: No .fasta files found in folder: {input_path}")
            sys.exit(1)

        print(f"Found {len(fasta_files)} FASTA file(s) in folder '{input_path}':")
        for f in fasta_files:
            print(f"  - {os.path.basename(f)}")
        print()

        for fasta_path in fasta_files:
            process_single_fasta(fasta_path, sim_times, n_fragments_arg, overwrite, seq_for_r, seq_for_g, bpp)

        print("All FASTA files processed.")
    else:
        # Process single FASTA file
        if not input_path.lower().endswith(('.fasta', '.fa')):
            print(f"Warning: Input file does not have .fasta or .fa extension: {input_path}")
        
        process_single_fasta(input_path, sim_times, n_fragments_arg, overwrite, seq_for_r, seq_for_g, bpp)


if __name__ == '__main__':
    main()