# DOM (Dual Optical Mapping)

DOM is a Python toolkit for analyzing dual-channel optical mapping data from TIFF images. It performs correlation analysis and alignment using PCC (Pearson Correlation Coefficient) and maligner methods.

## Features

- **GUI-based analysis**: Interactive visualization and matching results browser
- **Batch processing**: Process multiple TIFF files in a folder
- **Multiple alignment methods**: PCC-based and maligner-based alignment
- **FASTA to TIFF conversion**: Generate reference maps from FASTA sequence files (direct or simulation-based)

## Requirements

- Python 3.6+
- See `requirements.txt` for Python package dependencies
- `maligner_dp` executable (for maligner alignment method)

### About Maligner

This project includes a Python 3 port of the `maligner` alignment tool. The original `maligner` software was written in Python 2. We have ported it to Python 3 for compatibility with modern Python environments. The `maligner` code is included in the `maligner/` folder.

**Note on Licensing**: The `maligner` component is distributed under the GPL v3.0 license. If you modify or distribute this software, please ensure compliance with the GPL license terms. See the `maligner/` folder for the original license information and source code details.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jolaboffice/DOM.git
cd DOM
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up `maligner_dp`:
   - The `maligner_dp` executable is located in `maligner/build/bin/maligner_dp` (pre-built for Linux only)
   - **Platform Note**: The included executable is for Linux. For macOS or other platforms, you will need to rebuild from source (see `maligner/README.md`)
   - Ensure `maligner_dp` is available in your PATH, or specify its path using the `--maligner-path` option

## Usage

### Quick Start

1. **Navigate to the dom folder**:
```bash
cd dom
```

2. **Generate reference map** (if you have a FASTA file):
```bash
python seq2RGmap.py <fasta_file> --output ref_map/RG.tif
# or
python simulation.py <fasta_file> --output ref_map/RG.tif
```

3. **Run main GUI application**:
```bash
python DOM.py <data_folder> [--map MAP_FILE] [--bpp BPP] [--shift-window WINDOW] [--start-bpp START_BPP] [--end-bpp END_BPP] [--maligner-path PATH]
```

Example:
```bash
cd dom
python seq2RGmap.py MG1655.fasta --output ref_map/RG.tif
python DOM.py ./tif_data --map ref_map/RG.tif
```

### Main GUI Application

`DOM.py` automatically runs `DOM_match.py`, `pcc_tifFolder.py`, `mapping_tifFolder.py`, and `DOM_make_info.py` internally as needed. You can also run these scripts individually for batch processing without GUI.

```bash
python DOM.py <data_folder> \
  [--map MAP_FILE] \
  [--bpp BPP] \
  [--shift-window WINDOW] \
  [--start-bpp START_BPP] \
  [--end-bpp END_BPP] \
  [--maligner-path PATH]
```

### Generate Reference Map from FASTA

Generate reference TIFF map from FASTA sequence file. Two methods are available:

**Method 1: Direct sequence pattern matching** (`seq2RGmap.py`):
```bash
python seq2RGmap.py <fasta_file> \
  [--output OUTPUT.tif] \
  [--bpp BPP] \
  [--seq-for-R SEQ] \
  [--seq-for-G SEQ]
```

**Method 2: Simulation-based** (`sim2RGmap.py`):
Generates TIFF map using in-silico simulation of DNA binding sites:
```bash
python sim2RGmap.py <fasta_file> <sim_times>\
  [--output OUTPUT.tif] \
  [--bpp BPP] \
  [--n-fragments N] \
  [--sim-times N] \
  [--binding-len LEN] \
  [--seq-for-R SEQ] \
  [--seq-for-G SEQ]
  [--overwrite]
```

### Standalone Tools (Optional)

These scripts are automatically called by `DOM.py`, but can be run independently:

**Batch Processing (CSV output only)**:
```bash
python DOM_match.py <data_folder> [--map MAP_FILE] [--file FILE] [--bpp BPP] [--start-bpp START_BPP] [--end-bpp END_BPP] [--shift-window WINDOW]
```

**Calculate PCC for TIFF Folder**:
```bash
python pcc_tifFolder.py <data_folder> [--map MAP_FILE] [--bpp BPP] [--start-bpp START_BPP] [--end-bpp END_BPP]
```

**Generate Maligner Logs**:
```bash
python mapping_tifFolder.py <data_folder> \
  [--map MAP_OR_MAPS_FILE] \
  [--bpp BPP] \
  [--maligner-path PATH]
```
Note: The `--map` option accepts both `.tif` and `.maps` files.

**Generate Metadata Files**:
```bash
python DOM_make_info.py <data_folder> [--bpp BPP]
```

## File Structure

**Main Scripts:**
- `DOM.py` - Main GUI application (automatically calls other scripts as needed)
- `seq2RGmap.py` - Generate reference TIFF map from FASTA file (direct sequence pattern matching)
- `sim2RGmap.py` - Generate reference TIFF map from FASTA file (simulation-based method)

**Supporting Scripts** (called automatically by `DOM.py`, but can be run standalone):
- `DOM_match.py` - Batch matching results generator (CSV output)
- `pcc_tifFolder.py` - PCC calculation for TIFF folder
- `mapping_tifFolder.py` - Maligner log generator
- `DOM_make_info.py` - Metadata file generator

**Core Modules:**
- `DOM_lib.py` - Core utility functions
- `DOM_ui.py` - UI rendering and event handling
- `DOM_init.py` - Configuration and argument parsing
- `DOM_constants.py` - Default configuration constants

## Output Files

- `_matching_results.csv` - Matching results summary for each TIFF file
- `_pcc.log` - PCC calculation results
- `_mal.log` - Maligner alignment results
- `*.info` - Metadata files (molecule length, BPP)

## Citation

[Add citation information if applicable]
