#!/usr/bin/env python
"""
DOM_ui.py

UI rendering and event handling functions for DOM.py.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from scipy import signal
import DOM_lib as lib

BPP = None
file_list_state = None
control_state = None
g_pane = None
r_pane = None
ref_image = None
mol_image = None
X_graph = None
tif_files = None
RGmap = None
main_figure = None

def get_tif_files():
    """Get tif_files list from global state."""
    return tif_files if tif_files is not None else []

def init_ui_globals(bpp, file_state, control, panes, files, rgmap, main_fig=None):
    """Initialize global variables from DOM.py"""
    global BPP, file_list_state, control_state, g_pane, r_pane, ref_image, mol_image, X_graph, tif_files, RGmap, main_figure
    BPP = bpp
    file_list_state = file_state
    control_state = control
    g_pane, r_pane, ref_image, mol_image, X_graph = panes
    tif_files = files
    RGmap = rgmap
    main_figure = main_fig

def render_results_table(ax_table, df_display, current_row_idx=None):
    """Render matching results table in the right panel"""
    ax_table.clear()
    ax_table.axis('off')
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    
    if df_display.empty:
        ax_table.text(0.5, 0.5, "No results", ha="center", va="center", fontsize=12)
        return
    
    ax_table.text(0.5, 0.98, "Matching Results", ha="center", va="top", fontsize=12, fontweight="bold")
    
    # Determine column header name: 'Reference' for hDOM.py, 'method' for DOM.py
    # Check if 'method' column contains reference names (like 'chr1', 'chr2') or method names ('pcc', 'maligner')
    method_col_name = "method"
    if 'method' in df_display.columns and len(df_display) > 0:
        # Check first few non-empty method values
        method_values = df_display['method'].dropna().head(5).tolist()
        # If values look like reference names (contain 'chr' or are chromosome-like), use 'Reference'
        # Otherwise use 'method' (for 'pcc', 'maligner', etc.)
        if method_values:
            first_val = str(method_values[0]).lower()
            # Reference names typically contain 'chr' or are chromosome identifiers
            if 'chr' in first_val or (len(first_val) > 0 and first_val[0].isdigit()):
                method_col_name = "ref"
    
    columns = [
        ("rank", "rank", 0.05, "left"),
        (method_col_name, "method", 0.12, "left"),
        ("start_bp", "start_bp", 0.30, "right"),
        ("scale", "scale", 0.40, "right"),
        ("cc_r", "cc_r", 0.50, "right"),
        ("cc_g", "cc_g", 0.60, "right"),
        ("cc_rg", "cc_rg", 0.70, "right"),
        ("cc_rg2", "cc_rg2", 0.80, "right"),
        ("misc", "misc", 0.90, "right")
    ]
    
    y_start = 0.92
    line_height = 0.04
    
    for label, _, pos, align in columns:
        header_ha = "left" if align == "left" else "right"
        ax_table.text(pos, y_start, label, ha=header_ha, va="top",
                      fontsize=9, fontweight="bold")
    
    # Draw data rows
    max_rows = int((y_start - 0.05) / line_height)
    n_rows = min(len(df_display), max_rows)
    
    # Store table rows for click detection
    file_list_state['table_rows'] = []
    
    # Get current row index for highlighting
    if current_row_idx is None:
        current_row_idx = file_list_state.get('current_table_row_idx')
    
    for row_idx in range(n_rows):
        y = y_start - (row_idx + 1) * line_height
        row = df_display.iloc[row_idx]
        rank_val = row['rank']
        
        # Highlight current row (not based on rank, but based on row index)
        is_current = (current_row_idx is not None and row_idx == current_row_idx)
        if is_current:
            ax_table.axhspan(y - 0.015, y + 0.01, xmin=0.01, xmax=0.99, 
                           color="lightblue", zorder=0, alpha=0.3)
        
        # Format values
        method_raw = str(row['method'])
        method = 'mal' if method_raw == 'maligner' else ('cc' if method_raw == 'pcc' else method_raw)
        start_bp = f"{int(row['start_bp']):,}" if pd.notna(row['start_bp']) else "N/A"
        scale = f"{row['scale']:.3f}" if pd.notna(row['scale']) else "N/A"
        cc_rg = f"{row['cc_rg']:.3f}" if pd.notna(row['cc_rg']) else "N/A"
        cc_r = f"{row['cc_r']:.3f}" if pd.notna(row['cc_r']) else "N/A"
        cc_g = f"{row['cc_g']:.3f}" if pd.notna(row['cc_g']) else "N/A"
        cc_rg2 = f"{row['cc_rg2']:.3f}" if pd.notna(row['cc_rg2']) else "N/A"
        # Rank already 1-based in matching_result.csv
        rank = f"{int(rank_val)}" if pd.notna(rank_val) else "N/A"
        misc_val = row.get('misc')
        if pd.notna(misc_val):
            if method_raw == 'pcc':
                misc_str = f"{misc_val:.0f}"
            else:
                misc_str = f"{misc_val:.2f}"
        else:
            misc_str = ""

        values = [rank, method, start_bp, scale, cc_r, cc_g, cc_rg, cc_rg2, misc_str]
        color = "navy" if is_current else "black"
        weight = "bold" if is_current else "normal"
        
        for (label, _, pos, align), val in zip(columns, values):
            ha = "right" if align == "right" else "left"
            ax_table.text(pos, y, val, ha=ha, va="top",
                         fontsize=8, color=color, weight=weight)
        
        # Store clickable region: (row_idx, y_top, y_bottom, rank_val)
        file_list_state['table_rows'].append((row_idx, y + 0.01, y - 0.015, rank_val))
    
    # Show overflow message if there are more rows than can be displayed
    if len(df_display) > max_rows:
        ax_table.text(0.5, 0.02, f"... and {len(df_display) - max_rows} more",
                     ha="center", va="bottom", fontsize=6, style="italic", color="gray")

def render_file_list(ax_list, tif_files, current_file):
    """Render file list panel on the left side"""
    ax_list.clear()
    ax_list.set_xticks([])
    ax_list.set_yticks([])
    ax_list.set_xlim(0, 1)
    ax_list.set_ylim(0, 1)
    ax_list.set_facecolor("white")
    
    if not tif_files:
        ax_list.text(0.02, 0.98, "No files", va="top", ha="left", fontsize=10)
        return
    
    # Get folder path
    try:
        common_dir = os.path.dirname(os.path.commonpath([os.path.abspath(f) for f in tif_files]))
        if not common_dir or common_dir == os.sep:
            common_dir = os.path.dirname(os.path.abspath(current_file)) if current_file else ""
        folder_display = os.path.basename(common_dir) if common_dir else "Files"
    except (ValueError, OSError):
        common_dir = os.path.dirname(os.path.abspath(current_file)) if current_file else ""
        folder_display = os.path.basename(common_dir) if common_dir else "Files"
    
    # Header
    n = len(tif_files)
    header = f"{folder_display} ({n} files)"
    ax_list.text(0.02, 0.98, header, va="top", ha="left", fontsize=10, fontweight="bold")
    
    # File list
    y_top = 0.90
    line_h = 0.04
    max_lines = int((y_top - 0.05) / line_h)
    
    # Find current file index
    current_idx = 0
    if current_file:
        try:
            current_idx = tif_files.index(current_file)
            file_list_state['current_index'] = current_idx
        except ValueError:
            current_idx = file_list_state.get('current_index', 0)
    else:
        current_idx = file_list_state.get('current_index', 0)
    
    # Keep view stable - only adjust if current file is outside visible range
    # Get previous view range if available
    prev_start = file_list_state.get('view_start_idx')
    prev_end = file_list_state.get('view_end_idx')
    
    # Check if current file is in the previous view
    if prev_start is not None and prev_end is not None and prev_start <= current_idx < prev_end:
        # Keep the same view - don't scroll
        start_idx = prev_start
        end_idx = prev_end
    else:
        # Current file is outside view or no previous view, center around it
        start_idx = max(0, current_idx - max_lines // 2)
        end_idx = min(n, start_idx + max_lines)
    
    # Store current view range
    file_list_state['view_start_idx'] = start_idx
    file_list_state['view_end_idx'] = end_idx
    
    file_list_state['file_rows'] = []
    
    for k, idx in enumerate(range(start_idx, end_idx)):
        y = y_top - k * line_h
        base = os.path.basename(tif_files[idx])
        is_current = (idx == current_idx)
        color = "navy" if is_current else "black"
        weight = "bold" if is_current else "normal"
        
        if is_current:
            ax_list.axhspan(y - 0.02, y + 0.01, color="lightblue", zorder=0, alpha=0.3)
        
        txt = f"{idx+1:>3}. {base}"
        ax_list.text(0.03, y, txt, va="top", ha="left", fontsize=9, color=color, weight=weight)
        
        # Store clickable region: (index, y_top, y_bottom)
        # Make clickable area slightly larger for easier clicking
        file_list_state['file_rows'].append((idx, y + 0.015, y - 0.025))

def show_tiff_images(shift, image_width, image_height, RGmap_image, mol_img):
    ref_tiff_start = shift
    ref_tiff_end = shift + image_width
    ref_img = RGmap_image.crop((ref_tiff_start, 0, ref_tiff_end, image_height))
    ref_image.clear()
    ref_image.imshow(ref_img, aspect='auto')
    ref_image.set_yticks([])
    ref_image.axis('off')
    mol_image.clear()
    mol_image.imshow(mol_img, aspect='auto')
    mol_image.axis('off')

def show_graphs_compare_ref_mol(s_clip1r, s_clip1g, d_clip2r, d_clip2g, image_width, w2, scale, shift):
    x_ref, x_mol = _prepare_x_axes(image_width, w2, scale, len(s_clip1g))
    _plot_reference_profiles(x_ref, s_clip1g, s_clip1r)
    _plot_molecule_profiles(x_mol, d_clip2g, d_clip2r)
    _plot_molecule_peaks(d_clip2g, scale)
    plt.draw()

def _prepare_x_axes(image_width, w2, scale, ref_len):
    X_graph.clear()
    X_graph.set_xlim(0, image_width * BPP / 1e3)
    x_ref = np.arange(0, image_width) * BPP / 1e3
    x_mol = np.arange(0, w2) * BPP / 1e3 * scale
    if len(x_ref) > ref_len:
        x_ref = x_ref[:ref_len]
    return x_ref, x_mol

def _plot_reference_profiles(x_ref, s_clip1g, s_clip1r):
    X_graph.plot(x_ref, 0.85 + 0.15 * s_clip1g, 'c', linewidth=0.8)
    X_graph.plot(x_ref, 0.35 + 0.35 * s_clip1r, 'm', linewidth=0.8)

def _plot_molecule_profiles(x_mol, d_clip2g, d_clip2r):
    X_graph.plot(x_mol, 0.70 + 0.15 * d_clip2g, 'g', linewidth=1.0)
    X_graph.plot(x_mol, 0.00 + 0.35 * d_clip2r, 'r', linewidth=1.0)

def _plot_molecule_peaks(d_clip2g, scale):
    peaks, _ = signal.find_peaks(d_clip2g)
    if len(peaks) > 0:
        X_graph.plot(scale * peaks * BPP / 1e3,
                     0.70 + 0.15 * d_clip2g[peaks], 'x')

def render_header(ax_header, map_path, data_folder_path):
    """Render header with IGV-style map and data folder selection"""
    ax_header.clear()
    ax_header.axis('off')
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    
    # Format display text - show full path or basename
    if map_path:
        map_display = os.path.basename(map_path)
        map_full_path = map_path
    else:
        map_display = "Select reference map"
        map_full_path = None
    
    if data_folder_path:
        data_display = os.path.basename(data_folder_path)
        data_full_path = data_folder_path
    else:
        data_display = "Select data folder"
        data_full_path = None
    
    # IGV-style: Label + Value box + More button
    # Ref map section
    ax_header.text(0.02, 0.5, "Ref map:", ha="left", va="center", fontsize=10, fontweight="bold")
    
    # Value box (like IGV input field) - increased height
    map_box_x = 0.15
    map_box_width = 0.30
    map_box_y_bottom = 0.0
    map_box_y_top = 1.2
    
    # Draw value box background
    map_box = plt.Rectangle((map_box_x, map_box_y_bottom), map_box_width, map_box_y_top - map_box_y_bottom,
                            facecolor='white', edgecolor='gray', linewidth=1, transform=ax_header.transAxes)
    ax_header.add_patch(map_box)
    
    # Text inside box
    map_text = ax_header.text(map_box_x + 0.01, 0.5, map_display, ha="left", va="center", fontsize=9,
                              transform=ax_header.transAxes, clip_on=True)
    
    # Store box region (but not clickable)
    file_list_state['map_box'] = (map_box_x, map_box_x + map_box_width, map_box_y_bottom, map_box_y_top)
    file_list_state['map_more_btn'] = None  # No clickable button
    
    ax_header.text(0.50, 0.5, "Data folder:", ha="left", va="center", fontsize=10, fontweight="bold")
    
    # Value box - increased height
    data_box_x = 0.63
    data_box_width = 0.30
    data_box_y_bottom = 0.0
    data_box_y_top = 1.2
    
    # Draw value box background
    data_box = plt.Rectangle((data_box_x, data_box_y_bottom), data_box_width, data_box_y_top - data_box_y_bottom,
                             facecolor='white', edgecolor='gray', linewidth=1, transform=ax_header.transAxes)
    ax_header.add_patch(data_box)
    
    # Text inside box
    data_text = ax_header.text(data_box_x + 0.01, 0.5, data_display, ha="left", va="center", fontsize=9,
                               transform=ax_header.transAxes, clip_on=True)
    
    # More button removed - display only, no clickable functionality
    # Store box region (but not clickable)
    file_list_state['data_box'] = (data_box_x, data_box_x + data_box_width, data_box_y_bottom, data_box_y_top)
    file_list_state['data_more_btn'] = None
    
    file_list_state['map_text'] = map_text
    file_list_state['data_text'] = data_text
    file_list_state['map_full_path'] = map_full_path
    file_list_state['data_full_path'] = data_full_path

def handle_header_click(event, ax_header, ax_list):
    """Header display only; clicks do nothing."""
    return False

def handle_file_list_click(event, ax_list):
    if event.inaxes != ax_list:
        return False
    render_file_list(ax_list, file_list_state.get('tif_files', []), None)
    rows = file_list_state.get('file_rows', [])
    if not rows or event.ydata is None:
        return False
    clicked_idx = None
    for file_idx, y_top, y_bot in rows:
        if y_bot <= event.ydata <= y_top:
            clicked_idx = file_idx
            break
    if clicked_idx is None:
        min_dist = float('inf')
        for file_idx, y_top, y_bot in rows:
            row_center = (y_top + y_bot) / 2
            dist = abs(event.ydata - row_center)
            if dist < min_dist:
                min_dist = dist
                clicked_idx = file_idx
    if clicked_idx is not None:
        file_list_state['clicked_index'] = clicked_idx
        plt.draw()
        return True
    return False

def handle_table_click(event):
    ax_table = file_list_state.get('ax_table')
    if event.inaxes != ax_table:
        return False
    rows = file_list_state.get('table_rows', [])
    if not rows or event.ydata is None:
        return False
    for row_idx, y_top, y_bot, rank_val in rows:
        if y_bot <= event.ydata <= y_top:
            file_list_state['clicked_row_idx'] = row_idx
            return True
    return False

def handle_xgraph_click(event):
    """Handle clicks on the X_graph (bottom profile plot)"""
    return False

def Draw_Whole_Reference(RGmap):
    map_r, map_g, map_peaks = lib.tif2RGprofile(RGmap)
    map_peaks[-1] += map_peaks[0]
    clip1r = lib.normalize(map_r)
    clip1g = lib.normalize(map_g)
    x1 = np.arange(0, len(clip1r)) * BPP / 1e6
    L = len(clip1r)
    l = int(L / 10)
    x2 = x1.max() + np.arange(0, l) * BPP / 1e6
    g_pane.plot(x1, 0.95 + 0.08 * clip1g, 'c', linewidth=0.5)
    g_pane.plot(x2, 0.95 + 0.08 * clip1g[0:l], '0.3', linewidth=0.5)
    r_pane.plot(x1, 0.95 + 0.08 * clip1r, 'm', linewidth=0.5)
    r_pane.plot(x2, 0.95 + 0.08 * clip1r[0:l], '0.3', linewidth=0.5)
    x = x2.max()
    g_pane.set_xlim(0, x)
    g_pane.set_yticks([])
    r_pane.set_xlim(0, x)
    r_pane.set_yticks([])

def close_figure_safely(fig):
    """Safely close a matplotlib figure."""
    if fig is not None:
        try:
            if plt.fignum_exists(fig.number):
                try:
                    fig.canvas.stop_event_loop()
                except (AttributeError, RuntimeError):
                    pass
                plt.close(fig)
        except (AttributeError, RuntimeError, ValueError):
            pass

def clear_alignment_lines():
    """Clear alignment lines from g_pane and r_pane."""
    for pane in (g_pane, r_pane):
        for line in list(pane.lines):
            ydata = line.get_ydata()
            if len(ydata) > 0 and (ydata[0] < 0.94 or ydata[-1] < 0.94):
                line.remove()

def plot_alignment(x2, d_clip2g, d_clip2r, direction, data_image, shift_px, image_width, h1, RGmap_image, s_clip1r, s_clip1g, d_clip2r_arr, d_clip2g_arr, w2, scale):
    """Plot alignment visualization."""
    clear_alignment_lines()
    fixed_y_g = 0.87
    fixed_y_r = 0.87
    g_pane.plot(x2, fixed_y_g + 0.08 * d_clip2g, 'm', linewidth=0.8)
    r_pane.plot(x2, fixed_y_r + 0.08 * d_clip2r, 'k', linewidth=0.8)
    mol_img = data_image if direction > 0 else data_image.rotate(180)
    show_tiff_images(shift_px, image_width, h1, RGmap_image, mol_img)
    show_graphs_compare_ref_mol(s_clip1r, s_clip1g, d_clip2r_arr, d_clip2g_arr, image_width, w2, scale, shift_px)

def update_table_highlight(df_display, method, start_bp, scale):
    """Update table highlight using already-built df_display to avoid redundant processing."""
    current_row_idx = None
    # Support both 'method' and 'ref' column names
    method_col = 'method' if 'method' in df_display.columns else 'ref'
    for disp_idx, disp_row in df_display.iterrows():
        if (disp_row[method_col] == method and
                abs(disp_row['start_bp'] - start_bp) < 1 and
                abs(disp_row['scale'] - scale) < 0.001):
            current_row_idx = disp_idx
            break
    file_list_state['current_table_row_idx'] = current_row_idx
    display_cols = ['rank', method_col, 'start_bp', 'scale', 'cc_r', 'cc_g', 'cc_rg', 'cc_rg2', 'misc']
    render_results_table(
        file_list_state.get('ax_table'),
        df_display[display_cols],
        current_row_idx=current_row_idx
    )
    try:
        plt.draw()
    except Exception:
        pass

def update_current_file_list_display(current_file):
    """Update file list display."""
    ax_list = file_list_state.get('ax_list')
    if ax_list is None:
        return
    render_file_list(ax_list, file_list_state.get('tif_files', []), current_file)
    try:
        plt.draw()
    except Exception:
        pass

def render_results_panel(df_display, current_row_idx=None):
    """Render results panel."""
    ax_table = file_list_state.get('ax_table')
    if ax_table is None:
        return
    # Support both 'method' and 'ref' column names
    method_col = 'method' if 'method' in df_display.columns else 'ref'
    display_cols = ['rank', method_col, 'start_bp', 'scale', 'cc_r', 'cc_g', 'cc_rg', 'cc_rg2', 'misc']
    render_results_table(
        ax_table,
        df_display[display_cols],
        current_row_idx=current_row_idx
    )
    try:
        plt.draw()
    except Exception:
        pass

def control_RGmap_viewer(shift, w2, direction, data_file, method, scale, score, x2, r_pos, g_pos, d_clip2r, d_clip2g, cc_rg, cc_r, cc_g, start_bp=None):
    """Control the RG map viewer loop."""
    if start_bp is None:
        start_bp = shift * BPP
    control_state['action'] = None
    global main_figure
    while True:
        if main_figure is not None and main_figure.canvas is not None:
            try:
                main_figure.canvas.flush_events()
            except:
                pass
        time.sleep(0.1)
        
        clicked_idx = file_list_state.get('clicked_index')
        if clicked_idx is not None:
            return 'file_clicked'
        
        clicked_row_idx = file_list_state.get('clicked_row_idx')
        if clicked_row_idx is not None:
            return 'row_clicked'
        
        action = control_state.get('action')
        if not action:
            continue
        control_state['action'] = None
        if action == 'exit':
            close_figure_safely(main_figure)
            os._exit(0)
        if action == 'uncertain':
            return 'continue'
        if action == 'save':
            return 'continue'
        if action == 'next':
            return 'continue'
        if action == 'prev':
            return 'prev'

def create_figure():
    """Create and configure the main matplotlib figure."""
    plt.ion()
    dpi = 100
    fig_width = 2000 / dpi
    fig_height = 600 / dpi
    return plt.figure(figsize=(fig_width, fig_height))

def setup_layout(fig, tif_files_list):
    """Set up the GridSpec layout and create all subplot axes."""
    gs = GridSpec(6, 3, width_ratios=[0.12, 0.48, 0.40],
                  height_ratios=[0.3, 1, 1, 0.5, 0.5, 5],
                  hspace=0.4, wspace=0.12, left=0.01, right=0.99, top=0.99, bottom=0.15)

    ax_header = fig.add_subplot(gs[0, :])
    file_list_state['ax_header'] = ax_header
    
    g_pane_local = fig.add_subplot(gs[1, 1])
    r_pane_local = fig.add_subplot(gs[2, 1])
    ref_image_local = fig.add_subplot(gs[3, 1])
    mol_image_local = fig.add_subplot(gs[4, 1])
    X_graph_local = fig.add_subplot(gs[5, 1])
    
    globals()['g_pane'] = g_pane_local
    globals()['r_pane'] = r_pane_local
    globals()['ref_image'] = ref_image_local
    globals()['mol_image'] = mol_image_local
    globals()['X_graph'] = X_graph_local
    
    ax_list = fig.add_subplot(gs[1:, 0])
    file_list_state['ax_list'] = ax_list
    file_list_state['tif_files'] = tif_files_list

    ax_table = fig.add_subplot(gs[1:, 2])
    ax_table.axis('off')
    file_list_state['ax_table'] = ax_table
    
    return ax_header, (g_pane_local, r_pane_local, ref_image_local, mol_image_local, X_graph_local), ax_list, ax_table

def setup_page_buttons(fig, ax_list, tif_files_list):
    """Add Prev/Next page buttons below the file list."""
    list_pos = ax_list.get_position()
    control_height = 0.05
    spacing = 0.01
    y_pos = max(0.02, list_pos.y0 - control_height - spacing)
    ctrl_width = list_pos.width
    ax_prev = fig.add_axes([list_pos.x0, y_pos, ctrl_width * 0.45, control_height])
    ax_next = fig.add_axes([list_pos.x0 + ctrl_width * 0.55, y_pos, ctrl_width * 0.45, control_height])
    btn_prev = Button(ax_prev, "Prev Page")
    btn_next = Button(ax_next, "Next Page")
    file_list_state['btn_page_prev'] = btn_prev
    file_list_state['btn_page_next'] = btn_next

    max_lines = int((0.90 - 0.05) / 0.04)
    file_list_state['page_size'] = max_lines

    def _change_page(delta):
        n = len(tif_files_list)
        if n == 0:
            return
        page_size = file_list_state.get('page_size', max_lines)
        start_idx = file_list_state.get('view_start_idx')
        if start_idx is None:
            start_idx = 0
        new_start = start_idx + delta * page_size
        new_start = max(0, min(max(0, n - page_size), new_start))
        file_list_state['view_start_idx'] = new_start
        file_list_state['view_end_idx'] = min(n, new_start + page_size)

        # Set the first file of the new page as the current file and trigger visualization
        file_list_state['current_index'] = new_start
        current_file = tif_files_list[new_start]
        render_file_list(file_list_state['ax_list'], tif_files_list, current_file)
        
        # Trigger file click to visualize the first result of the first molecule on the new page
        file_list_state['clicked_index'] = new_start
        
        fig.canvas.draw_idle()

    btn_prev.on_clicked(lambda _event: _change_page(-1))
    btn_next.on_clicked(lambda _event: _change_page(1))

def setup_event_handlers(fig, ax_header, ax_list, close_figure_callback):
    """Set up event handlers for mouse and window events."""
    def on_click(event):
        if event.button != 1:
            return
        if handle_header_click(event, ax_header, ax_list):
            return
        if handle_file_list_click(event, ax_list):
            return
        if handle_table_click(event):
            return
        handle_xgraph_click(event)
    
    def on_close(event):
        """Handle window close event safely."""
        try:
            if hasattr(fig, '_closing'):
                return
            fig._closing = True
            close_figure_callback(fig)
            os._exit(0)
        except Exception:
            os._exit(0)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)

def setup_main_figure(tif_files_list, map_path, data_folder_path, rgmap, bpp, file_state, control):
    """Initialize the matplotlib layout, event handlers, and global panes."""
    global main_figure, BPP, file_list_state, control_state
    BPP = bpp
    file_list_state = file_state
    control_state = control
    fig = create_figure()
    main_figure = fig
    ax_header, panes, ax_list, ax_table = setup_layout(fig, tif_files_list)
    
    init_ui_globals(BPP, file_list_state, control_state, panes, tif_files_list, rgmap, main_figure)
    render_header(ax_header, map_path, data_folder_path)
    render_file_list(ax_list, tif_files_list, None)
    
    setup_page_buttons(fig, ax_list, tif_files_list)
    setup_event_handlers(fig, ax_header, ax_list, close_figure_safely)
    
    Draw_Whole_Reference(rgmap)
    return fig

def update_file_display(fig, idx, tif_files_list):
    """Update file display for the given index."""
    ax_list = file_list_state.get('ax_list')
    if ax_list is None:
        return
    tif_file = tif_files_list[idx]
    file_list_state['current_index'] = idx
    render_file_list(ax_list, tif_files_list, tif_file)
    fig.canvas.draw()
    fig.canvas.manager.set_window_title(tif_file)

def build_display_table(df, bpp):
    """Build display table from DataFrame."""
    # Support both 'method' and 'ref' column names
    method_col = 'method' if 'method' in df.columns else 'ref'
    required_cols = [method_col, 'shift', 'direction', 'scale', 'cc_r', 'cc_g', 'cc_rg', 'cc_rg2', 'rank', 'misc']
    df_display = df[required_cols].copy()
    # Rename 'ref' to 'method' internally for consistency
    if method_col == 'ref':
        df_display = df_display.rename(columns={'ref': 'method'})
    df_display['start_bp'] = df_display['shift'] * bpp
    df_display = df_display.drop(columns=['shift']).rename(columns={'start_bp': 'start_bp'})
    df_display = df_display.drop(columns=['rank'])
    df_display.insert(0, 'rank', range(1, len(df_display) + 1))
    return df_display

def find_clicked_row_index(df):
    """Find the index of clicked row in DataFrame."""
    clicked_row_idx = file_list_state.get('clicked_row_idx')
    if clicked_row_idx is None:
        return None
    df_sorted = df.sort_values(by='cc_rg2', ascending=False).reset_index(drop=True)
    if clicked_row_idx >= len(df_sorted):
        return None
    clicked_row_data = df_sorted.iloc[clicked_row_idx]
    for idx in df.index:
        if (df.loc[idx, 'method'] == clicked_row_data['method'] and
                abs(df.loc[idx, 'shift'] - clicked_row_data['shift']) < 1 and
                abs(df.loc[idx, 'scale'] - clicked_row_data['scale']) < 0.001):
            file_list_state['clicked_row_idx'] = None
            return idx
    return None

def maybe_jump_to_clicked_row(df, df_indices, current_index):
    """Jump to clicked row if available."""
    clicked_idx = find_clicked_row_index(df)
    if clicked_idx is None:
        return current_index, False
    new_position = df_indices.index(clicked_idx)
    return new_position, new_position != current_index

def check_file_click():
    """Check if a file was clicked and return the clicked index, or None."""
    if file_list_state is None:
        return None
    clicked_idx = file_list_state.get('clicked_index')
    files = get_tif_files()
    if clicked_idx is not None and 0 <= clicked_idx < len(files):
        file_list_state['clicked_index'] = None
        return clicked_idx
    return None

def process_file_at_index(fig, idx, count, tif_files_list, total_files, process_callback):
    """Process a single file at the given index and return the result."""
    update_file_display(fig, idx, tif_files_list)
    tif_file = tif_files_list[idx]
    if total_files > 0:
        print(f"{tif_file}, ({count}/{total_files})")
    else:
        print(tif_file)
    return process_callback(tif_file)

def handle_navigation_result(result, idx, count, tif_files_list):
    """Handle navigation result and return (new_idx, new_count, should_continue, should_exit)."""
    if result == 'prev':
        new_idx = max(0, idx - 1) if idx > 0 else idx
        return new_idx, count, True, False
    elif result == 'exit':
        return idx, count, False, True
    elif result == 'file_clicked':
        clicked_idx = file_list_state.get('clicked_index')
        if clicked_idx is not None and 0 <= clicked_idx < len(tif_files_list):
            file_list_state['clicked_index'] = None
            return clicked_idx, 1, True, False
        return idx, count, True, False
    elif result == 'row_clicked':
        return idx, count, True, False
    else:
        return idx + 1, count, True, False

def process_initial_files(fig, tif_files_list, process_callback):
    """Process all files initially, then return the final index."""
    ax_list = file_list_state.get('ax_list')
    ax_table = file_list_state.get('ax_table')
    if ax_list is None or ax_table is None:
        raise RuntimeError("File list or table axis not initialized.")
    
    count = 1
    idx = 0
    total_files = len(tif_files_list)
    
    while idx < len(tif_files_list):
        clicked_idx = check_file_click()
        if clicked_idx is not None:
            idx = clicked_idx
        
        result = process_file_at_index(fig, idx, count, tif_files_list, total_files, process_callback)
        count += 1
        
        new_idx, new_count, should_continue, should_exit = handle_navigation_result(result, idx, count, tif_files_list)
        if should_exit:
            close_figure_safely(fig)
            os._exit(0)
        if not should_continue:
            break
        idx = new_idx
        count = new_count
    
    if idx >= len(tif_files_list):
        idx = len(tif_files_list) - 1
    
    if idx >= 0 and idx < len(tif_files_list):
        result = process_file_at_index(fig, idx, count, tif_files_list, total_files, process_callback)
        if result == 'exit':
            close_figure_safely(fig)
            os._exit(0)
    
    return idx

def handle_clicked_file(fig, idx, tif_files_list, process_callback):
    """Handle a clicked file and return (new_idx, should_exit)."""
    clicked_idx = check_file_click()
    if clicked_idx is not None:
        idx = clicked_idx
        update_file_display(fig, idx, tif_files_list)
        result = process_callback(tif_files_list[idx])
        if result == 'exit':
            return idx, True
    return idx, False

def handle_control_action(fig, idx, action, tif_files_list, process_callback):
    """Handle a control action and return (new_idx, should_exit)."""
    if action == 'exit':
        return idx, True
    elif action == 'prev':
        if idx >= len(tif_files_list):
            idx = len(tif_files_list) - 1
        if idx > 0:
            idx -= 1
            update_file_display(fig, idx, tif_files_list)
            result = process_callback(tif_files_list[idx])
            if result == 'exit':
                return idx, True
        control_state['action'] = None
    elif action == 'next':
        if idx >= len(tif_files_list):
            idx = len(tif_files_list) - 1
        elif idx < 0:
            idx = 0
        if idx < len(tif_files_list) - 1:
            idx += 1
            update_file_display(fig, idx, tif_files_list)
            result = process_callback(tif_files_list[idx])
            if result == 'exit':
                return idx, True
        control_state['action'] = None
    elif action:
        control_state['action'] = None
    return idx, False

def handle_interactive_loop(fig, idx, tif_files_list, process_callback):
    """Handle the interactive loop after initial file processing."""
    while True:
        if fig is not None and fig.canvas is not None:
            try:
                fig.canvas.flush_events()
            except:
                pass
        time.sleep(0.1)
        
        idx, should_exit = handle_clicked_file(fig, idx, tif_files_list, process_callback)
        if should_exit:
            close_figure_safely(fig)
            os._exit(0)
        if file_list_state.get('clicked_index') is not None:
            continue
        
        action = control_state.get('action')
        if action:
            idx, should_exit = handle_control_action(fig, idx, action, tif_files_list, process_callback)
            if should_exit:
                close_figure_safely(fig)
                os._exit(0)

def run_main_loop(fig, tif_files_list, process_callback):
    """Iterate through files once, then keep the window interactive."""
    idx = process_initial_files(fig, tif_files_list, process_callback)
    handle_interactive_loop(fig, idx, tif_files_list, process_callback)


