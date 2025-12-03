#!/usr/bin/env python
"""
pcc_tifFolder.py

Calculates cross-correlation (CC) for all TIF files in a folder.
Generates _pcc.log files for each TIF file.
"""

import sys,time,os
import argparse
import numpy as np
from scipy import signal
import DOM_lib as lib
import multiprocessing
import parmap
from numba import njit
import DOM_constants

def parse_args():
    parser = argparse.ArgumentParser(description='PCC calculation for tif folder')
    parser.add_argument('data_folder', help='Folder containing tif data files')
    parser.add_argument('--map', default=DOM_constants.DEFAULT_MAP, help=f'Reference map (tif). Default: {DOM_constants.DEFAULT_MAP}')
    parser.add_argument('--bpp', type=int, default=DOM_constants.DEFAULT_BPP,
                       help=f'Base pairs per pixel (default: {DOM_constants.DEFAULT_BPP})')
    parser.add_argument('--start-bpp', type=int, default=DOM_constants.DEFAULT_START_BPP,
                       help=f'Start BPP for CC calculation range (default: {DOM_constants.DEFAULT_START_BPP})')
    parser.add_argument('--end-bpp', type=int, default=DOM_constants.DEFAULT_END_BPP,
                       help=f'End BPP for CC calculation range (default: {DOM_constants.DEFAULT_END_BPP})')
    return parser.parse_args()

def find_reference_map(data_folder):
    """Find reference map in common locations."""
    search_paths = [
        os.path.join(DOM_constants.DEFAULT_MAP),
        os.path.join(data_folder, "..", DOM_constants.DEFAULT_MAP),
        os.path.join(data_folder, "..", "..", DOM_constants.DEFAULT_MAP),
    ]
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    return None

@njit
def pcc(xx_r,xx_g,y_r,y_g):
    L=len(xx_r)
    n=len(y_r)
    cc2 = []
    for i in range(L-n):
        x_r=xx_r[i:n+i]
        x_g=xx_g[i:n+i]
        cc2.append(lib.corrcoef(x_r, y_r)*(lib.dcc(x_g, y_g))**2)
        cc2.append(lib.corrcoef(x_r, y_r[::-1])*(lib.dcc(x_g, y_g[::-1]))**2)

    shift=cc2.index(max(cc2))
    if (shift%2==0):
        direction=1
    else:
        direction=-1
    shift //= 2
    max_r=lib.corrcoef(xx_r[shift:n+shift],y_r[::direction])
    max_g=lib.dcc(xx_g[shift:n+shift],y_g[::direction])
    max_cc = max_r * max_g
    return shift, direction,max(cc2), max_cc, max_r, max_g

def scan_pcc(xx_r,xx_g,y_r,y_g,log, start_bpp, end_bpp, bpp):
    n=len(y_r)
    cc2_max=-np.inf
    for bp_per_pixel in np.arange(start_bpp, end_bpp, 1):
        s=bp_per_pixel/bpp
        yy_r=signal.resample(y_r,int(s*n))     
        yy_g=signal.resample(y_g,int(s*n))
        shift2,direction2,cc2,cc,cc_r,cc_g=pcc(xx_r,xx_g,yy_r,yy_g)
        print ("%.3f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f"% (s,bp_per_pixel,shift2,direction2,cc2,cc,cc_r,cc_g), file=log)
        if (cc2>cc2_max):
            cc2_max=cc2
            cc_max=cc
            cc_r_max=cc_r
            cc_g_max=cc_g
            shift=shift2
            direction=direction2
            scale=s
    log.close()
    return shift,direction,scale,cc2_max,cc_max,cc_r_max,cc_g_max

def cross_correlation(RGmap, data_file):
    data_log=data_file.replace(".tif","_pcc.log")
    log=open(data_log,'w')
    print ("scale,bpp,shift,direction,score,cc_rg,cc_r,cc_g",file=log)
    map_r,map_g,_=lib.tif2RGprofile(RGmap)
    data_r,data_g,_=lib.tif2RGprofile(data_file)

    clip1r=lib.normalize(map_r)
    clip1g=lib.normalize(map_g)
    clip2r=lib.normalize(data_r)
    clip2g=lib.normalize(data_g)
    w2=len(clip2r)

    # the circularity of the genome
    clip1r=np.concatenate((clip1r,clip1r[0:w2]), axis=0)
    clip1g=np.concatenate((clip1g,clip1g[0:w2]), axis=0)
    pcc_shift,pcc_direction,scale,cc2_max,cc_max,cc_r_max,cc_g_max=scan_pcc(clip1r,clip1g, clip2r,clip2g, log, start_bpp, end_bpp, bpp)
    matching_result=[data_file,pcc_shift,pcc_direction,scale,cc2_max,cc_max,cc_r_max,cc_g_max]
    
    return matching_result

if __name__ == '__main__':
    args = parse_args()

    tif_data_folder = args.data_folder

    bpp = args.bpp
    start_bpp = args.start_bpp
    end_bpp = args.end_bpp

    t=time.time()
    RGmap = lib.validate_tif_path(args.map)

    if os.path.isdir(tif_data_folder):
        tif_files = lib.Read_TIFF_folder(tif_data_folder)
    elif os.path.isfile(tif_data_folder) and tif_data_folder.lower().endswith(".tif"):
        tif_files = [tif_data_folder]
    else:
        raise ValueError("The provided data-folder is neither a directory nor a tif file.")

    if tif_data_folder[-1]!='/':
        tif_data_folder +='/'
    pcc_file=tif_data_folder+"pcc_data.csv"
    f=open(pcc_file,'w')

    tif_files_result = [(RGmap, df) for df in tif_files]

    num_cores = multiprocessing.cpu_count() if len(tif_files) > 1 else 1
    result = parmap.starmap(cross_correlation, tif_files_result, pm_processes=num_cores, pm_pbar=True)

    for line in result:    
        print (str(line)[1:-1], file=f)
    f.close()
    print ("time: %dhour %dmin %dsec" % ((time.time()-t)//3600, (time.time()-t)//60, round((time.time()-t)%60)))