from pathlib import Path
from flim_tools.flim import regionprops_omi
from flim_tools.io import read_asc
import pandas as pd
from tqdm import tqdm
import re
import tifffile
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
import pathlib
import numpy as np
import math

path_experiment = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_dataset = path_experiment / "3-18-2019" / "03182019_Katie"


path_excel = r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\3-18-2019\3-18-2019_sets.xlsx"

list_all_files = list(path_dataset.rglob("*"))
list_str_all_files = list(map(str,list_all_files))
df_dataset = pd.read_excel(path_excel)
print(df_dataset.head())

#%%
def load_image(path):
    # detects extension and loads image accordingly
    # tif/tiff vs asc
    if not isinstance(path, pathlib.PurePath):
        path = Path(path)
    pass
    if path.suffix == ".asc":
        return read_asc(path)
    if path.suffix in [".tiff", ".tif"]:
        return tifffile.imread(path)

#%%    

suffixes = {
    'im_photons': '_photons .tif',
    'im_toxo': '_Cycle00001_Ch1_000001.ome.tif',
    'mask_cell': '_photons _cells.tiff',
    # 'mask_cytoplasm': '_photons_cyto.tiff',
    'mask_toxo': '_Cycle00001_Ch1_000001.ome_toxo.tiff', 
    # 'mask_nuclei': '_photons_nuclei.tiff',
    'a1[%]': '_a1\[%\].asc',
    'a2[%]': '_a2\[%\].asc',
    't1': '_t1.asc',
    't2': '_t2.asc',
    'chi': '_chi.asc',
    'sdt': '.sdt',
 }
#%% load dataset 
base_name = "Cells-"

"".zfill(3)

for idx, row_data in tqdm(list(df_dataset.iterrows())[:]):
    print(row_data)
    
    # generate image handles 
    handle_nadh = base_name + str(int(row_data.nadh)).zfill(3)
    handle_fad = base_name + str(int(row_data.fad)).zfill(3)
    handle_toxo = None
    if not math.isnan(row_data.toxo):
        handle_toxo = base_name + str(int(row_data.toxo)).zfill(3)
    
    
    # skip 
    if  handle_nadh == 'Cells-007' or \
        handle_nadh == 'Cells-030' or \
        handle_nadh == 'Cells-070' or \
        handle_nadh == 'Cells-340' or \
        handle_nadh == 'Cells-001' or \
        handle_toxo == 'Cells-105' or\
        handle_nadh == 'Cells-133'  \
            : 
        continue
    
    # find images 
    path_nadh = list(filter(re.compile(handle_nadh + suffixes['im_photons']).search, list_str_all_files))[0]
    path_fad = list(filter(re.compile(handle_fad + suffixes['im_photons']).search, list_str_all_files))[0]
    
    if handle_toxo is not None:
        path_toxo = list(filter(re.compile(handle_toxo + suffixes['im_toxo']).search, list_str_all_files))[0]

    # find masks 
    path_mask_cell = list(filter(re.compile(handle_nadh + suffixes['mask_cell']).search, list_str_all_files))[0]
    if handle_toxo is not None:
        path_mask_toxo = list(filter(re.compile(handle_toxo + suffixes['mask_toxo']).search, list_str_all_files))[0]

    # find nadh a1
    path_nadh_a1 = list(filter(re.compile(handle_nadh + suffixes['a1[%]']).search, list_str_all_files))[0]
    # mask nuclei on nadh a1 > 79
    
    fig, ax = plt.subplots(2,3, figsize=(10,5))
    
    ax[0,0].imshow(load_image(path_nadh))
    ax[0,0].set_axis_off()
    ax[0,0].set_title("nadh")
    
    ax[0,1].imshow(load_image(path_fad))
    ax[0,1].set_axis_off()
    ax[0,1].set_title("fad")
    
    if handle_toxo is not None: 
        ax[0,2].imshow(load_image(path_toxo))
    ax[0,2].set_axis_off()
    ax[0,2].set_title("mChery / toxo")

    ax[1,0].imshow(load_image(path_mask_cell))
    ax[1,0].set_axis_off()
    ax[1,0].set_title("mask_cell")
    
    if handle_toxo is not None:
        ax[1,2].imshow(load_image(path_mask_toxo))
    ax[1,2].set_axis_off()
    ax[1,2].set_title("mask toxo")
    
    plt.show()
    
    
    
    
    # 
    # #paths to toxo masks
    # path_masks_toxo = path_dataset / "masks_toxo" / "TIFF"
    # list_path_masks_toxo = list(path_masks_toxo.glob("*.tiff"))
    
    # # paths to nadh masks
    # path_masks_nadh_toxo = path_dataset / "masks_nadh_toxo"
    # list_path_masks_nadh = list(path_masks_nadh_toxo.glob("*_photons _cells.tiff"))
    
    # ## nadh base_names 
    # base_names = list_path_masks_nadh[0].stem
    
    # #paths to all images 
    # path_images = path_dataset / "nadh_fad_no_toxo"
    # list_paths_all_images = list(path_images.glob("*"))    


#%%