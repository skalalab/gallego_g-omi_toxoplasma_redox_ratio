from pathlib import Path

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

from cell_analysis_tools.flim import regionprops_omi
from cell_analysis_tools.io import read_asc

from helper import load_image, visualize_dictionary

path_experiment = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_dataset = path_experiment / "4-6-2019" / "040619_Katie_SPC"

path_output = path_experiment / "dictionaries"
path_excel = r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\4-6-2019\04-19-2019-keys.xlsx"

list_all_files = list(path_dataset.rglob("*"))
list_str_all_files = list(map(str,list_all_files))
df_dataset = pd.read_excel(path_excel)
print(df_dataset.head())

#%%    

suffixes = {
    # 'im_photons': '_photons .tif',
    'im_photons': '_Cycle00001_Ch2_000001.ome.tif',
    'im_toxo': '_Cycle00001_Ch1_000001.ome.tif',
    'mask_cell': '_photons _cells.tif',
    # 'mask_cytoplasm': '_photons_cyto.tiff',
    'mask_toxo': '_Cycle00001_Ch1_000001.ome_mask_toxo.tiff', 
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


dict_dataset = {}

for idx, row_data in tqdm(list(df_dataset.iterrows())[:]):
    pass
    # print(row_data)
    
    # generate image handles 
    handle_nadh = base_name + str(int(row_data.nadh)).zfill(3)
    
    # 4-6-2019
    # one or more images missing, skipping row 
    # nadh_photons : 'Cells-001_photons .tif'
    # mask_cell : 'Cells-219_photons _cells.tif'
    # mask_cell : 'Cells-240_photons _cells.tif'
    # nadh_photons : 'Cells-304_photons .tif'
    # missing fad pair for nadh 'Cells-306'
    # mask_toxo : 'Cells-308_Cycle00001_Ch1_000001.ome_toxo.tiff'
    # mask_toxo : 'Cells-311_Cycle00001_Ch1_000001.ome_toxo.tiff'
    if handle_nadh == 'Cells-219' or \
        handle_nadh == 'Cells-240' or\
        handle_nadh == 'Cells-306' or \
        handle_nadh == 'Cells-309' or \
        handle_nadh == 'Cells-312' or\
        handle_nadh == 'Cells-304' or \
        handle_nadh == 'Cells-001' \
        : 
        print(f"skipping: {handle_nadh}")
        continue
        
    # generate handle for fad
    handle_fad = base_name + str(int(row_data.fad)).zfill(3)
    bool_has_toxo = False
    if not math.isnan(row_data.toxo):
        bool_has_toxo = True
        handle_toxo = base_name + str(int(row_data.toxo)).zfill(3)
    

    ##### assemble dataset    
    month, day, year, _= Path(path_excel).stem.split("-")
    date = f"{month}_{day}_{year}"
    handle = f"{date}_idx_{idx}"
    dict_dataset[handle] = {}
    
    
    ### paths to nadh
    # paths to files
    dict_dataset[handle]["nadh_photons"] = list(filter(re.compile(handle_nadh + suffixes['im_photons']).search, list_str_all_files))[0]
    dict_dataset[handle]["nadh_a1"] = list(filter(re.compile(handle_nadh +  suffixes['a1[%]']).search, list_str_all_files))[0]
    dict_dataset[handle]["nadh_a2"] = list(filter(re.compile(handle_nadh +  suffixes['a2[%]']).search, list_str_all_files))[0]
    dict_dataset[handle]["nadh_t1"] = list(filter(re.compile(handle_nadh +  suffixes['t1']).search, list_str_all_files))[0]
    dict_dataset[handle]["nadh_t2"] = list(filter(re.compile(handle_nadh +  suffixes['t2']).search, list_str_all_files))[0]
    dict_dataset[handle]["nadh_chi"] = list(filter(re.compile(handle_nadh +  suffixes['chi']).search, list_str_all_files))[0]

    # MASKS
    path_mask_cell = list(filter(re.compile(f".*napari_masks_cell.*{handle_nadh +  suffixes['mask_cell']}").search, list_str_all_files))[0]
    # path_mask_cyto = list(filter(re.compile(handle_nadh +  suffixes['mask_cytoplasm']).search, list_str_all_files))[0]
    # path_mask_nuclei = list(filter(re.compile(handle_nadh +  suffixes['mask_nuclei']).search, list_str_all_files))[0]    
    
    dict_dataset[handle]["mask_cell"] = path_mask_cell

    # paths to fad 
    # paths to images
    dict_dataset[handle]["fad_photons"] = list(filter(re.compile(handle_fad + suffixes['im_photons'].replace("Ch2", "Ch1")).search, list_str_all_files))[0]
    dict_dataset[handle]["fad_a1"] = list(filter(re.compile(handle_fad +  suffixes['a1[%]']).search, list_str_all_files))[0]
    dict_dataset[handle]["fad_a2"] = list(filter(re.compile(handle_fad +  suffixes['a2[%]']).search, list_str_all_files))[0]
    dict_dataset[handle]["fad_t1"] = list(filter(re.compile(handle_fad +  suffixes['t1']).search, list_str_all_files))[0]
    dict_dataset[handle]["fad_t2"] = list(filter(re.compile(handle_fad +  suffixes['t2']).search, list_str_all_files))[0]
    dict_dataset[handle]["fad_chi"] = list(filter(re.compile(handle_fad +  suffixes['chi']).search, list_str_all_files))[0]


    # paths to toxo
    if bool_has_toxo:
        dict_dataset[handle]["toxo_photons"] = list(filter(re.compile(handle_toxo + suffixes['im_toxo']).search, list_str_all_files))[0]
        dict_dataset[handle]["mask_toxo"] = list(filter(re.compile(f".*generated_toxo_masks.*{handle_toxo + suffixes['mask_toxo']}").search, list_str_all_files))[0]
    
    # additional information
    dict_dataset[handle]["treatment"] = row_data.treatment
    dict_dataset[handle]["time_hours"] = row_data.Time.split(" ",1)[0]
    dict_dataset[handle]["experiment"] = str(row_data.experiment).split(" ", 1)[0]

    # add image numbers
    dict_dataset[handle]["im_num_toxo"] = row_data.toxo
    dict_dataset[handle]["im_num_nadh"] = row_data.nadh 
    dict_dataset[handle]["im_num_fad"] = row_data.fad
    
    ########### plots for visualization
    # fig, ax = plt.subplots(3,5, figsize=(10,5))
    
    # fig.suptitle(f"dataset: {handle}")
     
    # ax[0,0].imshow(load_image(dict_dataset[handle]["nadh_photons"]))
    # ax[0,0].set_axis_off()
    # ax[0,0].set_title("nadh")
    
    # ax[0,1].imshow(load_image(dict_dataset[handle]["nadh_a1"]))
    # ax[0,1].set_axis_off()
    # ax[0,1].set_title("a1")
    
    # ax[0,2].imshow(load_image(dict_dataset[handle]["nadh_t1"]))
    # ax[0,2].set_axis_off()
    # ax[0,2].set_title("t1")
    
    # ax[0,3].imshow(load_image(dict_dataset[handle]["nadh_a2"]))
    # ax[0,3].set_axis_off()
    # ax[0,3].set_title("a2")
    
    # ax[0,4].imshow(load_image(dict_dataset[handle]["nadh_t2"]))
    # ax[0,4].set_axis_off()
    # ax[0,4].set_title("t2")
    
    # # fad 
    # ax[1,0].imshow(load_image(dict_dataset[handle]["fad_photons"]))
    # ax[1,0].set_axis_off()
    # ax[1,0].set_title("fad intensity")
    
    # ax[1,1].imshow(load_image(dict_dataset[handle]["fad_a1"]))
    # ax[1,1].set_axis_off()
    # ax[1,1].set_title("fad a1")
    
    # ax[1,2].imshow(load_image(dict_dataset[handle]["fad_t1"]))
    # ax[1,2].set_axis_off()
    # ax[1,2].set_title("fad t1")
    
    # ax[1,3].imshow(load_image(dict_dataset[handle]["fad_a2"]))
    # ax[1,3].set_axis_off()
    # ax[1,3].set_title("fad a2")
    
    # ax[1,4].imshow(load_image(dict_dataset[handle]["fad_t2"]))
    # ax[1,4].set_axis_off()
    # ax[1,4].set_title("fad t2")
    
    # #  MASKS AND TOXO
    
    # ax[2,0].imshow(load_image(path_mask_cell))
    # ax[2,0].set_axis_off()
    # ax[2,0].set_title("mask_cell")
    
    
    # if bool_has_toxo: 
    #     ax[2,1].set_title("mCherry / toxo")
    #     ax[2,1].imshow(load_image(dict_dataset[handle]["toxo_photons"]))
        
    #     ax[2,2].imshow(load_image(dict_dataset[handle]["mask_toxo"]))
    #     ax[2,2].set_title("mask toxo")

    # ax[2,1].set_axis_off()
    # ax[2,2].set_axis_off()
    # ax[2,3].set_axis_off()
    # ax[2,4].set_axis_off()

    # plt.show()
    
#%%
df_output_dataset = pd.DataFrame(dict_dataset).transpose()
df_output_dataset.index.name = "index"
df_output_dataset.to_csv(path_output / f"{Path(path_excel).stem}.csv")

#%%
for row_data in tqdm(list(dict_dataset)[:2]):
    pass
    visualize_dictionary(row_data,dict_dataset[row_data])
    
    
    # im = tifffile.imread((dict_dataset[row_data]['toxo_photons']))
    # plt.imshow(np.clip(im,0, np.percentile(im, 99)))
        
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