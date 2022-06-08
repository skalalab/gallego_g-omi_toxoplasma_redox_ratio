from pathlib import Path
import pandas as pd
import shutil
import tifffile
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
import numpy as np
import napari
from skimage import data

# LOAD DATASET
path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
dataset_id = '3-18-2019' #'04-19-2019' 
path_dict = path_project / fr"dictionaries\{dataset_id}-keys.csv"
df_dataset = pd.read_csv(path_dict, index_col="index")

# LOAD PATHS TO PREVIOUSLY GENERATED MASKS
# path_output = path_project / fr"{dataset_id}\040619_Katie_SPC\napari_masks_cell"
path_output = path_project / fr"{dataset_id}\03182019_Katie\napari_masks_cell"

list_str_path_generated_masks = list(map(str,list(path_output.glob("*.tiff"))))


# edit suffixes 

dict_suffixes = {
    "nadh_photons" : "_photons.tiff",
    "mask_cells" : "_mask_cells.tiff"
    }


index_start = 0 # remember lists are zero index 
index_end = 3   # upto but not including
# ITERATE THROUGH ALL THE IMAGES
for idx, row_data in list(df_dataset.iterrows())[index_start : index_end]:
    pass

    # CREATE AND MODIFY VIEWER
    viewer = napari.Viewer(show=False)
    
    # @viewer.bind_key("x", overwrite=True)
    # def exit(viewer):
    #     import napari
    #     from qtpy.QtCore import QTimer
    #     # with napari.gui_qt() as app:
    #         # viewer = napari.Viewer()
    #     time_in_msec = 1000
    #     QTimer().singleShot(time_in_msec, app.quit)
    #     # viewer.close()
    #     print("end")
            
    
    # LOAD IMAGES 
    intensity = tifffile.imread(row_data.nadh_photons) 
    
    # LOAD MASKS 
    # load generated first if available
    mask = None
    for path_mask in list_str_path_generated_masks:
        if Path(row_data.mask_cell).name in path_mask:
            mask = tifffile.imread(path_mask)
            print(f"loaded generated mask: {path_mask}")
            break
    
    if mask is None:
        print("loaded original mask")
        mask = tifffile.imread(row_data.mask_cell)
    
    # POPULATE VIEWER
    layer_intensity = viewer.add_image(intensity, name=Path(row_data.nadh_photons).name)
    layer_mask = viewer.add_labels(mask, name=Path(row_data.mask_cell).name)
    layer_mask.opacity = 0.4
    viewer.show(block=True)
    
    # FIND CELLPOSE MASK IF AVAILABLE
    # for layer in viewer.layers:
    #     if "_cp_masks_000" in layer.name:
    #         print(f"found cellpose mask: {layer.name}")
    
    # SAVE MASKS
    path_im_output = path_output / Path(row_data.mask_cell).name
    print(path_im_output)
    tifffile.imwrite(path_im_output, layer_mask.data)
    
    # @viewer.bind_key("s", overwrite=True)
    # def save_image(viewer):
    #     print("saving")
        
    # @viewer.bind_key("n", overwrite=True)
    # def next_image(viewer):
    #     print("next image")    
    
    

    
    
    
    
    