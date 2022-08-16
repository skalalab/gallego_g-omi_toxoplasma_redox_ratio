from cell_analysis_tools.io import read_asc
from pathlib import Path
import pathlib
import tifffile
import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
import math

from skimage.morphology import (disk, 
                                binary_closing, 
                                binary_opening, 
                                remove_small_objects,
                                remove_small_holes,
                                label)

from cell_analysis_tools.visualization import compare_images
from cell_analysis_tools.image_processing import four_color_theorem, four_color_to_unique

def load_image(path)-> np.ndarray:
    """
    Detects the extension and loads image accordingly
    if its a tif/tiff or an asc

    Parameters
    ----------
    path : pathlib path or str
        path to the image.

    Returns
    -------
    np.ndarray
        array containig image.

    """
    if not isinstance(path, pathlib.PurePath):
        path = Path(path)
    pass
    if path.suffix == ".asc":
        return read_asc(path)
    if path.suffix in [".tiff", ".tif"]:
        return tifffile.imread(path)
    
    
def visualize_dictionary(index_name : str, dict_set : dict, save_path : Path = None)-> None:
    ########## plots for visualization
    
    bool_has_toxo = False
    if 'mask_toxo' in dict_set and \
        isinstance(dict_set['mask_toxo'], str) and \
            Path(dict_set['mask_toxo']).exists():
        bool_has_toxo = True


    fig, ax = plt.subplots(3,5, figsize=(10,5))
    
    fig.suptitle(f"dataset: {index_name}  |  timepoint: {dict_set['time_hours']}")
     
    ax[0,0].imshow(load_image(dict_set["nadh_photons"]))
    ax[0,0].set_axis_off()
    ax[0,0].set_title("nadh")
    
    ax[0,1].imshow(load_image(dict_set["nadh_a1"]))
    ax[0,1].set_axis_off()
    ax[0,1].set_title("a1")
    
    ax[0,2].imshow(load_image(dict_set["nadh_t1"]))
    ax[0,2].set_axis_off()
    ax[0,2].set_title("t1")
    
    ax[0,3].imshow(load_image(dict_set["nadh_a2"]))
    ax[0,3].set_axis_off()
    ax[0,3].set_title("a2")
    
    ax[0,4].imshow(load_image(dict_set["nadh_t2"]))
    ax[0,4].set_axis_off()
    ax[0,4].set_title("t2")
    
    # fad 
    ax[1,0].imshow(load_image(dict_set["fad_photons"]))
    ax[1,0].set_axis_off()
    ax[1,0].set_title("fad intensity")
    
    ax[1,1].imshow(load_image(dict_set["fad_a1"]))
    ax[1,1].set_axis_off()
    ax[1,1].set_title("fad a1")
    
    ax[1,2].imshow(load_image(dict_set["fad_t1"]))
    ax[1,2].set_axis_off()
    ax[1,2].set_title("fad t1")
    
    ax[1,3].imshow(load_image(dict_set["fad_a2"]))
    ax[1,3].set_axis_off()
    ax[1,3].set_title("fad a2")
    
    ax[1,4].imshow(load_image(dict_set["fad_t2"]))
    ax[1,4].set_axis_off()
    ax[1,4].set_title("fad t2")
    
    #  MASKS AND TOXO
    
    ax[2,0].imshow(load_image(dict_set["mask_cell"]))
    ax[2,0].set_axis_off()
    ax[2,0].set_title("mask_cell")
    
    
    if bool_has_toxo: 
        if "toxo_photons" in dict_set:
            ax[2,1].set_title("mCherry / toxo")
            ax[2,1].imshow(load_image(dict_set["toxo_photons"]), vmax=30)
        
        ax[2,2].imshow(load_image(dict_set["mask_toxo"]))
        ax[2,2].set_title("mask toxo")

    ax[2,1].set_axis_off()
    ax[2,2].set_axis_off()
    ax[2,3].set_axis_off()
    ax[2,4].set_axis_off()


    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    
def preprocess_mask(mask, row_data, pixels = 100, debug=False):
    
    # PROCESS MASK
    # Gina's whole cell masks sometimes contain stray pixels 
    # and sometimes missing pixels in mask
    # this tries to remove those pixels and fill in the rest
    
    # this guarantees that rois have a unique value
    # original masks have some rois with the same value currently
    mask_original = mask.copy()
    mask_unique, solutions = four_color_theorem(mask)
    mask = four_color_to_unique(mask_unique)
    
    list_roi_values = list(np.unique(mask))
    list_roi_values.remove(0)
    
    mask_edited = np.zeros_like(mask)
    
    # fill small holes in masks
    for roi_value in list_roi_values:
        pass
        temp_mask = mask == roi_value
               
        # remove gaps in masks
        temp_mask = binary_closing(temp_mask)
        
        # remove small objects
        temp_mask = binary_opening(temp_mask)
        
        # if roi_value == 11:
        #     print('stop')
        # remove small objects
        pixels = 100
        temp_mask = remove_small_holes(temp_mask, area_threshold=pixels)
        temp_mask = remove_small_objects(temp_mask, min_size=pixels)
        mask_edited[temp_mask] = roi_value
        

    mask = mask_edited

    if debug:
        filename = Path(rf"{row_data.mask_cell}")
        compare_images(mask_original, f"original mask \n{filename.stem}", mask_edited, "processed mask")
    
    return mask
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    