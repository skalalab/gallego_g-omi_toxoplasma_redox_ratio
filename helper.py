from flim_tools.io import read_asc
from pathlib import Path
import pathlib
import tifffile
import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
import math


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
    
    
def visualize_dictionary(index_name : str, dict_set : dict)-> None:
    ########## plots for visualization
    
    bool_has_toxo = False
    if 'mask_toxo' in dict_set and \
        isinstance(dict_set['mask_toxo'], str) and \
            Path(dict_set['mask_toxo']).exists():
        bool_has_toxo = True


    fig, ax = plt.subplots(3,5, figsize=(10,5))
    
    fig.suptitle(f"dataset: {index_name}")
     
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

    plt.show()