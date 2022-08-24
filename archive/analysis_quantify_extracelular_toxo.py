from pathlib import Path
from cell_analysis_tools.flim import regionprops_omi
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
from tqdm import tqdm
import pandas as pd
from helper import load_image, visualize_dictionary, preprocess_mask
from datetime import date
import math
import numpy as np
from cell_analysis_tools.image_processing import kmeans_threshold, four_color_theorem, four_color_to_unique
from cell_analysis_tools.metrics import percent_content_captured
from cell_analysis_tools.visualization import compare_images
from natsort import natsorted
import tifffile
from skimage.morphology import binary_closing, remove_small_holes, binary_opening

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_datasets = path_project / "dictionaries"
path_output_features = path_project / "features"

list_dataset_dicts = list(path_datasets.glob("*.csv"))

#%%


debug = False
skipped_images = []

######## 
pixels_nadh_photons = np.zeros(1,)
pixels_nadh_a1 = np.zeros(1,)
pixels_nadh_a2 = np.zeros(1,)
pixels_nadh_t1 = np.zeros(1,)
pixels_nadh_t2 = np.zeros(1,)
pixels_nadh_chi = np.zeros(1,)

pixels_fad_photons = np.zeros(1,)
pixels_fad_a1 = np.zeros(1,)
pixels_fad_a2 = np.zeros(1,)
pixels_fad_t1 = np.zeros(1,)
pixels_fad_t2 = np.zeros(1,)
pixels_fad_chi = np.zeros(1,)
pixels_rr_normalized = np.zeros(1,)


# iterate through dictionaries
for dict_dataset in tqdm(list_dataset_dicts[1:]):
    pass
    df_data = pd.read_csv(dict_dataset, index_col=("index"))
    
    print(dict_dataset)
    # itereate through rows of dict
    for idx, row_data in tqdm(list(df_data.iterrows())[:]): # iterate through sets
        pass
    
        # only grab data for media+toxo condition
        if row_data.treatment != 'media+toxo':
            continue
        
        # Relabel whole cell mask
        mask_cell = load_image(row_data.mask_cell)
        mask_cell = preprocess_mask(mask_cell, row_data=row_data, debug=False) # see docstring for preprocessing steps
        # compare_images(nadh_photons, f"original image \n{filename.stem}", mask, "edited mask")
        
        # load toxo mask
        mask_toxo = load_image(row_data.mask_toxo)
        im_toxo = load_image(row_data.toxo_photons)

        # compare_images(mask_cell, "cells", mask_toxo, "toxo")
        
        # LOAD IMAGES 
        nadh_photons = load_image(row_data.nadh_photons)
        nadh_a1 = load_image(row_data.nadh_a1)
        nadh_a2 = load_image(row_data.nadh_a2)
        nadh_t1 = load_image(row_data.nadh_t1)
        nadh_t2 = load_image(row_data.nadh_t2)
        nadh_chi = load_image(row_data.nadh_chi)
        
        fad_photons = load_image(row_data.fad_photons)
        fad_a1 = load_image(row_data.fad_a1)
        fad_a2 = load_image(row_data.fad_a2)
        fad_t1 = load_image(row_data.fad_t1)
        fad_t2 = load_image(row_data.fad_t2)
        fad_chi = load_image(row_data.fad_chi)
        
        #NOTE: this will produce nan's from division by zero
        rr_normalized = nadh_photons / (nadh_photons + fad_photons) 
        
        # COMPUTE EXTRACELULAR PIXELS
        mask_extracelular = np.invert(mask_cell > 0)
        mask_toxo_extra = mask_toxo * mask_extracelular
        # plt.imshow(mask_toxo_extra)
        # plt.show()
                
        # visualize images and masks
        fig, ax = plt.subplots(1,5, figsize=(10,3))
        fig.suptitle(f"{idx}")
        ax[0].imshow(nadh_photons)
        ax[0].set_axis_off()
        ax[0].set_title("nadh")
        ax[1].imshow(mask_cell)
        ax[1].set_axis_off()
        ax[1].set_title("mask")
        ax[2].imshow(np.clip(im_toxo,0, np.percentile(im_toxo,99)))
        ax[2].set_axis_off()
        ax[2].set_title("toxo")
        ax[3].imshow(mask_toxo)
        ax[3].set_axis_off()
        ax[3].set_title("mask")
        ax[4].imshow(mask_toxo_extra)
        ax[4].set_axis_off()
        ax[4].set_title("extracelular toxo")
        plt.show()
        
        ########## 
        pixels_nadh_photons = np.concatenate((pixels_nadh_photons, nadh_photons[mask_toxo_extra])) # 
        pixels_nadh_a1 = np.concatenate((nadh_a1[mask_toxo_extra], pixels_nadh_a1))
        pixels_nadh_a2 = np.concatenate((nadh_a2[mask_toxo_extra], pixels_nadh_a2))
        pixels_nadh_t1 = np.concatenate((nadh_t1[mask_toxo_extra], pixels_nadh_t1))
        pixels_nadh_t2 = np.concatenate((nadh_t2[mask_toxo_extra], pixels_nadh_t2))
        pixels_nadh_chi = np.concatenate((nadh_chi[mask_toxo_extra], pixels_nadh_chi))
        
        pixels_fad_photons = np.concatenate((fad_photons[mask_toxo_extra], pixels_fad_photons))
        pixels_fad_a1 = np.concatenate((fad_a1[mask_toxo_extra], pixels_fad_a1))
        pixels_fad_a2 = np.concatenate((fad_a2[mask_toxo_extra], pixels_fad_a2))
        pixels_fad_t1 = np.concatenate((fad_t1[mask_toxo_extra], pixels_fad_t1))
        pixels_fad_t2 = np.concatenate((fad_t2[mask_toxo_extra], pixels_fad_t2 ))
        pixels_fad_chi = np.concatenate((fad_chi[mask_toxo_extra], pixels_fad_chi))
        pixels_rr_normalized = np.concatenate((rr_normalized[mask_toxo_extra], pixels_rr_normalized))
        
        
        # would compute regionprops here
        # pull put pixels for each of these images and aggregate into an array
        
        

#%% 


DICT_FEATURE_PIXELS ={
    'toxo_nadh_photons': pixels_nadh_photons,  
    'toxo_nadh_a1' : pixels_nadh_a1,
    'toxo_nadh_a2'  : pixels_nadh_a2,
    'toxo_nadh_t1'  : pixels_nadh_t1 ,
    'toxo_nadh_t2'  : pixels_nadh_t2,
    'toxo_nadh_chi'  : pixels_nadh_chi,
    # fad
    'toxo_fad_photons'  : pixels_fad_photons,
    'toxo_fad_a1'  : pixels_fad_a1,
    'toxo_fad_a2'  : pixels_fad_a2,
    'toxo_fad_t1'  : pixels_fad_t1,
    'toxo_fad_t2'  : pixels_fad_t2,
    'toxo_fad_chi' : pixels_fad_chi,
    'toxo_rr_normalized' : pixels_rr_normalized[~np.isnan(pixels_rr_normalized)] # remove nan's from zero division
    }

dict_extracellular_toxo = {}
for feature_key, feature_values in DICT_FEATURE_PIXELS.items():
    pass
    dict_extracellular_toxo[feature_key] = {} # skip first item used to initialize array, remove
    dict_extracellular_toxo[feature_key]['mean'] = np.mean(feature_values[1:])
    dict_extracellular_toxo[feature_key]['stdev'] = np.std(feature_values[1:])

    plt.hist(feature_values[1:], histtype='step', bins=100)
    plt.title(f"{feature_key} histogram")
    plt.xlabel(f"{feature_key}")
    plt.ylabel("count")
    plt.show()
                