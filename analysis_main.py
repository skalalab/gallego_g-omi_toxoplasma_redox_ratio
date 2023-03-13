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

# DEFINE ANALYSIS TYPE* SEE NOTE BELOW
# analysis_type = "toxo_inside_cells_high_vs_low"
# analysis_type = "whole_cell_vs_high_toxo"
# analysis_type = "high_toxo_cells_vs_outside_toxo" # outside toxo may require it's own script

# NOTE: offloading thresholding to script that creates figures
# just quantifying here

# path_output_features = path_output_features
#%% COMPUTE OMI PARAMETERS

debug = False
skipped_images = []

# iterate through dictionaries
for dict_dataset in tqdm(list_dataset_dicts[:]):
    pass
    df_data = pd.read_csv(dict_dataset, index_col=("index"))
    
    print(dict_dataset)
    # itereate through rows of dict
    for idx, row_data in tqdm(list(df_data.iterrows())[:]): # 23:24 iterate through sets
        pass
    
        # only look at media nad media+toxo experiments
        # if row_data.treatment not in ["media", "media+toxo"]:
        #     print(f"skipped: {row_data.treatment}")
        #     continue
        
        # Relabel whole cell mask
        mask_cell = load_image(row_data.mask_cell)
        mask_cell = preprocess_mask(mask_cell, row_data=row_data, debug=False) # see docstring for preprocessing steps
        
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
        
        # VISUALIZE INTENSITY AND MASK
        if debug:
            suptitle = f"{idx} \n{Path(row_data.nadh_photons).name}  |  {row_data.treatment}  |  {row_data.time_hours} hrs"
            compare_images(f"original image", nadh_photons,
                           " mask", mask_cell,
                           suptitle=suptitle)

        
        # COMPUTE REGIONPROPS
        omi_props = regionprops_omi(idx, 
                                     label_image = mask_cell, 
                                     im_nadh_intensity = nadh_photons, 
                                     im_nadh_a1 = nadh_a1, 
                                     im_nadh_a2 = nadh_a2, 
                                     im_nadh_t1 = nadh_t1, 
                                     im_nadh_t2 = nadh_t2,
                                     
                                     im_fad_intensity = fad_photons, 
                                     im_fad_a1 = fad_a1,
                                     im_fad_a2 = fad_a2,
                                     im_fad_t1 = fad_t1, 
                                     im_fad_t2 = fad_t2,
                                     # optional chi paramters
                                     im_nadh_chi=nadh_chi,
                                     im_fad_chi=fad_chi,
                                     
                                     other_props = ["area",
                                                    "perimeter",
                                                    "eccentricity",
                                                    "centroid",
                                                    "perimeter"
                                                    ]) 
        
        ################################# QUALITY CONTROL 
        # look for bad fits based on chi squared
        # plot and then skip them
        outliers_found = False
        for r in omi_props:
            pass
            if omi_props[r]["nadh_chi_median"] > 1.5: # omi_props[r]["nadh_t2_mean"] > 3500 or \
                  outliers_found = True
                
        if outliers_found:
            for param in [
                    # "redox_ratio_mean", \
                    # "nadh_t2_mean"
                    "nadh_chi_median"
                    ]:
                # param = "redox_ratio_mean"
                plt.title(f"{idx} \n{param}")
                plt.imshow(mask_cell)
                for key in omi_props:
                    text = f"{omi_props[key][param]:.3f}"
                    plt.text(omi_props[key]["centroid"][1],
                              omi_props[key]["centroid"][0],
                              text,
                              fontsize=7,
                              c="w")
                    plt.plot(omi_props[key]["centroid"][1],
                              omi_props[key]["centroid"][0],
                              marker='o',
                              markersize=3,
                              c="w")
                plt.show()
                print(f"skipping exporting outliers: {idx}")
                skipped_images.append(idx)
            continue
                

        ################################# END QUALITY CONTROL
        
        # if toxo mask, determine high_toxo cells
        if 'toxo' in row_data.treatment:
           
            # label toxo according to cell roi value
            mask_toxo = load_image(row_data.mask_toxo)
            mask_toxo = mask_cell * mask_toxo
             
            if debug:
                    
                compare_images(f"original image  ", nadh_photons,
                               "mask toxo", mask_toxo, 
                               suptitle=suptitle)

            # quantify percent toxo in cell
            for key_roi in omi_props: # iterate through toxo masks 
                roi_toxo = mask_toxo == omi_props[key_roi]['mask_label']
                roi_cell = mask_cell == omi_props[key_roi]['mask_label']

                if roi_toxo.sum() == 0: 
                    omi_props[key_roi]["percent_toxo"] = 0
                else:
                    omi_props[key_roi]["percent_toxo"] = percent_content_captured(roi_cell, roi_toxo)
                    
                # VISUALIZE TOXO PER ROI -- use for debugging
                compare_images("roi_cell", roi_cell, "roi_toxo", roi_toxo, 
                          suptitle=f"{key_roi} \npercent toxo={omi_props[key_roi]['percent_toxo']:.3f}" )
        
        # if this image doesn't contain toxo then fill values with 0
        else:
            for key_roi in omi_props:
                pass
                omi_props[key_roi]["percent_toxo"] = 0
    
        ## CREATE DATAFRAME
        df_props = pd.DataFrame(omi_props).transpose()
        df_props.index.name = "base_name_with_roi"
        
        # save image id
        df_props["image_set"] = idx # * len(df_props)

        ## add other dictionary data to df
        for item_key in row_data.keys():
            pass
            df_props[item_key] = row_data[item_key]

        # df_props.to_csv(path_output_features / f"{idx}.csv")  

#%% CREATE ONE CSV 
 
    # path_output_props = Path(r"Z:\0-Projects and Experiments\RD - redox_ratio_development\Data Combined + QC Complete\0-features_csv")
    # path_output_proprs_type = path_output_props / analysis_type    
    list_path_features_csv = list(path_output_features.glob("*.csv"))
    
    df_all_props = None
    for path_feat_csv in tqdm(list_path_features_csv):
        pass
        # initialize first df entry else append to running df
        df_omi = pd.read_csv(path_feat_csv)
        df_all_props = df_omi if df_all_props is None else \
            pd.concat([df_all_props, df_omi],ignore_index=True)
            
    # # label index of complete dictionary 
    # df_all_props = df_all_props.set_index("base_name", drop=True)
    
    # treatment has pH, make those a string or pandas will be angry
    # df_all_props["treatment"] = df_all_props["treatment"].astype(str)

    # rename control to position it first
    # df_all_props["treatment"] = df_all_props["treatment"].replace({"control":"0-control"})

    ### Save df for analysis 
    path_feature_summaries = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
    d = date.today().strftime("%Y_%m_%d")
    df_all_props = df_all_props.set_index('base_name_with_roi', drop=True)
    # df_all_props.to_csv(path_feature_summaries / f"{d}_all_props_cells.csv")
    
#%% PLOT VALUES WITH LARGE MEDIAN CHI SQUARED

# # plot all nadh chi

# df_all_props = pd.read_csv(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\2022_05_26_all_props.csv")

# key = 'nadh_chi_median'
# plt.hist(df_all_props[key], bins=100)
# plt.title(key)
# plt.show()

# df_outliers = df_all_props[df_all_props[key] > 1.5]
# set_paths_nadh_chi_files = natsorted(set(df_outliers["nadh_chi"]))

# # #%%
# from ast import literal_eval
# for path_im_chi_outlier in set_paths_nadh_chi_files[:]:
#     pass
    
#     df_subset = df_outliers[df_outliers["nadh_chi"] == path_im_chi_outlier]
#     df_subset = df_subset.reset_index(drop=True)
#     fig, ax = plt.subplots(1,4, figsize=(10,3))
    
#     fig.suptitle(f"{df_subset.base_name_with_roi[0].rsplit('_',1)[0]}")
#     ax[0].imshow(load_image(df_subset.nadh_photons[0]))
#     ax[0].set_title('nadh photons')
#     ax[0].set_axis_off()
    
#     im_chi = load_image(df_subset.nadh_chi[0])
#     ax[1].imshow(im_chi, vmax=1.5)
#     ax[1].set_title('nadh chi')
#     ax[1].set_axis_off()
    
#     mask = load_image(df_subset.mask_cell[0])
#     ax[2].imshow( (mask > 0) * im_chi, )#vmax=1.5
#     ax[2].set_title('nadh chi masked')
#     ax[2].set_axis_off()
    
#     ax[3].imshow(mask)
#     ax[3].set_title('mask')
#     ax[3].set_axis_off()
    
#     for idx, outlier_row in df_subset.iterrows():
#         pass
#         text = f"{outlier_row.nadh_chi_median:.2f}"
#         # text = f"{outlier_row.mask_label}"
#         if outlier_row.mask_label == 71:
#             print("roi 71")
#         centroid = literal_eval(outlier_row["centroid"])
#         ax[2].text(centroid[1],
#                  centroid[0],
#                  text,
#                  c="w",
#                  fontsize=6)
#         ax[2].plot(centroid[1],
#                      centroid[0],
#                      marker='o',
#                      markersize=1,
#                      c="w")
#     plt.show()

#%%