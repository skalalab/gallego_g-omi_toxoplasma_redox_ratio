from pathlib import Path
from flim_tools.flim import regionprops_omi
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
from tqdm import tqdm
import pandas as pd
from helper import load_image, visualize_dictionary
from datetime import date
import math
import numpy as np
from flim_tools.image_processing import kmeans_threshold
from natsort import natsorted

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_datasets = path_project / "dictionaries"
path_output_features = path_project / "features"

list_dataset_dicts = list(path_datasets.glob("*.csv"))

#%% COMPUTE OMI PARAMETERS

# iterate through dictionaries
for dict_dataset in tqdm(list_dataset_dicts[:]):
    pass
    df_data = pd.read_csv(dict_dataset, index_col=("index"))
    
    print(dict_dataset)
    # itereate through rows of dict
    for idx, row_data in tqdm(list(df_data.iterrows())[:]):#iterate through sets
        pass
    
        # if idx != '3_18_2019_idx_118' :# "3_18_2019_idx_111" or :
        #     continue
        # else:
        #     print("check set")
        
        # visualize_dictionary(idx, row_data)
        
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
        
        mask_cell = load_image(row_data.mask_cell)
        
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
        
        
        
        # look for extreme outliers, show image with outliers 
        outliers_found = False
        for r in omi_props:
            pass
            if omi_props[r]["nadh_t2_mean"] > 3500:
                outliers_found = True
                
        if outliers_found:
            for param in [
                    "redox_ratio_mean", \
                    "nadh_t2_mean"
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
        
        
        # quality control for rois  
        # chi square mean above 1.5 should be looked at
        
        
        
        # look for ROIs with 
        # import matplotlib.patches as mpatches
        # label_image = mask_cell
        # param = "label"
        # fig, ax = plt.subplots()
        # ax.set_title(param)
        # ax.imshow(label_image)
        # for region in omi_props:
        #     pass
        #     text = f"{region[param]}"
        #     ax.text(region["centroid"][1],
        #              region["centroid"][0],
        #              text,
        #              c="w")
        #     ax.plot(region["centroid"][1],
        #              region["centroid"][0],
        #              marker='o',
        #              markersize=5,
        #              c="w")
            
        #     minr, minc, maxr, maxc = region.bbox
        #     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                       fill=False, edgecolor='red', linewidth=2)
        #     ax.add_patch(rect)
        # plt.show()
        
        
        ## initialize column of toxo pixels
        for key_roi in omi_props:
            pass
            omi_props[key_roi]["pixels_toxo"] = 0
        
        ### quantify toxo in cells
        bool_has_toxo = False
        if isinstance(row_data.mask_toxo, str) and Path(row_data.mask_toxo).exists():
            
            mask_toxo = load_image(row_data.mask_toxo)
            bool_has_toxo = True
            mask_cell_toxo_intersection = mask_cell * (mask_toxo > 0) # IoU of toxo/cell
            list_roi_values = np.unique(mask_cell_toxo_intersection) # roi values with toxo
            
            # visualize toxo
            # fig, ax = plt.subplots(1,5)
            # ax[0].imshow(mask_cell)
            # ax[0].set_title("cell masks")
            # ax[0].set_axis_off()
            # ax[1].imshow(mask_toxo > 0)
            # ax[1].set_title("toxo masks")
            # ax[1].set_axis_off()
            # ax[2].imshow(mask_cell_toxo_intersection)
            # ax[2].set_title("intersection")
            # ax[2].set_axis_off()
            
            # ax[3].imshow(load_image(row_data.toxo_photons), vmax=30)
            # ax[3].set_title("toxo photons")
            # ax[3].set_axis_off()
            # im_kmeans = kmeans_threshold(load_image(row_data.toxo_photons), k=2, n_brightest_clusters=1)
            # ax[4].imshow(im_kmeans, vmax=1)
            # ax[4].set_title("toxo kmeans ")
            # ax[4].set_axis_off()
            # plt.show()
            
            
            for key in omi_props: 
                if omi_props[key]["mask_label"] in list_roi_values:
                    omi_props[key]["pixels_toxo"] = np.sum(mask_cell_toxo_intersection == omi_props[key]["mask_label"])
         
        
        ## create dataframe
        df_props = pd.DataFrame(omi_props).transpose()
        if bool_has_toxo:
            df_props["pixels_toxo"] = df_props["pixels_toxo"].fillna(0) 
        df_props.index.name = "base_name_with_roi"
        
        # save image id
        df_props["image_set"] = idx # * len(df_props)

        
        ## add other dictionary data to df
        for item_key in row_data.keys():
            pass
            df_props[item_key] = row_data[item_key]

        df_props.to_csv(path_output_features / f"{idx}.csv")  



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
    df_all_props.to_csv(path_feature_summaries / f"{d}_all_props.csv")
    
#%%

# plot all nadh chi

df_all_props = pd.read_csv(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\2022_04_11_all_props.csv")

plt.hist(df_all_props['nadh_chi_mean'], bins=100)
plt.title("nadh_chi_mean")
plt.show()

df_outliers = df_all_props[df_all_props['nadh_chi_median'] > 1.5]
set_paths_nadh_chi_files = natsorted(set(df_outliers["nadh_chi"]))

#%%
from ast import literal_eval
for path_im_chi_outlier in set_paths_nadh_chi_files[:]:
    pass
    
    df_subset = df_outliers[df_outliers["nadh_chi"] == path_im_chi_outlier]
    df_subset = df_subset.reset_index(drop=True)
    fig, ax = plt.subplots(1,4, figsize=(10,3))
    
    fig.suptitle(f"{df_subset.base_name_with_roi[0].rsplit('_',1)[0]}")
    ax[0].imshow(load_image(df_subset.nadh_photons[0]))
    ax[0].set_title('nadh photons')
    ax[0].set_axis_off()
    
    im_chi = load_image(df_subset.nadh_chi[0])
    ax[1].imshow(im_chi, vmax=1.5)
    ax[1].set_title('nadh chi')
    ax[1].set_axis_off()
    
    mask = load_image(df_subset.mask_cell[0])
    ax[2].imshow( (mask > 0) * im_chi, )#vmax=1.5
    ax[2].set_title('nadh chi masked')
    ax[2].set_axis_off()
    
    ax[3].imshow(mask)
    ax[3].set_title('mask')
    ax[3].set_axis_off()
    
    for idx, outlier_row in df_subset.iterrows():
        pass
        text = f"{outlier_row.nadh_chi_mean:.2f}"
        # text = f"{outlier_row.mask_label}"
        if outlier_row.mask_label == 71:
            print("roi 71")
        centroid = literal_eval(outlier_row["centroid"])
        ax[2].text(centroid[1],
                 centroid[0],
                 text,
                 c="w",
                 fontsize=6)
        ax[2].plot(centroid[1],
                     centroid[0],
                     marker='o',
                     markersize=1,
                     c="w")
    plt.show()
    


    
    
