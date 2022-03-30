from pathlib import Path
from flim_tools.flim import regionprops_omi
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
from tqdm import tqdm
import pandas as pd
from helper import load_image, visualize_dictionary
from datetime import date


path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_datasets = path_project / "dictionaries"
path_output_features = path_project / "features"

list_dataset_dicts = list(path_datasets.glob("*.csv"))

#%% COMPUTE OMI PARAMETERS

# iterate through dictionaries
for dict_dataset in tqdm(list_dataset_dicts[:]):
    pass
    df_data = pd.read_csv(dict_dataset, index_col=("index"))
    
    # itereate through rows of dict
    for idx, row_data in tqdm(list(df_data.iterrows())[:]):#iterate through sets
        pass
        
        # visualize_dictionary(idx, row_data)
        
        nadh_photons = load_image(row_data.nadh_photons)
        nadh_a1 = load_image(row_data.nadh_a1)
        nadh_a2 = load_image(row_data.nadh_a2)
        nadh_t1 = load_image(row_data.nadh_t1)
        nadh_t2 = load_image(row_data.nadh_t2)
        
        fad_photons = load_image(row_data.fad_photons)
        fad_a1 = load_image(row_data.fad_a1)
        fad_a2 = load_image(row_data.fad_a2)
        fad_t1 = load_image(row_data.fad_t1)
        fad_t2 = load_image(row_data.fad_t2)
        
        label_image = load_image(row_data.mask_cell)
        
        omi_props = regionprops_omi(idx, 
                                     label_image, 
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
                                     other_props = ["area",
                                                    "perimeter",
                                                    "eccentricity"
                                                    ])
        ## create dataframe
        df_props = pd.DataFrame(omi_props).transpose()
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
