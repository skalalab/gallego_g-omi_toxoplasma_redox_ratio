import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

from pathlib import Path
import pandas as pd
import numpy as np
#%% import dataset 

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")

filename_csv = "2022_05_25_all_props.csv"
# path_all_features = list(path_project.glob("*props.csv"))[0] 
df_data = pd.read_csv(path_project / filename_csv)
path_output_figures = path_project / "figures" / "holoviews"

#%% PLOT ALL REDOX RATIO BY EXPERIMENT, TREATMENT, HOURS 

kdims = [
        ("experiment", "Experiment"),
        ("treatment","Treatment"),
        ("time_hours","Time (hrs)")
        ]

vdims = [("redox_ratio_norm_mean","Redox Ratio Mean")]

bw_experiment_treatment_time = hv.BoxWhisker(df_data, vdims=vdims, kdims=kdims)

list_experiments = " | ".join(np.unique(df_data.experiment))
bw_experiment_treatment_time.opts(
    title=f"Redox Ratio | NAD(P)H/(NAD(P)H + FAD) \n Experiments: {list_experiments}",
    width=1600,
    height=800,
    tools=["hover"],
    xrotation=90
    )

hv.save(bw_experiment_treatment_time, path_output_figures / f"all_data_boxwhisker_experiment_treatment_time.html")
#%% PLOT LIFETIME PARAMETERS BY EXPERIMENT by image

for experiment in np.unique(df_data["experiment"]):
    pass
    df_props_exp = df_data[df_data["experiment"] == experiment]
    # list_image_names = [base_name.rsplit("_",1)[0] for base_name in list(df_props_exp["base_name"].values)]
    # df_props_exp["image_name"] = list_image_names
    
    kdims = [
            # ("experiment", "Experiment"),
            ("treatment","Treatment"),
            ("time_hours","Time (hrs)"),
            ("image_set","Image Set")
            ]
    
    vdims = [("redox_ratio_norm_mean","Redox Ratio Mean")]
    
    
    bw_treatment_time_image = hv.BoxWhisker(df_props_exp, vdims=vdims, kdims=kdims)
    
    bw_treatment_time_image.opts(
        title=f"Redox Ratio | NAD(P)H/(NAD(P)H + FAD)  \nExperiment: {experiment}",
        width = 2000,
        height = 800,
        tools=["hover"],
        xrotation=90
        )
    
    hv.save(bw_treatment_time_image, path_output_figures / f"{experiment}_by_images.html")
    
#%% PLOT MEDIA AND MEDIA+TOXO - ALL EXPERIMENTS

df_media_media_toxo = df_data[df_data["treatment"].isin(["media", "media+toxo"])]

kdims = [
        ("time_hours","Timepoint"),
        ("treatment","Media"),
    ]

vdims = [("redox_ratio_norm_mean","Redox Ratio")] 
boxwhisker_media_toxo = hv.BoxWhisker(df_media_media_toxo, kdims=kdims, vdims=vdims)

str_experiments = ' |'.join(np.unique(df_media_media_toxo.experiment))
boxwhisker_media_toxo.opts(
    width=1600,
    height=800,
    tools=["hover"],
    title=f"Redox Ratio \nExperiments: {str_experiments}",
    box_color="treatment",
    # cmap='Category20',
    cmap=[ '#FC5353','#1690FF'],
    )

hv.save(boxwhisker_media_toxo, path_output_figures / f"all_data_boxwhisker_media_media_toxo.html")

#%% MEDIA AND MEDIA+TOXO NORMALIZED TO MEDIA/CONTROL


df_media_media_toxo = df_data[df_data["treatment"].isin(["media", "media+toxo"])]

list_timepoints_hrs = np.unique(df_data["time_hours"])

# Determine mean of control('media') timepoints to normalize against
# compute means of timepoints just for the media condition
dict_means = { timepoint_hrs : np.mean(df_media_media_toxo.loc[
                                                (df_media_media_toxo["time_hours"] == timepoint_hrs) & \
                                                (df_media_media_toxo["treatment"] == "media")
                                              ]["redox_ratio_norm_mean"]
                                       ) for timepoint_hrs in list_timepoints_hrs  }

# for timepoint in dict_means:
#     pass

#     df_media_media_toxo[(df_media_media_toxo["time_hours"]==timepoint)]["redox_ratio_normalized"] = \
#         df_media_media_toxo[(df_media_media_toxo["time_hours"]==timepoint) ]["redox_ratio_mean"] - dict_means[timepoint]

# add new 
for idx, row_data in df_media_media_toxo.iterrows():
    pass
    df_media_media_toxo.loc[idx,"redox_ratio_normalized_by_mean"] = row_data.loc[("redox_ratio_norm_mean")] - dict_means[row_data.time_hours]

kdims = [
        ("time_hours","Timepoint"),
        ("treatment","Media"),
    ]

vdims = [("redox_ratio_normalized_by_mean","Redox Ratio")] 
boxwhisker_media_toxo = hv.BoxWhisker(df_media_media_toxo, kdims=kdims, vdims=vdims)

boxwhisker_media_toxo.opts(
    width=1600,
    height=800,
    tools=["hover"],
    title="Redox Ratio \nAll Data | normalized to media/control",
    box_color="treatment",
    cmap=[ '#FC5353','#1690FF'],
    )

hv.save(boxwhisker_media_toxo, path_output_figures / f"all_data_boxwhisker_normalized_media_media_toxo.html")

#%%
# parameters to plot
list_omi_parameters = {
    'NAD(P)H Intensity' :'nadh_intensity_mean',
    'NAD(P)H a1':'nadh_a1_mean',  
    'NAD(P)H a2':'nadh_a2_mean',
    'NAD(P)H t1':'nadh_t1_mean',  
    'NAD(P)H t2':'nadh_t2_mean',
    'NAD(P)H Tm':'nadh_tau_mean_mean', 
    'FAD Intensity':'fad_intensity_mean',  
    'FAD a1': 'fad_a1_mean',
    'FAD a2': 'fad_a2_mean',  
    'FAD t1': 'fad_t1_mean',
    'FAD t2': 'fad_t2_mean',  
    'FAD Tm' : 'fad_tau_mean_mean',
    'Redox Ratio' : 'redox_ratio_norm_mean'
    }
#%% PLOT ALL LIFETIME PARAMETERS - ALL EXPERIMENTS

df_media_media_toxo = df_data[df_data["treatment"].isin(["media", "media+toxo"])]

kdims = [
        ("time_hours","Timepoint"),
        ("treatment","Media"),
    ]

# iterate through the various paremeters
for dict_key in list_omi_parameters:
    pass
    vdims = [(list_omi_parameters[dict_key], dict_key)] 
    
    boxwhisker_media_toxo = hv.BoxWhisker(df_media_media_toxo, kdims=kdims, vdims=vdims)
    
    str_experiments = ' |'.join(np.unique(df_media_media_toxo.experiment))
    boxwhisker_media_toxo.opts(
        width=1600,
        height=800,
        tools=["hover"],
        title=f"{dict_key} \nExperiments: {str_experiments}",
        box_color="treatment",
        # cmap='Category20',
        cmap=[ '#FC5353','#1690FF'],
        )
    
    hv.save(boxwhisker_media_toxo, path_output_figures / f"all_data_boxwhisker_media_media_toxo_{dict_key}.html")


#%% PLOT LIFETIME PARAMETERS BY EXPERIMENT 

for experiment in np.unique(df_data["experiment"]):
    pass

    df_props_exp = df_data[df_data["experiment"] == experiment]
    df_props_exp = df_props_exp[df_props_exp["treatment"].isin(["media", "media+toxo"])]
 

    # list_image_names = [base_name.rsplit("_",1)[0] for base_name in list(df_props_exp["base_name"].values)]
    # df_props_exp["image_name"] = list_image_names
    
    
    kdims = [
            # ("experiment", "Experiment"),
            ("time_hours","Time (hrs)"),
            ("treatment","Treatment"),
            # ("image_set","Image Set")
            # ("experiment", "Experiment")
            ]
    for dict_key in list_omi_parameters:
        pass
        vdims = [(list_omi_parameters[dict_key], dict_key),
                 ] 
        # vdims = [("redox_ratio_norm_mean","Redox Ratio Mean")]
    
        bw_treatment_time_image = hv.BoxWhisker(df_props_exp, vdims=vdims, kdims=kdims)
        
        figure_title = f"{dict_key} | NAD(P)H/(NAD(P)H + FAD)" if dict_key == "Redox ratio" else f"{dict_key}"
        
        bw_treatment_time_image.opts(
            title=f"{figure_title} \nExperiment: {experiment}",
            width = 1000,
            height = 600,
            tools=["hover"],
            xrotation=90,
            cmap=[ '#FC5353','#1690FF'],
            box_color="treatment",
            )
        
        hv.save(bw_treatment_time_image, path_output_figures / f"{experiment}_{dict_key}.html")


