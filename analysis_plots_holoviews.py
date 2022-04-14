import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

from pathlib import Path
import pandas as pd
import numpy as np
#%% import dataset 

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")

path_all_features = list(path_project.glob("*props.csv"))[0] 
df_data = pd.read_csv(path_all_features)
path_output_figures = path_project / "figures"

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

hv.save(bw_experiment_treatment_time, path_output_figures / f"boxwhisker_experiment_treatment_time.html")
#%% PLOT REDOX RATIO BY EXPERIMENT 

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
    
#%% PLOT MEDIA  AND MEDIA+TOXO

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

hv.save(boxwhisker_media_toxo, path_output_figures / f"boxwhisker_media_media_toxo.html")

#%% MEDIA AND MEDIA+TOXO NORMALIZED TO MEDIA/CONTROL


df_media_media_toxo = df_data[df_data["treatment"].isin(["media", "media+toxo"])]

list_timepoints_hrs = np.unique(df_data["time_hours"])

# redox ratio means across timepoints for control/media condition
dict_means = { timepoint_hrs : np.mean(df_media_media_toxo.loc[(df_media_media_toxo["time_hours"] == timepoint_hrs) & \
                                                            (df_media_media_toxo["treatment"] == "media")]["redox_ratio_norm_mean"]) \
                                                            for timepoint_hrs in list_timepoints_hrs  }

# df_media_media_toxo["redox_ratio_normalized"] = np.NaN

# for timepoint in dict_means:
#     pass

#     df_media_media_toxo[(df_media_media_toxo["time_hours"]==timepoint)]["redox_ratio_normalized"] = \
#         df_media_media_toxo[(df_media_media_toxo["time_hours"]==timepoint) ]["redox_ratio_mean"] - dict_means[timepoint]

for idx, row_data in df_media_media_toxo.iterrows():
    pass
    df_media_media_toxo.loc[idx,"redox_ratio_normalized"] = row_data.loc[("redox_ratio_norm_mean")] - dict_means[row_data.time_hours]

kdims = [
        ("time_hours","Timepoint"),
        ("treatment","Media"),

    ]

vdims = [("redox_ratio_normalized","Redox Ratio")] 
boxwhisker_media_toxo = hv.BoxWhisker(df_media_media_toxo, kdims=kdims, vdims=vdims)

boxwhisker_media_toxo.opts(
    width=1600,
    height=800,
    tools=["hover"],
    title="Redox Ratio \nAll Data | normalized to media/control"
    
    )

hv.save(boxwhisker_media_toxo, path_output_figures / f"boxwhisker_normalized_media_media_toxo.html")


#%% UMAP


list_omi_parameters = [
    'nadh_intensity_mean',
    'nadh_a1_mean',  
    'nadh_a2_mean',
    'nadh_t1_mean',  
    'nadh_t2_mean',
    'nadh_tau_mean_mean', 
    'fad_intensity_mean',  
    'fad_a1_mean',
    'fad_a2_mean',  
    'fad_t1_mean',
    'fad_t2_mean',  
    'fad_tau_mean_mean',
    'redox_ratio_mean'
    ]



