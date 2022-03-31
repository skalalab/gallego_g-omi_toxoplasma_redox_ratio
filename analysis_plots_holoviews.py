import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

from pathlib import Path
import pandas as pd
import numpy as np
#%% import dataset 

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")

path_all_features = list(path_project.glob("*props.csv"))[0] 
#%% PLOT REDOX RATIO BY EXPERIMENT, TREATMENT, HOURS 

df_data = pd.read_csv(path_all_features)
# assemble 

path_output_figures = path_project / "figures"


kdims = [
        ("experiment", "Experiment"),
        ("treatment","Treatment"),
        ("time_hours","Time (hrs)")
        ]

vdims = [("redox_ratio_mean","Redox Ratio Mean")]


bw_experiment_treatment_time = hv.BoxWhisker(df_data, vdims=vdims, kdims=kdims)


bw_experiment_treatment_time.opts(
    title=f"Redox Ratio \n All Data",
    width=1600,
    height=800,
    tools=["hover"],
    xrotation=90
    )

hv.save(bw_experiment_treatment_time, path_output_figures / f"bw_experiment_treatment_time.html")

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
    
    vdims = [("redox_ratio_mean","Redox Ratio Mean")]
    
    
    bw_treatment_time_image = hv.BoxWhisker(df_data, vdims=vdims, kdims=kdims)
    
    bw_treatment_time_image.opts(
        title=f"Redox Ratio \nExperiment: {experiment}",
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

vdims = [("redox_ratio_mean","Redox Ratio")] 
bars_media_toxo = hv.Bars(df_media_media_toxo, kdims=kdims, vdims=vdims)
boxwhisker_media_toxo = hv.BoxWhisker(df_media_media_toxo, kdims=kdims, vdims=vdims)


bars_media_toxo.opts(
    width=1600,
    height=800,
    tools=["hover"],
    title="Redox Ratio \nAll Data"
    
    )

boxwhisker_media_toxo.opts(
    width=1600,
    height=800,
    tools=["hover"],
    title="Redox Ratio \nAll Data"
    
    )

hv.save(bars_media_toxo, path_output_figures / f"bars_media_media_toxo.html")
hv.save(boxwhisker_media_toxo, path_output_figures / f"boxwhisker_media_media_toxo.html")


#%% MEDIA AND MEDIA+TOXO NORMALIZED TO MEDIA/CONTROL


df_media_media_toxo = df_data[df_data["treatment"].isin(["media", "media+toxo"])]

list_timepoints_hrs = np.unique(df_data["time_hours"])

# redox ratio means across timepoints for control/media condition
dict_means = { timepoint_hrs : np.mean(df_media_media_toxo.loc[(df_media_media_toxo["time_hours"]== timepoint_hrs) & \
                                                            (df_media_media_toxo["treatment"] == "media")]["redox_ratio_mean"]) \
                                                            for timepoint_hrs in list_timepoints_hrs  }

df_media_media_toxo["redox_ratio_normalized"] = np.NaN

# for timepoint in dict_means:
#     pass

#     df_media_media_toxo[(df_media_media_toxo["time_hours"]==timepoint)]["redox_ratio_normalized"] = \
#         df_media_media_toxo[(df_media_media_toxo["time_hours"]==timepoint) ]["redox_ratio_mean"] - dict_means[timepoint]

for idx, row_data in df_media_media_toxo.iterrows():
    pass
    df_media_media_toxo.loc[idx,"redox_ratio_normalized"] = row_data["redox_ratio_mean"] - dict_means[row_data.time_hours]

kdims = [
        ("time_hours","Timepoint"),
        ("treatment","Media"),

    ]

vdims = [("redox_ratio_normalized","Redox Ratio")] 
bars_media_toxo = hv.Bars(df_media_media_toxo, kdims=kdims, vdims=vdims)
boxwhisker_media_toxo = hv.BoxWhisker(df_media_media_toxo, kdims=kdims, vdims=vdims)


bars_media_toxo.opts(
    width=1600,
    height=800,
    tools=["hover"],
    title="Redox Ratio \nAll Data"
    
    )

boxwhisker_media_toxo.opts(
    width=1600,
    height=800,
    tools=["hover"],
    title="Redox Ratio \nAll Data"
    
    )

hv.save(bars_media_toxo, path_output_figures / f"bars_normalized_media_media_toxo.html")
hv.save(boxwhisker_media_toxo, path_output_figures / f"boxwhisker_normalized_media_media_toxo.html")













