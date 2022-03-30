import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

from pathlib import Path
import pandas as pd
import numpy as np
#%% import dataset 

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")

path_all_features = list(path_project.glob("*props.csv"))[0] 
#%% PLOT 

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

#%%


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
    
