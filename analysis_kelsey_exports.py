from pathlib import Path
import pandas as pd
import numpy as np

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts


path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")

list_datasets = [path_project / "3-18-2019\kathe_031819modified3.csv", 
                 path_project / "4-6-2019\kathe040619modified4.csv"
                 ]

path_output_figures = path_project / "figures"
path_dataset = list_datasets[0]
df_data = pd.read_csv(path_dataset)

#%% Kelseys data

df_all_data = None
for dataset_name in list_datasets:
    if df_all_data is None:
        df_all_data = pd.read_csv(dataset_name)
        df_all_data["experiment"] = str(dataset_name.parent.stem)
    else:
        df_temp = pd.read_csv(dataset_name)
        df_temp["experiment"] = str(dataset_name.parent.stem)
        df_all_data = pd.concat([df_all_data, df_temp])
#%% PLOT ALL DATA

experiment = path_dataset.parent.name
kdims = [
        # ("experiment", "Experiment"),
        # ("treatment","Treatment"),
        ("Time","Time (hrs)"),
        ("Group","Treatment")
        ]

vdims = [("rr.mean","Redox Ratio Mean")]

bw_treatment_time_image = hv.BoxWhisker(df_all_data, vdims=vdims, kdims=kdims)

bw_treatment_time_image.opts(
    title=f"Redox Ratio \nAll Data | NAD(P)H/FAD",
    width = 2000,
    height = 800,
    tools=["hover"],
    xrotation=90
    )
hv.save(bw_treatment_time_image, path_output_figures / f"kelsey_boswhisker_redox_ratio_all_data.html")
#%% PLOT REDOX RATIO BY EXPERIMENT 

# for experiment in np.unique(df_data["experiment"]):
#     pass
#     df_props_exp = df_data[df_data["experiment"] == experiment]
    # list_image_names = [base_name.rsplit("_",1)[0] for base_name in list(df_props_exp["base_name"].values)]
    # df_props_exp["image_name"] = list_image_names

experiment = path_dataset.parent.name
kdims = [
        # ("experiment", "Experiment"),
        # ("treatment","Treatment"),
        ("Time","Time (hrs)"),
        ("Group","Treatment")
        ]

vdims = [("rr.mean","Redox Ratio Mean")]


bw_treatment_time_image = hv.BoxWhisker(df_data, vdims=vdims, kdims=kdims)

bw_treatment_time_image.opts(
    title=f"Redox Ratio \n{experiment}",
    width = 2000,
    height = 800,
    tools=["hover"],
    xrotation=90
    )
hv.save(bw_treatment_time_image, path_output_figures / f"kelsey_boswhisker_redox_ratio_{experiment}.html")