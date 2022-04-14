import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

from pathlib import Path
import pandas as pd
import numpy as np


from flim_tools.visualization import compute_umap
#%% import dataset 

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")

path_all_features = list(path_project.glob("*props.csv"))[0] 
df_data = pd.read_csv(path_all_features)
path_output_figures = path_project / "figures" / "umap"




#%% plot data 

def plot_data(df_data, reducer, title):
    import holoviews as hv
    hv.extension("bokeh")
    from holoviews import opts
    
    ## additional params
    hover_vdim = "image_set"
    legend_entries = "treatment" # "cell_line"
    
    kdims = ["umap_x"]
    vdims = ["umap_y", hover_vdim]
    list_entries = np.unique(df_data[legend_entries])
    
    umap_parameters =   f"metric: {reducer.metric} | " \
                        f"n_neighbors: {reducer.n_neighbors} | " \
                        f"distance: {reducer.min_dist:.2f} | " \
                        
    scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                                vdims=vdims, label=entry) for entry in list_entries]
    
    overlay = hv.Overlay(scatter_umaps)
    overlay.opts(
        opts.Scatter(
            tools=["hover"],
            muted_alpha=0,
            aspect="equal",
            width=1600, 
            height=800,
            ),
        opts.Overlay(
            title=f"Timepoint: {title} |  \n {umap_parameters}",
            legend_opts={"click_policy": "hide"},
            legend_position='right',
            )       
        )
    
                              
    filename=f"umap_metric_{reducer.metric}_nneighbors_{reducer.n_neighbors}_mindist_{reducer.min_dist:.2f}"
    
    #holoviews
    hv.save(overlay, path_output_figures / f"{filename}_{title}.html" )


#%% compute UMAP

for timepoint in np.unique(df_data["time_hours"]):
    pass
    
    df_data_subset = df_data[df_data['time_hours']==timepoint]
    df_plot_data = df_data_subset[[
                            # nadh
                            'nadh_intensity_mean',
                            'nadh_a1_mean', 
                            'nadh_a2_mean', 
                            'nadh_t1_mean', 
                            'nadh_t2_mean', 
                            'nadh_tau_mean_mean',
                            
                            # fad
                            'fad_intensity_mean',
                            'fad_a1_mean',
                            'fad_a2_mean', 
                            'fad_t1_mean',
                            'fad_t2_mean',
                            'fad_tau_mean_mean',
                            
                            # rr
                            "redox_ratio_mean",
                            ]]
    
    df_data_subset = df_data_subset.reset_index(drop=True)
    df_umap, reducer = compute_umap(df_plot_data.values)
    df_data_subset = pd.concat([df_data_subset, df_umap], axis=1)
    plot_data(df_data_subset, reducer, title=f'{timepoint}')
