#%% UMAP Coordinate calculations

# Loads in all required dependencies
from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

# import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mpl.rcParams["figure.dpi"] = 300
plt.imshow(np.zeros((512, 512)))
plt.show()

#%% LOAD AND SELECT DATA

path_project = Path(r"/mnt/Z/0-Projects and Experiments/GG - toxo_omi_redox_ratio/")
path_all_features = list(path_project.glob(f"*all_props_cells.csv"))[0] 
df_data = pd.read_csv(path_all_features)

df_data = df_data[(df_data['time_hours'] != 3) &
                  (df_data['time_hours'] != 72) 
                  # (df_data['time_hours'] != 48) & 
                  # (df_data['treatment'].isin(['media', 'media+toxo']) )
                      ]

#%%
analysis_type = "whole_cell"
# analysis_type = "nuclei"
path_analysis = Path(
    r"Z:\0-Projects and Experiments\RD - redox_ratio_development\Data Combined + QC Complete\0-analysis"
)
filename = f"2022_02_28_{analysis_type}_all_props.csv"

base_path_output = Path(
    r"Z:\0-Projects and Experiments\RD - redox_ratio_development\Data Combined + QC Complete\0-figures"
)

path_figures = base_path_output / analysis_type / "umap"
path_figures.mkdir(exist_ok=True)

df_redox_ratio = pd.read_csv(path_analysis / filename)
# (27,35,37,38)
# df_redox_ratio.iloc[:,35]

list_omi_parameters = [
    "nadh_intensity_mean",
    "nadh_a1_mean",
    "nadh_a2_mean",
    "nadh_t1_mean",
    "nadh_t2_mean",
    "nadh_tau_mean_mean",
    "fad_intensity_mean",
    "fad_a1_mean",
    "fad_a2_mean",
    "fad_t1_mean",
    "fad_t2_mean",
    "fad_tau_mean_mean",
    "redox_ratio_mean",
]

# df_data = df_redox_ratio[df_redox_ratio["treatment"] == "0-control"]

#%% seahorse regular media
# filename_id = "exp_seahose_media_DMEM"
# df_exp_subset = df_redox_ratio[df_redox_ratio["experiment"] == "glucose"]
# df_media_subset = df_exp_subset[df_exp_subset["media"] == "DMEM"]

# list_cell_lines = np.unique(df_media_subset["cell_line"])

#%% glucose
filename_id = "exp_glucose"
experiment = "glucose"
df_exp_subset = df_redox_ratio[df_redox_ratio["experiment"] == experiment]
# df_media_subset = df_exp_subset[df_exp_subset["media"] == "DMEM"]

list_cell_lines = np.unique(df_exp_subset["cell_line"])

path_figures_exp = path_figures / experiment
path_figures_exp.mkdir(exist_ok=True)

#%%

for cell_line in list_cell_lines[:]:  # iterate throug the cell lines
    pass

    dict_params = list(
        ParameterGrid(
            {
                "metric": ["euclidean", "cosine", "manhattan"],
                "n_neighbors": np.arange(15, 100, step=20),
                "min_dist": np.arange(0.1, 1.0, step=0.3),
            }
        )
    )

    ## make dirs for this cell line
    path_figures_cell_line = path_figures_exp / f"{cell_line}"
    path_figures_cell_line.mkdir(exist_ok=True)

    df_data = df_exp_subset[df_exp_subset["cell_line"] == cell_line]

    data = df_data[list_omi_parameters].values
    scaled_data = StandardScaler().fit_transform(data)
    ##%% FIT UMAP

    for params in tqdm(dict_params):
        pass
        # reducer = umap.UMAP()
        reducer = umap.UMAP(
            # n_neighbors=15,
            # min_dist=0.1,
            # metric='euclidean',
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            metric=params["metric"],
            n_components=2,
            random_state=0,
        )

        fit_umap = reducer.fit(scaled_data)
        ##%% PLOT UMAP

        import holoviews as hv

        hv.extension("bokeh")
        from holoviews import opts

        # import hvplot.pandas

        ## additional params
        hover_vdim = "base_name"
        legend_entries = "treatment"  # "cell_line"

        ########
        df_data = df_data.copy()
        df_data["umap_x"] = fit_umap.embedding_[:, 0]
        df_data["umap_y"] = fit_umap.embedding_[:, 1]

        kdims = ["umap_x"]
        vdims = ["umap_y", hover_vdim]
        list_entries = np.unique(df_data[legend_entries])

        umap_parameters = (
            f"masks: {analysis_type} | "
            f"metric: {reducer.metric} | "
            f"n_neighbors: {reducer.n_neighbors} | "
            f"distance: {reducer.min_dist:.2f} | "
            f"{filename_id}"
        )

        scatter_umaps = [
            hv.Scatter(
                df_data[df_data[legend_entries] == entry],
                kdims=kdims,
                vdims=vdims,
                label=entry,
            )
            for entry in list_entries
        ]

        overlay = hv.Overlay(scatter_umaps)
        overlay.opts(
            opts.Scatter(
                tools=["hover"], muted_alpha=0, aspect="equal", width=1600, height=800
            ),
            opts.Overlay(
                title=f"UMAP | {cell_line} \n {umap_parameters}",
                legend_opts={"click_policy": "hide"},
                legend_position="right",
            ),
        )

        filename = f"{analysis_type}_metric_{reducer.metric}_nneighbors_{reducer.n_neighbors}_mindist_{reducer.min_dist:.2f}"

        # holoviews
        hv.save(
            overlay,
            path_figures_cell_line / f"umap_{filename}_{filename_id}_{cell_line}.html",
        )

#%%
# hvplot
# overlay = df_data.hvplot.scatter(x='umap_x', y='umap_y',
#                                   by=legend_entries,
#                                   s=10,
#                                   title=f"{cell_line}  |  {filename}",
#                                   aspect="equal",
#                                   hover_cols=["base_name",
#                                               "treatment",
#                                               "cell_line",
#                                               "media"]
#                                   ).opts(
#                                                     width=1600,
#                                                     height=800,
#                                                     # aspect="equal",
#                                                     legend_opts={"click_policy": "hide"},
#     )

# hvplot.save(overlay, path_figures / f"umap_{filename}_{filename_id}_{cell_line}_hvplot.html")


#%% PLOT UMAP EMBEDDINGS DATA

# umap_embeddings = reducer.fit_transform(data)

# umap_embeddings = draw_umap(scaled_penguin_data).fit_transform(data)

# # print(embedding.shape)

# fig = plt.figure()
# if n_components == 1:
#     ax = fig.add_subplot(111)
#     ax.scatter(umap_points[:,0], range(len(umap_points)), labels=labels , c=[sns.color_palette()[x] for x in labels.map({"SCM":0, "nonSCM":1})])
# if n_components == 2: # plot in 2d
#     # ax = fig.add_subplot(111)
#     # ax.scatter(u[:,0], u[:,1], labels=list(labels), c=[sns.color_palette()[x] for x in labels.map({"SCM":0, "nonSCM":1})])
#     #print(u)
#     sns.scatterplot(umap_points[:,0], umap_points[:,1], hue=list(labels), alpha=0.5)
# if n_components == 3: #plot in 3d
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(umap_points[:,0], umap_points[:,1],
#                umap_points[:,2],
#                c=[sns.color_palette()[x] for x in labels.map({"SCM":0, "nonSCM":1})], s=100)
# plt.legend(loc="upper right", prop={'size': 16})
# plt.title(title, fontsize=18)
# plt.show()


#%% OPTIONAL GENERATE MULTIPLE UMAPS
# n_neighbors: captures local vs global features. Small n local, large n global. Range 1-200
# min_dist: how closely points are allowed to pack together. Range 0-.99
# n_components: 1 2 or 3d plot
# metric: distance metric between data

# param_grid = {
#     "n_neighbors":  [10], # [2, 5, 10, 20, 50, 100, 200],
#     "metric": [
#                 "euclidean", ##### aka l2,  straight line
#                 "manhattan", ##### aka l1, block by block
#                 "chebyshev", ##### chessboard distance (8 degrees of movement but only one space at a time (King))
#                 # # # # # "minkowski", # generalization of both the Euclidean distance and the Manhattan distance
#                 "canberra", ##### weighted version of L1 (Manhattan) distance
#                 # # "braycurtis",
#                 # ### "haversine", # error when running
#                 # ### "mahalanobis", # error when running
#                 # # "wminkowski",
#                 # ### "seuclidean", # error when running - division by zero
#                 "cosine",
#                 "correlation", ##### Pearson correlation
#                 # "hamming",
#                 # "jaccard",
#                 # "dice",
#                 # "russellrao",
#                 # "kulsinski",
#                 # "rogerstanimoto",
#                 # "sokalmichener",
#                 # "sokalsneath",
#                 # "yule"
#                 ]
#     }

# parameters = list(ParameterGrid(param_grid))

# returned_umap = None
# scaled_data_umap = StandardScaler().fit_transform(data_umap.copy()) # don't forget this step!!
# for n in parameters:
#     print(f"parameters: {n}")
#     returned_umap = draw_umap(scaled_data_umap, n_neighbors=n['n_neighbors'], \
#               metric=n['metric'], n_components=2, min_dist=0.3, \
#                   title=f'parameters = {n}', labels=labels, random_state=0)

#%% UMAP documentation Penguin
# https://umap-learn.readthedocs.io/en/latest/basic_usage.html#penguin-data
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import umap

# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
# print(penguins.head())

# penguins = penguins.dropna()
# penguins.species_short.value_counts()

# # sns.pairplot(penguins, hue='species_short')
# # plt.show()


# penguin_data = penguins[
#     [
#         "culmen_length_mm",
#         "culmen_depth_mm",
#         "flipper_length_mm",
#         "body_mass_g",
#     ]
# ].values

# scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

# reducer = umap.UMAP(random_state=0)
# embedding = reducer.fit_transform(scaled_penguin_data)
# print(embedding.shape)

# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Penguin dataset', fontsize=24)
# plt.show()

#%%

if __file__ == "__main__":
    pass

    # load data
    # setup reducer
    # plot data