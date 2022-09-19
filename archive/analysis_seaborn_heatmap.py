import seaborn as sns 
# import cmcrameri.cm as cmc
import numpy as np
import matplotlib.pylab as plt
from natsort import natsorted
from flim_tools.io import read_asc
import matplotlib as mpl
mpl.rcParams["figure.dpi"]=300
from pathlib import Path
import pandas as pd

from sklearn import preprocessing
#%%


analysis_type = 'whole_cell_wo_toxo'
# analysis_type = 'whole_cell'
# analysis_type = 'toxo_inside_vs_outside'

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_all_features = list(path_project.glob(f"*{analysis_type}.csv"))[0] 

#%%

df_data = pd.read_csv(path_all_features)

#%%
df_plot_data = df_data[[
                        # nadh
                        'nadh_a1_mean', 
                        'nadh_a2_mean', 
                        'nadh_t1_mean', 
                        'nadh_t2_mean', 
                        'nadh_tau_mean_mean',
                        
                        # fad
                        'fad_a1_mean',
                        'fad_a2_mean', 
                        'fad_t1_mean',
                        'fad_t2_mean',
                        'fad_tau_mean_mean',
                        ]]


## scale data
scaler_robust = preprocessing.RobustScaler()
data = scaler_robust.fit_transform(df_plot_data)
                    
df_plot_data_scaled = pd.DataFrame(data, columns=df_plot_data.keys())                         


list_time_hours_colors = [list(np.random.choice(range(256), size=3)/256) for n in range(len(np.unique(df_data["time_hours"])))]
lut1 = dict(zip(np.unique(df_data["time_hours"]), list_time_hours_colors))

list_treatment_colors = [list(np.random.choice(range(256), size=3)/256) for n in range(len(np.unique(df_data["treatment"])))]
lut2 = dict(zip(np.unique(df_data["treatment"]), list_treatment_colors ))

row_colors_time_hours = df_data["time_hours"].map(lut1)
row_colors_treatment = df_data["treatment"].map(lut2)

row_colors = pd.concat([row_colors_time_hours,row_colors_treatment],axis=1)

clustermap = sns.clustermap(df_plot_data, # df_plot_data_scaled 
                            # cmap=cmc.batlow,
                            z_score=1,
                            # standard_scale=1,
                            row_colors = row_colors,
                            row_cluster=True
                            )

# add legends
for pos, label in enumerate(natsorted(df_data["time_hours"].unique())):
    clustermap.ax_col_dendrogram.bar(0, 0, color=list_time_hours_colors[pos], label=label, linewidth=0);
l1 = clustermap.ax_col_dendrogram.legend(title='Time Hours', loc="center", ncol=1, bbox_to_anchor=(1.25, 0.7), bbox_transform=plt.gcf().transFigure)

for pos, label in enumerate(df_data["treatment"].unique()):
    clustermap.ax_row_dendrogram.bar(0, 0, color=list_treatment_colors[pos], label=label, linewidth=0);
l2 = clustermap.ax_row_dendrogram.legend(title='treatment', loc="center", ncol=1, bbox_to_anchor=(1.25, 0.5), bbox_transform=plt.gcf().transFigure)

#%%

for key in df_plot_data:
    pass
    plt.title(key)
    plt.xlabel("entry")
    plt.xlabel("value")
    data = df_plot_data[key]
    print(f"{key} | min: {data.min():.4f}|  max: {data.max():.4f} |  stdev: {data.std():.4f} | {data.max()/data.std()}")
    # plt.plot(df_plot_data[key], 'o', markersize=2)
    plt.hist(df_plot_data[key], bins=100)
    plt.show()