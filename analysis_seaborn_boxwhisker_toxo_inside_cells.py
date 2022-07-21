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
from statannotations.Annotator import Annotator
from sklearn import preprocessing
from itertools import combinations
import natsort
import proplot as pplt
#%% import dataset 

analysis_type = 'toxo_inside_cells'

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_all_features = list(path_project.glob(f"*{analysis_type}.csv"))[0] 

# filename_csv = "2022_05_25_all_props.csv"
# path_all_features = list(path_project.glob("*props.csv"))[0] 
df_data = pd.read_csv(path_all_features)
path_output_figures = path_project / "figures" / analysis_type / "seaborn"

#%% PLOT PERCENT CAPTURED

# plt.hist(df_data['percent_toxo'], histtype='step', bins=100)
# plt.show()

for timepoint in np.unique(df_data['time_hours']):
    pass
    df_data_subset = df_data[df_data['time_hours']==timepoint]
    df_data_subset = df_data_subset[df_data_subset['percent_toxo'] > .05]
    #proplot
    # fig = pplt.figure()
    # ax = fig.add_subplot()
    # ax.hist(df_data_subset['percent_toxo'], histtype='step', bins=100)
    # ax.format(suptitle=f"percent toxo in cells \ntime point = {timepoint} \nmean: {np.mean(df_data_subset['percent_toxo']):.3f} \ntotal cells={len(df_data_subset)}",
    #           xlabel='percent toxo', 
    #           ylabel='cell count',
    #           # xlocator='null',
    #           # ylocator='null'
    #           )
    #plt
    plt.hist(df_data_subset['percent_toxo'], histtype='step', bins=10, label=timepoint)
plt.legend()
plt.show()