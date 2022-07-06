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
#%% import dataset 

# analysis_type = 'whole_cell_wo_toxo'
# analysis_type = 'whole_cell'
analysis_type = 'toxo_inside_vs_outside'

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_all_features = list(path_project.glob(f"*{analysis_type}.csv"))[0] 

# filename_csv = "2022_05_25_all_props.csv"
# path_all_features = list(path_project.glob("*props.csv"))[0] 
df_data = pd.read_csv(path_all_features)
path_output_figures = path_project / "figures" / analysis_type / "seaborn"

#%% PLOT ALL by timepoint 

# kdims = [
#         ("experiment", "Experiment"),
#         ("treatment","Treatment"),
#         ("time_hours","Time (hrs)")
#         ]

# vdims = [("redox_ratio_norm_mean","Redox Ratio Mean")]

# bw_experiment_treatment_time = hv.BoxWhisker(df_data, vdims=vdims, kdims=kdims)

# list_experiments = " | ".join(np.unique(df_data.experiment))
# bw_experiment_treatment_time.opts(
#     title=f"Redox Ratio | NAD(P)H/(NAD(P)H + FAD) \n Experiments: {list_experiments}",
#     width=1600,
#     height=800,
#     tools=["hover"],
#     xrotation=90
#     )
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
#%% MEDIA VS MEDIA+TOXO - ALL EXPERIMENTS

p_values = f"ns: p <= 1 | "\
           f"*: .01 < p <= .05  | "\
           f"**: .001.00e-03 < p <= .01  | "\
           f"***: .0001 < p <= .001  | "\
           f"****: p <= .0001"

mpl.rcParams['figure.figsize'] = 11.7,8.27
for dict_key in list_omi_parameters:
    pass
    palette ={"media": '#1690FF', "media+toxo": '#FC5353'}

    data = df_data[df_data['treatment'].isin(['media','media+toxo'])]
    data = data.astype({"time_hours" : str})
    
    x = "time_hours"
    y = list_omi_parameters[dict_key]
    hue = "treatment"
    hue_order=['media','media+toxo']
    order = list(map(str,np.unique(df_data['time_hours'])))
    pairs=[((str(x) , 'media'), (str(x),'media+toxo')) for x in np.unique(df_data["time_hours"]) ]
        
    ax = sns.boxplot(
                    x=x, 
                    y=y, 
                    hue=hue, 
                    # kind="box", 
                    data=data,
                    palette=palette, 
                    hue_order=hue_order,
                    order=order,
                )
    
    # ax = sns.violinplot(
    #                 x=x, 
    #                 y=y, 
    #                 hue=hue, 
    #                 # kind="box", 
    #                 data=data,
    #                 palette=palette, 
    #                 hue_order=hue_order,
    #                 order=order,
    #                 dodge=True
    #             )
    figure_title = f"{dict_key} | NAD(P)H/(NAD(P)H + FAD)" if dict_key == "Redox ratio" else f"{dict_key}"
    plt.title(f"{figure_title} \n {p_values}")
    plt.xlabel("Time Point (hrs)")
    plt.ylabel(dict_key)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)

    # Annotations
    annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    # Finally save fig    
    plt.savefig(path_output_figures / f"all_data_{dict_key}.svg", bbox_inches='tight')
    plt.show()

#%% MEDIA VS MEDIA+TOXO - BY EXPERIMENTS

p_values = f"ns: p <= 1 | "\
           f"*: .01 < p <= .05  | "\
           f"**: .001.00e-03 < p <= .01  | "\
           f"***: .0001 < p <= .001  | "\
           f"****: p <= .0001"


# mpl.rcParams['figure.figsize'] = 11.7,8.27


for experiment in np.unique(df_data['experiment']):
    pass

    for dict_key in list_omi_parameters:
        pass

        palette ={"media": '#1690FF', "media+toxo": '#FC5353'}
    
        data = df_data[df_data['treatment'].isin(['media','media+toxo'])]
        
        ## select only experiment data
        data = data[data['experiment'] == experiment]
        ##
        data = data.astype({"time_hours" : str})
        
        x = "time_hours"
        y = list_omi_parameters[dict_key]
        hue = "treatment"
        hue_order=['media','media+toxo']
        order = natsorted(list(map(str,np.unique(data['time_hours']))))
        
        pairs=[((x , 'media'), (x,'media+toxo')) for x in np.unique(data["time_hours"]) ]
            
        ax = sns.boxplot(
                        x=x, 
                        y=y, 
                        hue=hue, 
                        # kind="box", 
                        data=data,
                        palette=palette, 
                        hue_order=hue_order,
                        order=order,
                    )
        
        # ax = sns.violinplot(
        #                 x=x, 
        #                 y=y, 
        #                 hue=hue, 
        #                 # kind="box", 
        #                 data=data,
        #                 palette=palette, 
        #                 hue_order=hue_order,
        #                 order=order,
        #                 dodge=True
        #             )
        figure_title = f"{dict_key} | Dataset: {experiment} | NAD(P)H/(NAD(P)H + FAD)" if dict_key == "Redox ratio" else f"{dict_key} | Dataset: {experiment}"
        plt.title(f"{figure_title} \n {p_values}")
        plt.xlabel("Time Point (hrs)")
        plt.ylabel(dict_key)
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    
        # Annotations
        annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside')
        annotator.apply_and_annotate()
    
        # Finally save fig    
        plt.savefig(path_output_figures / f"{experiment}_{dict_key}.svg", bbox_inches='tight')
        plt.show()
        # plt.close()