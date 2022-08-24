import seaborn as sns 
# import cmcrameri.cm as cmc
import numpy as np
import matplotlib.pylab as plt
from natsort import natsorted
from cell_analysis_tools.io import read_asc
import matplotlib as mpl
mpl.rcParams["figure.dpi"]=300
from pathlib import Path
import pandas as pd
from statannotations.Annotator import Annotator
from sklearn import preprocessing
from itertools import combinations
import natsort
import proplot as pplt
#%% Load data
path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_all_features = list(path_project.glob(f"*all_props_cells.csv"))[0] 
df_data = pd.read_csv(path_all_features)

df_data = df_data[(df_data['time_hours'] != 3) & (df_data['time_hours'] != 72) ]

###
# df_toxo = df_data[(df_data['treatment']=='media+toxo')]
# plt.title("percent toxo in media+toxo cells")
# plt.hist(df_toxo['percent_toxo'], histtype="step", bins=100)
# plt.show()


#%% parameter keys 

LIST_OMI_PARAMETERS = {
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

#%% TOXO CONDITION --> LOW VS HIGH TOXO 

analysis_type = 'toxo_inside_cells_high_vs_low'
path_output_figures = path_project / "figures" / analysis_type / "seaborn"

for experiment in np.unique(df_data['experiment']):
    pass

    ##%% PLOT PERCENT TOXO CAPTURED -- ALL IN ONE FIGURE
    for timepoint in np.unique(df_data['time_hours']):
        pass
        df_data_subset = df_data[(df_data['time_hours']==timepoint) & (df_data['experiment']==experiment)]
        plt.hist(df_data_subset['percent_toxo'], histtype='step', bins=20, label=timepoint)
    plt.title(f"Percent toxo captured per timepoint \nexperiment: {experiment}")
    plt.legend()
    plt.savefig(path_output_figures / f"percent_toxo_{experiment}_{analysis_type}.png", bbox_inches='tight')
    plt.show()
    
    ##%% PLOT PERCENT CAPTURED -- ONE FIGURE PER TIMEPOINT per experiment
    for timepoint in np.unique(df_data['time_hours']):
        pass
        df_data_subset = df_data[(df_data['time_hours']==timepoint) & (df_data['experiment']==experiment)]
        #proplot
        fig = pplt.figure()
        ax = fig.add_subplot()
        ax.hist(df_data_subset['percent_toxo'], histtype='step', bins=20)
        ax.format(suptitle=f"percent toxo in cells \n experiment: {experiment}  \ntime point = {timepoint} \nmean: {np.mean(df_data_subset['percent_toxo']):.3f} \ntotal cells={len(df_data_subset)}",
                  xlabel='percent toxo', 
                  ylabel='cell count',
                  )
        plt.savefig(path_output_figures / f"percent_toxo_{experiment}_timepoint_{timepoint}_{analysis_type}.png", bbox_inches='tight')
        plt.show()
#%%
##########%% MEDIA+TOXO - LOW VS HIGH TOXO --> ALL EXPERIMENTS

p_values = "ns: p <= 1 | "\
           "*: .01 < p <= .05  | "\
           "**: .001 < p <= .01  | "\
           "***: .0001 < p <= .001  | "\
           "****: p <= .0001"

mpl.rcParams['figure.figsize'] = 11.7,8.27
for dict_key in LIST_OMI_PARAMETERS:
    pass
    palette ={"low_toxo": '#1690FF', "high_toxo": '#FC5353'}

    data = df_data[df_data['treatment'].isin(['media+toxo'])]
    data = data.astype({"time_hours" : str})
    
    ## threshold df by percent_toxo here
    threshold_percent_toxo = 0.1
    
    # data = data.drop(data[(data['treatment']=='media+toxo')&(data['percent_toxo'] < threshold_percent_toxo)].index)
    
    # add col for low and high toxo content
    data['toxo_class'] = np.where(data['percent_toxo'] < threshold_percent_toxo, 'low_toxo', 'high_toxo')
    
    x = "time_hours"
    y = LIST_OMI_PARAMETERS[dict_key]
    hue = "toxo_class"
    hue_order=['low_toxo','high_toxo']
    order = list(map(str,np.unique(df_data['time_hours'])))
    pairs=[((str(x) , 'low_toxo'), (str(x),'high_toxo')) for x in np.unique(df_data["time_hours"]) ]
        
    ax = sns.boxplot(
                    x=x, 
                    y=y, 
                    hue=hue, 
                    data=data,
                    palette=palette, 
                    hue_order=hue_order,
                    order=order,
                )
    figure_title = f"{dict_key} | NAD(P)H/(NAD(P)H + FAD) \npercent threshold toxo content: {threshold_percent_toxo} | cell count low:high = {sum(data['toxo_class']=='low_toxo')}:{sum(data['toxo_class']=='high_toxo')}" \
        if dict_key == "Redox Ratio" else f"{dict_key} \npercent threshold toxo content: {threshold_percent_toxo} | cell count low:high = {sum(data['toxo_class']=='low_toxo')}:{sum(data['toxo_class']=='high_toxo')}"
    plt.title(f"{analysis_type} \n{figure_title} \n {p_values}")
    plt.xlabel("Time Point (hrs)")
    plt.ylabel(dict_key)
    if dict_key == "Redox Ratio":
        plt.ylim((0,1))
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    # plt.show()

    # Annotations
    annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    # Finally save fig    
    plt.savefig(path_output_figures / f"threshold_{threshold_percent_toxo}_all_data_{analysis_type}_{dict_key}.png", bbox_inches='tight')
    plt.show()

#%% CONTROL VS HIGH TOXO CELLS
analysis_type = 'whole_cell_vs_high_toxo'
path_output_figures = path_project / "figures" / analysis_type / "seaborn"

##########%% MEDIA VS MEDIA+TOXO - ALL EXPERIMENTS

p_values = "ns: p <= 1 | "\
           "*: .01 < p <= .05  | "\
           "**: .001 < p <= .01  | "\
           "***: .0001 < p <= .001  | "\
           "****: p <= .0001"

mpl.rcParams['figure.figsize'] = 11.7,8.27
for dict_key in LIST_OMI_PARAMETERS:
    pass
    palette ={"media": '#1690FF', "media+toxo": '#FC5353'}

    data = df_data[df_data['treatment'].isin(['media','media+toxo'])]
    data = data.astype({"time_hours" : str})
    
    ## threshold df by percent_toxo here
    threshold_percent_toxo = 0.1
    data = data.drop(data[(data['treatment']=='media+toxo')&(data['percent_toxo'] < threshold_percent_toxo)].index)
    
    x = "time_hours"
    y = LIST_OMI_PARAMETERS[dict_key]
    hue = "treatment"
    hue_order=['media','media+toxo']
    order = list(map(str,np.unique(df_data['time_hours'])))
    pairs=[((str(x) , 'media'), (str(x),'media+toxo')) for x in np.unique(df_data["time_hours"]) ]
    
    ############ NORMALIZE REDOX RATIO AND OTHER PARAMETERS TO CONTROL 
    data_normalized_to_control = data.copy()
    if dict_key in ['Redox Ratio', 'NAD(P)H a1', 'NAD(P)H a2', 'NAD(P)H Tm']:
        pass
        for tp in np.unique(data['time_hours']):
            pass
            values_name = LIST_OMI_PARAMETERS[dict_key]
            control_data_for_timepoint = data[(data['time_hours'] == tp) & (data['treatment'] == 'media')][values_name]
            control_mean = control_data_for_timepoint.mean()
            
            data_normalized_to_control.loc[data_normalized_to_control['time_hours'] == tp,values_name] -= control_mean
        data = data_normalized_to_control
    ############
    
    ax = sns.boxplot(
                    x=x, 
                    y=y, 
                    hue=hue, 
                    data=data,
                    palette=palette, 
                    hue_order=hue_order,
                    order=order,
                )

    figure_title = f"{dict_key} | threshold: {threshold_percent_toxo} |  NAD(P)H/(NAD(P)H + FAD)" if dict_key == "Redox Ratio" else f"{dict_key} | threshold: {threshold_percent_toxo}"
    plt.title(f"{analysis_type} \n{figure_title} \n {p_values}")
    plt.xlabel("Time Point (hrs)")
    plt.ylabel(dict_key)
    # if dict_key == "Redox Ratio":
    #     plt.ylim((0,1))
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    # Annotations
    annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    # Finally save fig    
    plt.savefig(path_output_figures / f"all_data_{analysis_type}_threshold_{threshold_percent_toxo}_{dict_key}.png", bbox_inches='tight')
    plt.show()

#%% MEDIA VS MEDIA+TOXO - BY EXPERIMENTS

for experiment in np.unique(df_data['experiment']):
    pass

    for dict_key in LIST_OMI_PARAMETERS:
        pass

        palette ={"media": '#1690FF', "media+toxo": '#FC5353'}
        data = df_data[df_data['treatment'].isin(['media','media+toxo'])]
        
        ## select only experiment data
        data = data[data['experiment'] == experiment]
        data = data.astype({"time_hours" : str})
        
        ## filter by high toxo content
        ## threshold df by percent_toxo here
        threshold_percent_toxo = 0.1
        data = data.drop(data[(data['treatment'] == 'media+toxo') & (data['percent_toxo'] < threshold_percent_toxo)].index)
        
        x = "time_hours"
        y = LIST_OMI_PARAMETERS[dict_key]
        hue = "treatment"
        hue_order=['media','media+toxo']
        order = natsorted(list(map(str,np.unique(data['time_hours']))))
        
        pairs = [((x , 'media'), (x,'media+toxo')) for x in np.unique(data["time_hours"]) ]
        
        ############ NORMALIZE REDOX RATIO TO CONTROL 
        data_normalized_to_control = data.copy()
        if dict_key == 'Redox Ratio':
            pass
            for tp in np.unique(data['time_hours']):
                pass
                control_data_for_timepoint = data[(data['time_hours'] == tp) & 
                                                  (data['treatment'] == 'media') & 
                                                  (data['experiment'] == experiment)
                                                  ]['redox_ratio_norm_mean']
                control_mean = control_data_for_timepoint.mean()
                
                data_normalized_to_control.loc[data_normalized_to_control['time_hours'] == tp,'redox_ratio_norm_mean'] -= control_mean
            data = data_normalized_to_control
        ############
        
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

        figure_title = f"{dict_key} | Dataset: {experiment} | threshold: {threshold_percent_toxo} | NAD(P)H/(NAD(P)H + FAD)" if dict_key == "Redox ratio" else f"{dict_key} | Dataset: {experiment} | threshold: {threshold_percent_toxo}"
        plt.title(f"{analysis_type} \n{figure_title} \n{p_values}")
        plt.xlabel("Time Point (hrs)")
        plt.ylabel(dict_key)
        plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
        plt.tight_layout()
    
        # Annotations
        # annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
        # annotator.configure(test='t-test_ind', text_format='star', loc='inside')
        # annotator.apply_and_annotate()
    
        # Finally save fig    
        plt.savefig(path_output_figures / f"{experiment}_{analysis_type}_{threshold_percent_toxo}_{dict_key}.png", bbox_inches='tight')
        plt.show()
        # plt.close()



#%% TOXO CONDITION --> HIGH TOXO CELLS VS EXTERNAL TOXO
analysis_type = 'high_toxo_cells_vs_outside_toxo'
path_output_figures = path_project / "figures" / analysis_type / "seaborn"



