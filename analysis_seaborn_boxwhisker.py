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

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from natsort import natsort_keygen

from tqdm import tqdm
# import proplot as pplt

output_format = 'png'
# output_format = 'svg'
#%% Load data
path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_all_features = list(path_project.glob(f"*all_props_cells.csv"))[0] 
df_data = pd.read_csv(path_all_features)

df_data = df_data[(df_data['time_hours'] != 3) &
                  (df_data['time_hours'] != 72) 
                  # (df_data['time_hours'] != 48) & 
                  # (df_data['treatment'].isin(['media', 'media+toxo']) )
                      ]

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

#%% TOXO CONDITION --> HISTOGRAMS OF PERCENT LOW VS HIGH TOXO 

analysis_type = 'toxo_inside_cells_high_vs_low'
path_output_figures = path_project / "figures" / analysis_type / "seaborn"


font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 10}
mpl.rc('font', **font)

bool_boxwhisker = False # false plots lineplots
plot_type = "boxwhisker" if bool_boxwhisker else "lineplot"

for experiment in np.unique(df_data['experiment']):
    pass

    ##%% PLOT PERCENT TOXO CAPTURED -- ALL IN ONE FIGURE
    for timepoint in np.unique(df_data['time_hours']):
        pass
        df_data_subset = df_data[(df_data['time_hours']==timepoint) & 
                                 (df_data['experiment']==experiment) &
                                 (df_data['treatment']=='media+toxo') ## only for media+toxo
                                 ]
        
        if bool_boxwhisker:
            plt_data = plt.hist(df_data_subset['percent_toxo'] * 100, range=(0,100), histtype='step', bins=20, label=timepoint)
        else: # line plots 
            plt_data = np.histogram(df_data_subset['percent_toxo'] * 100,  bins=20, range=(0,100))
            plt.plot(np.linspace(5, 100 , 20 ), plt_data[0], label=timepoint)
            
    plt.title(f"Percent toxo captured per timepoint \nexperiment: {experiment}")
    plt.legend()
    plt.grid()
    plt.locator_params(nbins=10)
    plt.xlim(0,100)
    plt.xlabel("Percent toxoplasma in cell")
    plt.ylabel("Cell Count")
    
    
    ax = plt.gca()
    ax.axhline(linewidth=3, color="k")        # inc. width of x-axis and color it green
    ax.axvline(linewidth=3, color="k")  
    # ax.set_ylim(0,100)
    
    plt.savefig(path_output_figures / f"percent_toxo_{experiment}_{analysis_type}_{plot_type}.{output_format}", bbox_inches='tight')
    plt.show()
    
    ##%% PLOT PERCENT CAPTURED -- ONE FIGURE PER TIMEPOINT per experiment
    for timepoint in np.unique(df_data['time_hours']):
        pass
        df_data_subset = df_data[(df_data['time_hours']==timepoint) & 
                                 (df_data['experiment']==experiment) &
                                 (df_data['treatment']=='media+toxo')
                                 ]
        #plot
        fig, ax = plt.subplots(1,1, figsize=(10,11))
        fig.suptitle(f"percent toxo in cells \n experiment: {experiment} | time point = {timepoint} \nmean: {np.mean(df_data_subset['percent_toxo']):.3f} | total cells={len(df_data_subset)}")
        
        ## set font and weights
        ax.axhline(linewidth=4, color="k")        # inc. width of x-axis and color it green
        ax.axvline(linewidth=5, color="k")  
        # ax.set_ylim(0,40)

        #boxwhisker
        # ax.hist(df_data_subset['percent_toxo'] * 100, histtype='step', bins=20)
        if bool_boxwhisker:
           plt_data = plt.hist(df_data_subset['percent_toxo'] * 100, range=(0,100), histtype='step', bins=20, label=timepoint)
        
        else: # line plots 
           plt_data = np.histogram(df_data_subset['percent_toxo'] * 100,  bins=20, range=(0,100))
           plt.plot(np.linspace(5, 100 , 20 ), plt_data[0], label=timepoint)
        
        plt.xlabel("Percent toxoplasma in cell")
        plt.ylabel("Cell Count")
        plt.locator_params(nbins=10)
        plt.xlim(0,100)
        plt.grid()
        plt.savefig(path_output_figures / f"percent_toxo_{experiment}_timepoint_{timepoint}_{analysis_type}_{plot_type}.{output_format}", bbox_inches='tight')
        plt.show()

    
##%% TOXO CONDITION --> LOW VS HIGH TOXO  -- TOXO CONTENT,  ALL EXPERIMENTS BY TIMEPOINT

for timepoint in np.unique(df_data['time_hours']):
    pass
    df_data_subset = df_data[(df_data['time_hours']==timepoint) & 
                             (df_data['treatment']=='media+toxo') ## only for media+toxo
                             ]
    # boxshiker
    # ax = plt.hist(df_data_subset['percent_toxo'] * 100, histtype='step', bins=20, label=timepoint)
     
     
    # lineplot
    if bool_boxwhisker:
       plt_data = plt.hist(df_data_subset['percent_toxo'] * 100, range=(0,100), histtype='step', bins=20, label=timepoint)
    else: # line plots 
       plt_data = np.histogram(df_data_subset['percent_toxo'] * 100,  bins=20, range=(0,100))
       plt.plot(np.linspace(5, 100 , 20 ), plt_data[0], label=timepoint)

plt.title(f"Percent Toxo Captured Per Timepoint \nAll Experiments")
plt.xlabel("Percent toxoplasma in cell")
plt.ylabel("Cell Count")
plt.grid()
plt.legend()
plt.locator_params(nbins=10)
plt.xlim(0,100)
# plt.ylim(0,150)
ax = plt.gca()
ax.axhline(linewidth=4, color="k")
ax.axvline(linewidth=5, color="k")
plt.savefig(path_output_figures / f"percent_toxo_all_experiments_{analysis_type}_{plot_type}.{output_format}", bbox_inches='tight')
plt.show()
#%%
##########%% MEDIA+TOXO - LOW VS HIGH TOXO --> ALL EXPERIMENTS

p_values = "ns: p <= 1 | "\
           "*: .01 < p <= .05  | "\
           "**: .001 < p <= .01  | "\
           "***: .0001 < p <= .001  | "\
           "****: p <= .0001"

mpl.rcParams['figure.figsize'] = 20,15

font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 35}
mpl.rc('font', **font)


bool_boxwhisker = True # false plots lineplots
plot_type = "boxwhisker" if bool_boxwhisker else "lineplot"

for dict_key in LIST_OMI_PARAMETERS:
    pass
    palette ={"low_toxo": '#FCA853', "high_toxo": '#FC5353'} # #1690FF

    data = df_data[df_data['treatment'].isin(['media+toxo'])]
    data = data.astype({"time_hours" : str})
    
    ## threshold df by percent_toxo here
    threshold_percent_toxo = 0.05
    
    # data = data.drop(data[(data['treatment']=='media+toxo')&(data['percent_toxo'] < threshold_percent_toxo)].index)
    
    # add col for low and high toxo content
    data['toxo_class'] = np.where(data['percent_toxo'] < threshold_percent_toxo, 'low_toxo', 'high_toxo')
    
    x = "time_hours"
    y = LIST_OMI_PARAMETERS[dict_key]
    hue = "toxo_class"
    hue_order=['low_toxo','high_toxo']
    order = list(map(str,np.unique(df_data['time_hours'])))
    pairs=[((str(x) , 'low_toxo'), (str(x),'high_toxo')) for x in np.unique(df_data["time_hours"]) ]
        
    if bool_boxwhisker:
        ax = sns.boxplot(
                        x=x, 
                        y=y, 
                        hue=hue, 
                        data=data,
                        palette=palette, 
                        hue_order=hue_order,
                        order=order,
                    )
    else:
        ## plot line plots
        
        ax_line = sns.lineplot(
            x=x, 
            y=y, 
            hue=hue, 
            data=data.sort_values(by='time_hours', key=natsort_keygen()),
            palette=palette, 
            hue_order=hue_order,
            estimator='mean'
            # order=order,
            )
        
    figure_title = f"{dict_key} | NAD(P)H/(NAD(P)H + FAD) \npercent threshold toxo content: {threshold_percent_toxo} | cell count low:high = {sum(data['toxo_class']=='low_toxo')}:{sum(data['toxo_class']=='high_toxo')}" \
        if dict_key == "Redox Ratio" else f"{dict_key} \npercent threshold toxo content: {threshold_percent_toxo} | cell count low:high = {sum(data['toxo_class']=='low_toxo')}:{sum(data['toxo_class']=='high_toxo')}"
    
    # plt.title(f"{analysis_type} \n{figure_title} \n {p_values}")
    plt.xlabel("Time Point (hrs)")
    plt.ylabel(dict_key)
    # if dict_key == "Redox Ratio":
    #     plt.ylim((0,1))
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, title="Conditions")
    plt.tight_layout()
    
    ###########
    # x and y axis weights
    ax = plt.gca()
    l,r,b,t = list(ax.spines.values())
    plt.setp([l,b], linewidth=3)
    
    # The ticks
    ax.xaxis.set_tick_params(width=5)
    ax.yaxis.set_tick_params(width=5)
        
    if bool_boxwhisker:
        # Annotations
        annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside')
        annotator.apply_and_annotate()

    # Finally save fig    
    plt.savefig(path_output_figures / f"threshold_{threshold_percent_toxo}_all_data_{analysis_type}_{dict_key}_{plot_type}.{output_format}", bbox_inches='tight')
    plt.show()

#%% CONTROL VS HIGH TOXO CELLS vs KISS AND SPIT

font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 35}
mpl.rc('font', **font)

analysis_type = 'whole_cell_vs_high_toxo'
path_output_figures = path_project / "figures" / analysis_type / "seaborn"

##########%% MEDIA VS MEDIA+TOXO - ALL EXPERIMENTS

# all conditions if not just media and media+toxo
# data_to_plot = "all"
# data_to_plot = "media_vs_high_toxo"
data_to_plot = "kiss_and_spit"

bool_boxwhisker_plots = True
bool_normalize_redox_ratio = True

plot_type = "boxwhisker" if bool_boxwhisker_plots else "lineplot"

p_values = "ns: p <= 1 | "\
           "*: .01 < p <= .05  | "\
           "**: .001 < p <= .01  | "\
           "***: .0001 < p <= .001  | "\
           "****: p <= .0001"

mpl.rcParams['figure.figsize'] =25,15

for dict_key in LIST_OMI_PARAMETERS:
    pass
    # if dict_key != "NAD(P)H Intensity":
    #     continue
    
    # GET SUBSET OF DTASET
    if data_to_plot == "all":
        palette ={"media": '#1690FF', "media+toxo": '#FC5353', 'media+inhibitor' : '#a83dc2', 'media+inhibitor+toxo' : '#1ad2e5'}
        data = df_data[df_data['treatment'].isin(['media','media+toxo','media+inhibitor','media+inhibitor+toxo'])]
        
    elif data_to_plot == "kiss_and_spit":
        palette ={'media+inhibitor' : '#a83dc2', 'media+inhibitor+toxo' : '#1ad2e5'}
        data = df_data[df_data['treatment'].isin(['media+inhibitor','media+inhibitor+toxo'])]
    
    elif data_to_plot == "media_vs_high_toxo":
        palette ={"media": '#1690FF', "media+toxo": '#FC5353'}
        data = df_data[df_data['treatment'].isin(['media','media+toxo'])]
        
    data = data.astype({"time_hours" : str})
    
    ##### COMMENT IN/OUT NUM CONDITIONS
    ## threshold df by percent_toxo here -- TWO CONDITIONS
    # threshold_percent_toxo = 0.05
    # data = data.drop(data[(data['treatment']=='media+toxo')&(data['percent_toxo'] < threshold_percent_toxo)].index)
    
    # ## threshold df by percent_toxo here ALL CONDITIONS
    threshold_percent_toxo = 0.05
    data = data.drop(data[
                    (data['treatment'].isin(['media+toxo','media+inhibitor+toxo'])) &
                    (data['percent_toxo'] < threshold_percent_toxo)
                    ].index)
    
    #### export mean and stdev for dataset
    
    for timepoint in np.unique(data['time_hours']):
        pass
        print(f"{'-'*20} timepoint: {timepoint} {'-'*20}")
        print("Mean")
        df_sub = data[data['time_hours'] == timepoint]
        df_groupby_mean = df_sub[['treatment','redox_ratio_norm_mean']].groupby('treatment').mean()
        print(df_groupby_mean)
        print("Standard Deviation")
        df_groupby_std = df_sub[['treatment','redox_ratio_norm_mean']].groupby('treatment').std()
        print(df_groupby_std)
                          
    #####
    x = "time_hours"
    y = LIST_OMI_PARAMETERS[dict_key]
    hue = "treatment"
    
    
    if data_to_plot == "all":
        hue_order=['media','media+toxo', 'media+inhibitor', 'media+inhibitor+toxo'] # ALL CONDITIONS
        
    elif data_to_plot == "media_vs_high_toxo":
        hue_order=['media','media+toxo'] # TWO CONDITIONS 

    elif data_to_plot == "kiss_and_spit":
        hue_order=['media+inhibitor', 'media+inhibitor+toxo'] # kiss and spit conditions

    order = list(map(str,np.unique(df_data['time_hours'])))
    
    
    # FOR STATANNOTATIONS
    pairs_control_vs_high_toxo=[((str(x) , 'media'), (str(x),'media+toxo')) for x in np.unique(df_data["time_hours"]) ]
    pairs_inhibitor=[((str(x) , 'media+inhibitor'), (str(x),'media+inhibitor+toxo')) for x in np.unique(df_data["time_hours"]) ]

    if data_to_plot == "all":
        pairs = pairs_control_vs_high_toxo + pairs_inhibitor
    elif data_to_plot == "media_vs_high_toxo":
        pairs = pairs_control_vs_high_toxo
    elif data_to_plot == "kiss_and_spit":
        pairs = pairs_inhibitor
    
    
    ############ NORMALIZE REDOX RATIO TO CONTROL 
    if bool_normalize_redox_ratio:
        data_normalized_to_control = data.copy()
        if dict_key in ['Redox Ratio']: #
            pass
            for tp in np.unique(data['time_hours']):
                pass
                values_name = LIST_OMI_PARAMETERS[dict_key]
                
                ## normalize data to media as control
                if data_to_plot in ["all", "media_vs_high_toxo"]:
                    control_data_for_timepoint = data[(data['time_hours'] == tp) & 
                                                      (data['treatment'] == 'media')][values_name]
                
                # normalize kiss and spit data to media+inhibitor
                elif data_to_plot == "kiss_and_spit":
                    control_data_for_timepoint = data[(data['time_hours'] == tp) & 
                                                      (data['treatment'] == 'media+inhibitor')][values_name]
                
                control_mean = control_data_for_timepoint.mean()
                
                data_normalized_to_control.loc[data_normalized_to_control['time_hours'] == tp, values_name] /= control_mean # -= control_mean
            data = data_normalized_to_control
    
    ############ plot boxplots
    if bool_boxwhisker_plots:
        ax = sns.boxplot(
                        x=x, 
                        y=y, 
                        hue=hue, 
                        data=data,
                        palette=palette, 
                        hue_order=hue_order,
                        order=order,
                    )
    else:
        ## plot line plots
        
        ax_line = sns.lineplot(
            x=x, 
            y=y, 
            hue=hue, 
            data=data.sort_values(by='time_hours', key=natsort_keygen()),
            palette=palette, 
            hue_order=hue_order,
            estimator='mean'
            # order=order,
            )
    
    #### rest of figure
    # figure_title = f"{dict_key} | threshold: {threshold_percent_toxo} |  NAD(P)H/(NAD(P)H + FAD)" if dict_key == "Redox Ratio" else f"{dict_key} | threshold: {threshold_percent_toxo}"
    # plt.title(f"{analysis_type} \n{figure_title} \n {p_values}")
    plt.xlabel("Time Point (hrs)")
    plt.ylabel(dict_key) if dict_key != 'Redox Ratio' else plt.ylabel("Normalized Redox Ratio")
    # if dict_key == "Redox Ratio":
    #     plt.ylim((0,1))
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    #####
    # x and y axis weights
    ax = plt.gca()
    l,r,b,t = list(ax.spines.values())
    plt.setp([l,b], linewidth=3)
    
    # The ticks
    ax.xaxis.set_tick_params(width=5)
    ax.yaxis.set_tick_params(width=5)

    # Annotations
    if bool_boxwhisker_plots:
        annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside')
        annotator.apply_and_annotate()

    # Finally save fig
    if data_to_plot == "all":
        plt.savefig(path_output_figures / "all_conditions" / f"all_data_{analysis_type}_threshold_{threshold_percent_toxo}_{dict_key}_{plot_type}_kiss_and_spit.{output_format}", bbox_inches='tight')   
    elif data_to_plot == "media_vs_high_toxo":
        plt.savefig(path_output_figures / "media_vs_high_toxo" / f"all_data_{analysis_type}_threshold_{threshold_percent_toxo}_{dict_key}_{plot_type}.{output_format}", bbox_inches='tight')
    elif data_to_plot == "kiss_and_spit":
        plt.savefig(path_output_figures / "kiss_and_spit" / f"all_data_{analysis_type}_threshold_{threshold_percent_toxo}_{dict_key}_{plot_type}.{output_format}", bbox_inches='tight')

    plt.show()

#%% MEDIA VS MEDIA+TOXO - BY EXPERIMENTS

analysis_type = 'whole_cell_vs_high_toxo'
path_output_figures = path_project / "figures" / analysis_type / "seaborn"

mpl.rcParams['figure.figsize'] =20,15


bool_boxwhisker_plots = False
plot_type = "boxwhisker" if bool_boxwhisker_plots else "lineplot"

for experiment in np.unique(df_data['experiment']):
    pass

    for dict_key in tqdm(LIST_OMI_PARAMETERS):
        pass

        palette ={"media": '#1690FF', "media+toxo": '#FC5353'}
        data = df_data[df_data['treatment'].isin(['media','media+toxo'])]
        
        ## select only experiment data
        data = data[data['experiment'] == experiment]
        data = data.astype({"time_hours" : str})
        
        ## filter by high toxo content
        ## threshold df by percent_toxo here
        threshold_percent_toxo = 0.05
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
                
                data_normalized_to_control.loc[data_normalized_to_control['time_hours'] == tp,'redox_ratio_norm_mean'] /= control_mean
            data = data_normalized_to_control
        ############
        
        
        if bool_boxwhisker_plots:
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
        else:
            ax_line = sns.lineplot(
                x=x, 
                y=y, 
                hue=hue, 
                data=data.sort_values(by='time_hours', key=natsort_keygen()),
                palette=palette, 
                hue_order=hue_order,
                estimator='mean'
                # order=order,
                )
        # figure_title = f"{dict_key} | Dataset: {experiment} | threshold: {threshold_percent_toxo} | NAD(P)H/(NAD(P)H + FAD)" if dict_key == "Redox ratio" else f"{dict_key} | Dataset: {experiment} | threshold: {threshold_percent_toxo}"
        # plt.title(f"{analysis_type} \n{figure_title} \n{p_values}")
        plt.xlabel("Time Point (hrs)")
        plt.ylabel(dict_key) if dict_key != 'Redox Ratio' else plt.ylabel("Normalized Redox Ratio")
        plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        
        # x and y axis weights
        ax = plt.gca()
        l,r,b,t = list(ax.spines.values())
        plt.setp([l,b], linewidth=3)
        
        # The ticks

        ax.xaxis.set_tick_params(width=5)
        ax.yaxis.set_tick_params(width=5)
    
        # Annotations
        # annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
        # annotator.configure(test='t-test_ind', text_format='star', loc='inside')
        # annotator.apply_and_annotate()
    
        # Finally save fig    
        plt.savefig(path_output_figures / f"{experiment}_{analysis_type}_{threshold_percent_toxo}_{dict_key}_{plot_type}.{output_format}", bbox_inches='tight')
        plt.show()
        # plt.close()



#%% TOXO CONDITION --> HIGH TOXO CELLS VS EXTERNAL TOXO
# analysis_type = 'high_toxo_cells_vs_outside_toxo'
# path_output_figures = path_project / "figures" / analysis_type / "seaborn"



