import seaborn as sns 
import cmcrameri.cm as cmc


from pathlib import Path
import pandas as pd
#%%

path_project = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio")
path_all_features = list(path_project.glob("*props.csv"))[0] 

#%%

df_data = pd.read_csv(path_all_features)

df_plot_data = df_data[["redox_ratio_mean","nadh_a1_mean"]] #[:1000] 

clustermap = sns.clustermap(df_plot_data, 
                            cmap=cmc.batlow,
                            # standard_scale = 0,
                            z_score=1,
                            )


#%%

import seaborn as sns; sns.set_theme(color_codes=True)
iris = sns.load_dataset("iris")
species = iris.pop("species")
g = sns.clustermap(iris)

# plt.plot(g)
# plt.show()