from pathlib import Path
import pandas as pd
import tifffile
import matplotlib.pylab as plt

from skimage import filters

from flim_tools.image_processing import kmeans_threshold

from sklearn.preprocessing import RobustScaler

path_dicts = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\dictionaries")

list_datasets = list(path_dicts.glob("*.csv"))

#%%
for path_dict_dataset in list_datasets:
    pass

    df_dataset = pd.read_csv(path_dict_dataset)

    # get images with toxo
    
    df_dataset_toxo = df_dataset[df_dataset["toxo_photons"].apply(lambda x : isinstance(x, str))]


    for index, row_data in df_dataset_toxo.iterrows():
        pass
        im = tifffile.imread(row_data['toxo_photons'])
        
        plt.title(Path(row_data['toxo_photons']).name)
        plt.imshow(im,)
        plt.show()
        
        # plt.hist(im, bins=255, histtype='step')
        # plt.show()
        # im_scaled = RobustScaler().fit_transform(im)
        # plt.hist(im_scaled, bins=255, histtype='step')
        # plt.show()
        # plt.imshow(im_scaled)
        # plt.show()
        
        plt.title(Path(row_data['toxo_photons']).name)
        im_diff_gauss = filters.difference_of_gaussians(im, low_sigma=1)
        # plt.imshow(im_diff_gauss) #, vmax=30
        # plt.show()
        
        plt.title(Path(row_data['toxo_photons']).name)
        im_media = filters.median(im_diff_gauss)
        # plt.imshow(im_media)
        # plt.show()
        
                
        plt.title(Path(row_data['toxo_photons']).name)
        im_kmeans = kmeans_threshold(im_media, k=2, n_brightest_clusters=1)
        plt.imshow(im_kmeans)
        plt.show()
        

        

