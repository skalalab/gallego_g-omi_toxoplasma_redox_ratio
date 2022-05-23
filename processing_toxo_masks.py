from pathlib import Path
import pandas as pd
import tifffile
import matplotlib.pylab as plt
import numpy as np
from skimage import filters

from flim_tools.image_processing import kmeans_threshold
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import binary_fill_holes

path_dicts = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\dictionaries")
list_datasets = list(path_dicts.glob("*.csv"))
from skimage.morphology import white_tophat, disk

#%%
for path_dict_dataset in list_datasets[:1]:
    pass


    debug = False
    
    df_dataset = pd.read_csv(path_dict_dataset)

    # get images with toxo
    df_dataset_toxo = df_dataset[df_dataset["toxo_photons"].apply(lambda x : isinstance(x, str))]

    ## create dir for toxo masks
    path_first_image = str(Path(df_dataset_toxo.nadh_photons[0])) # determine dataset being processed
    
    if "3-18-2019" in path_first_image:
        path_dataset = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\3-18-2019\03182019_Katie")
        path_output = path_dataset / "generated_toxo_masks"
    elif "4-6-2019" in path_first_image:
            path_dataset = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\4-6-2019\040619_Katie_SPC")
            path_output = path_dataset / "generated_toxo_masks"


    for index, row_data in list(df_dataset_toxo.iterrows())[-10:]:
        pass
        im = tifffile.imread(row_data['toxo_photons']) # 
        im_clipped = np.clip(im, 0, np.percentile(im,99))
        im_media = filters.median(im_clipped)
        im_diff_gauss = filters.difference_of_gaussians(im_media, 
                                                        low_sigma=0.5, 
                                                        high_sigma=5,
                                                        mode="nearest")
        
        # toxo gets brighter as the timecourse goes on
        # experiment different k values for different 
        # timepoints
        # if row_data['time_hours'] == 1:
        #     k = 6 ; n_brightest_clusters=2
        # elif row_data['time_hours'] == 3:
        #     k = 6 ; n_brightest_clusters=2
        # elif row_data['time_hours'] == 6:
        #     k = 6 ; n_brightest_clusters=2
        # elif row_data['time_hours'] == 9:
        #     k = 6 ; n_brightest_clusters=2
        # elif row_data['time_hours'] == 12:
        #     k = 6 ; n_brightest_clusters=2
        # elif row_data['time_hours'] == 24:
        #     k = 6 ; n_brightest_clusters=2
        # elif row_data['time_hours'] == 48:
        #     k = 6 ; n_brightest_clusters=2
        # elif row_data['time_hours'] == 72:
        #     k = 6 ; n_brightest_clusters=2
            
        k = 6 ; n_brightest_clusters=2
            
        im_kmeans = kmeans_threshold(im_diff_gauss, 
                                     k=k, 
                                     n_brightest_clusters=n_brightest_clusters)
        # im_kmeans = np.clip(im, np.percentile(im_diff_gauss,60), np.max(im_diff_gauss))
        # ax[4].set_title("percentile")
        # ax[4].imshow(im_kmeans > 0)
        # ax[4].set_axis_off()    
        
        im_fill_holes = binary_fill_holes(im_kmeans, disk(1))

        # plt.savefig(path_output / f"{Path(row_data['toxo_photons']).stem}_grid.png")
        # plt.close()
        plt.show()
        
        if debug:
            fig, ax = plt.subplots(1,6, figsize=(14,3))
            
            fig.suptitle(f"{Path(row_data['toxo_photons']).name} \n hours: {row_data['time_hours']}")
            
            ax[0].set_title("original")
            ax[0].imshow(im, ) # vmax=np.percentile(im,99)
            ax[0].set_axis_off()
            
            ax[1].set_title("clip at 99th percentile")
            ax[1].imshow(im_clipped)
            ax[1].set_axis_off()
            
            ax[2].set_title("median filter")
            ax[2].imshow(im_media)
            ax[2].set_axis_off()
            
            ax[3].set_title("diff of gauss")
            ax[3].imshow(im_diff_gauss) #, vmax=30
            ax[3].set_axis_off()
            
            ax[4].set_title(f"k-means {k} keep {n_brightest_clusters}")
            ax[4].imshow(im_kmeans)
            ax[4].set_axis_off()
            
            ax[5].set_title("binary fill holes")
            ax[5].imshow(im_fill_holes)
            ax[5].set_axis_off()
            
    # tifffile.imwrite(path_output / f"{Path(row_data['toxo_photons']).stem}_mask_toxo.tiff", im_fill_holes)
        
