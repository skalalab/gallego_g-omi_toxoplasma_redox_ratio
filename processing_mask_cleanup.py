from pathlib import Path
import pandas as pd
import tifffile
import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300

from tqdm import tqdm
from helper import load_image

import scipy.ndimage as ndi

from skimage.filters import median
from helper import visualize_dictionary


path_dict_dataset = r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\dictionaries\04-19-2019-keys.csv"

df_dataset = pd.read_csv(path_dict_dataset)
# path_output = Path(r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\4-6-2019\masks_whole_cell")
#%%
for index_name, row_data in tqdm(list(df_dataset.iterrows())[:]):
    pass
    im = tifffile.imread(row_data.mask_cell)
    # plt.imshow(im)
    # np.unique(im)

    # for row_data in dict_dataset:
    
    # visualize_dictionary(index_name, row_data)

    mask = load_image(row_data['mask_cell'])
    # plt.imshow(b)
    # m1 = median(b)
    # plt.imshow(m1)
    # np.unique(m1)
    # m2 = median(m1)
    # plt.imshow(m2)
    # plt.show()
    
    from skimage import filters
    from skimage.feature import canny
    from skimage.morphology import (disk, 
                                    dilation, 
                                    skeletonize, 
                                    binary_opening,
                                    remove_small_objects)
    
   
    binary_mask = np.array(mask > 0, dtype=np.uint8)
    # plt.imshow(binary_mask)

  
    ### isolate good binary mask
    f1 = binary_opening(binary_mask, disk(1))
    # plt.imshow(f1)
    f2 = remove_small_objects(f1)
    # plt.imshow(f2)
    
    ## get good edges to separate boundaries
    f3 = canny(mask)
    # plt.imshow(f3)
    f4 = dilation(f3, disk(1) )
    # plt.imshow(f4)
    f5 = skeletonize(f4)
    # plt.imshow(f5)
    
    # combine
    f6 = f2 * np.invert(f5)
    # plt.imshow(f6)
    f7 = remove_small_objects(f6)
    # plt.imshow(f7)
    
    new_mask, _ = ndi.label(np.array(f7, dtype=np.uint16))
    # plt.imshow(new_mask)
    # plt.imshow(filters.sobel(mask))
    

    
    fig, ax = plt.subplots(1,4, figsize=(10,5))
    
    filename_mask = Path(row_data['mask_cell']).stem
    fig.suptitle(filename_mask)
    
    ax[0].imshow(load_image(row_data['nadh_photons']))
    ax[0].set_title(f"nadh")
    ax[0].set_axis_off()
    
    ax[1].imshow(mask)
    ax[1].set_title(f"original")
    ax[1].set_axis_off()
    
    fig.suptitle(Path(row_data['mask_cell']).stem)
    ax[2].imshow(mask > 0)
    ax[2].set_title(f"original binary")
    ax[2].set_axis_off()
    
    ax[3].imshow(new_mask)
    ax[3].set_title("new mask")
    ax[3].set_axis_off()
    plt.savefig(path_output / f"{filename_mask}_grid.png")
    plt.close()
    # plt.show()
    
    # save images 
    # tifffile.imwrite(path_output / f"{filename_mask}.tiff", new_mask)
    
    #%%
    
# from skimage import feature
# img = mask
# edges_canny = feature.canny(img) # Canny
# plt.imshow(edges_canny)
# edges_sobel = filters.sobel(img) # Sobel
# plt.imshow(edges_sobel)
# edges_laplace = filters.laplace(img) # Laplacian
# plt.imshow(edges_laplace)
# edges_scharr = filters.scharr(img) # Scharr
# plt.imshow(edges_scharr)
# edges_prewitt = filters.prewitt(img) # Prewitt
# plt.imshow(edges_prewitt)
# edges_roberts = filters.roberts(img) # Roberts
# plt.imshow(edges_roberts)

