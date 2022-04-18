from pathlib import Path
import pandas as pd
import tifffile
import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300

import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread

from helper import load_image

from sklearn.cluster import KMeans


from skimage.filters import median
from helper import visualize_dictionary


path_dict_dataset = r"Z:\0-Projects and Experiments\GG - toxo_omi_redox_ratio\dictionaries\04-19-2019-keys.csv"

df_dataset = pd.read_csv(path_dict_dataset)

#%%
for index_name, row_data in df_dataset.iterrows():
    pass
    im = tifffile.imread(row_data.mask_cell)
    # plt.imshow(im)
    # np.unique(im)

#%%



# for row_data in dict_dataset:
    
    visualize_dictionary(index_name, row_data)

    b = load_image(row_data['mask_cell'])
    plt.imshow(b)
    m1 = median(b)
    plt.imshow(m1)
    np.unique(m1)
    m2 = median(m1)
    plt.imshow(m2)
    plt.show()
    
    # https://stackoverflow.com/questions/48222977/python-converting-an-image-to-use-less-colors
    from sklearn.cluster import KMeans
    
    # arr = original.reshape((-1, 3))
    arr = m2
    # n_colors = 12
    # kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    # labels = kmeans.labels_
    # centers = kmeans.cluster_centers_
    # plt.hist(m2, bins=255)
    # less_colors = centers[labels].reshape(arr.shape).astype('uint8')
    
    counts, bin_edges = np.histogram(m2,bins=255)
    
    threshold = 400 # pixels
    
    binary_mask 
    
    plt.imshow(less_colors)
#%% Cellpose segmentation

    
    # model_type='cyto' or 'nuclei' or 'cyto2'
    model = models.Cellpose(model_type='cyto2')
    
    # list of files
    # PUT PATH TO YOUR FILES HERE!
    # files = ['/media/carsen/DATA1/TIFFS/onechan.tif']
    
    # imgs = [imread(f) for f in files]
    # nimg = len(imgs)
    
    imgs = [m2]
    nimg = len(imgs)
    
    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
    
    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended)
    # diameter can be a list or a single number for all images
    
    masks, flows, styles, diams = model.eval(imgs, 
                                             diameter=30, 
                                             flow_threshold=0.4,
                                             cellprob_threshold=0.0,
                                             channels=channels)
    
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(masks[0])
    
    #%%
    
# from scipy import ndimage as ndi

# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# distance = ndi.distance_transform_edt(m2)
# coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=m2)
# mask = np.zeros(distance.shape, dtype=bool)
# mask[tuple(coords.T)] = True
# markers, _ = ndi.label(mask)
# labels = watershed(-distance, markers, mask=m2)

#%%


from PIL import Image, ImageFilter
  
  
# Opening the image (R prefixed to string
# in order to deal with '\' in paths)
# image = Image.open(r"Sample.png")
  
# Converting the image to grayscale, as edge detection 
# requires input image to be of mode = Grayscale (L)
# image = image.convert("L")
  
# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
image = Image.fromarray(m2)
image = image.filter(ImageFilter.FIND_EDGES)
  
plt.imshow(np.array(image)>0)
# Saving the Image Under the name Edge_Sample.png
image.save(r"Edge_Sample.png")


#%%

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def simple_edge_detection(image): 
   edges_detected = cv2.Canny(np.array(image), 100, 200) 
   images = [image , edges_detected]
   
simple_edge_detection(image)
