# METABOLISM OF _TOXOPLASMA GONDII_ ON HUMAN FORESKIN FIBROBLAST CELLS

Analysis code for this article is found here. It contains all files that were used to load the data, do the analysis and generate the figures. 

<hr>
### Dependencies and System Requirements

cell-analysis-tools == 0.0.5
matplotlib == 3.4.3
tqdm == 4.62.3
pandas == 1.3.4
numpy == 1.20.3
natsort == 8.1.0
skimage == 0.18.3
tifffile ==2021.7.2

<hr>
### How to run the code

To run the code it is suggested that you create a conda environment and install the necessary dependencies there. 

#### Description of Files

* **toxo_3-18-2019.py** and  **toxo_3-18-2019.py** find paths to all the images and creates a set of matching NADH(P)H and FAD image paths, finally exporting a CSV to be used on later analysis.

* **processing_toxo_masks.py** this is the file used to generate the toxoplasmam masks from the mCherry fluorescence images

* **napari_mask_editor.py** this is a helper script that loads intensity images alongside the masks in Napari so scientists can review the masks and edit them as needed. This script also takes care of the saving of the images as the viewer is closed.

* **analysis_main.py** uses the csv outputs of the previous files to load all the sets of images, it then computes feature extraction using the regionprops_omi function within cell-analysis-tools library. The results are computed and saved at the image level into a folder called _features_ and then aggregated into a single csv file for all the images/experiments. This code also excludes images that have a rois with high chi squared values (poor fits).

* **helper.py**  holds a couple functions used for loading images and visualizing the sets of images gathered in the scripts: **toxo_3-18-2019.py** and  **toxo_3-18-2019.py**

* **analysis_seaborn_boxwhisker.py** main script used to generate all the figures.

### Run order

To recreate the findings you will need a copy of the data, you can then run the two scripts that generate csv files to all the images **toxo_3-18-2019.py** and  **toxo_3-18-2019.py**, from there you can run the **analysis_main.py** script to perform feature extraction and save results in a csv file. Lastly run the cells in **analysis_seaborn_boxwhisker.py** to generate plots.

<hr>
### Special notes/instructions

* Images and data are available upon request 
* Paths inside scripts will need to be updated as needed to point to the correct directories.

<hr>
### How to Cite this Paper

TBD
