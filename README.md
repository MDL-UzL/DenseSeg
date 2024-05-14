# DenseSeg: Joint Learning for Semantic Segmentation and Landmark Detection Using Dense Image-to-Shape Representation

## Environment
Please use the provided yaml (environment.yml) file to create the environment. 
```bash
conda env create -f environment.yml
```

## Data
All the data should be in the 'dataset/data' folder.
### JSRT Datset
Please refer to ngaggion's [repository](https://github.com/ngaggion/Chest-xray-landmark-dataset/blob/main/Preprocess-JSRT.ipynb) to download and preprocess the JSRT dataset as well for his provided landmarks.
We further reduce the resolution to 256x256 pixel and organize the images and landmarks as two pytorch tensors in a python dictionary `JSRT_img0_lms.pth`.
Please notice, we store the already z-normalized versions of the images.
To create the segmentation mask from the landmarks, we utilize `skimage.draw.polygon2mask` function. Under `dataset/generate_jsrt_seg_lbl.py` we provide the code to generate the segmentation masks from the landmarks. Please notice that this script may need minor adjustments to the return parameters of the current implementation of `JSRTDataset` due to the missing segmentation masks.

### GrazPedWriDataset
Please download the dataset using the provided link in the original [paper](https://www.nature.com/articles/s41597-022-01328-z) and preprocess it with their provided notebooks to obtain the 8-bit images. After this, please use our provided preprocessing script `dataset/copy_and_process_graz_imgs` to create the homogeneous dataset (all images flipped to left).
We then use `dataset/create_grazer_h5.py` to create the h5 file with resized images, landmarks, distance maps, and segmentation masks, which is then used by our dataset implementation.

For this dataset, we provide the segmentation masks (`dataset/data/graz/raw_segmentations_no_cast.h5`) for 17 bones (not including images with cast) as well corresponding landmarks for these images.
The segmentation masks are obtained by a UNet which was trained utilizing SAM as postprocessing technique to refine initial predictions.
The landmarks were created by training an unbiased point cloud registration network that aligns
4850 surface points extracted from each segmentation contours using a two-step Seg-
ResNet with the DiVRoC loss. We select one scan as reference and uniformly
subsample points on each of the 17 bones - in total 720. We filter out images with a segmentation alignment to the reference below a defined Dice threshold providing the landmarks for three different Dice thresholds (0.8, 0.875, 0.9): `dataset/data/graz/lms_dsc_X.pth`.


## UV maps
Please use the provided scripts `dataset/jsrt_create_uv_maps.py` and `dataset/graz_create_uv_maps.py` to create the UV maps for the JSRT and GrazPedWri datasets. The UV maps are stored in the 'dataset/data' folder.
Note, that currently only the "polar" version of the UV maps work for the graz dataset.

## Training
We use Clear-ML to log and monitor our experiments. Your can either comment out the Clear-ML code lines or run it offline (see [here](https://clear.ml/docs/latest/docs/faq/#can-i-run-clearml-task-while-working-offline---)) if you do not want to create a free account.
To reproduce our results you can run the training script `training/train.py` with the provided configuration in `training/hyper_params.py`.
Please provide the index of the gpu you want to use as an argument to the script (e.g. `python train.py --gpu_id 0`) and make sure, your working directory is pointing to `DenseSeg`.
To change the dataset, please adjust the index in line 16 of `training/train.py` to either 0 (GrazPedWri) or 1 (JSRT).