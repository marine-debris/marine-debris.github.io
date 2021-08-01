# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: spectral_extraction.py extraction of the spectral signature, indices or texture features
             in a hdf5 table format for analysis and for the pixel-level semantic segmentation with 
             random forest classifier.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from osgeo import gdal
from os.path import dirname as up

root_path = up(up(os.path.abspath(__file__)))

sys.path.append(os.path.join(root_path, 'utils'))
from assets import s2_mapping, cat_mapping, conf_mapping, indexes_mapping, texture_mapping

rev_cat_mapping = {v:k for k,v in cat_mapping.items()}
rev_conf_mapping = {v:k for k,v in conf_mapping.items()}

def ImageToDataframe(RefImage, cols_mapping = {}, keep_annotated = True, coordinates = True):
    # This function transform an image with the associated class and 
    # confidence tif files (_cl.tif and _conf.tif) to a dataframe

    # Read patch
    ds = gdal.Open(RefImage)
    IM = np.copy(ds.ReadAsArray())

    # Read associated confidence level patch
    ds_conf = gdal.Open(os.path.join(up(RefImage), '_'.join(os.path.basename(RefImage).split('.tif')[0].split('_')[:4]) + '_conf.tif'))
    IM_conf = np.copy(ds_conf.ReadAsArray())[np.newaxis, :, :]
    
    # Read associated class patch
    ds_cl = gdal.Open(os.path.join(up(RefImage), '_'.join(os.path.basename(RefImage).split('.tif')[0].split('_')[:4]) + '_cl.tif'))
    IM_cl = np.copy(ds_cl.ReadAsArray())[np.newaxis, :, :]
    
    # Stack all these together
    IM_T = np.moveaxis(np.concatenate([IM, IM_conf, IM_cl], axis = 0), 0, -1)
    
    if coordinates:
        # Get the coordinates in space.
        padfTransform = ds.GetGeoTransform()
        
        x_coords, y_coords = np.meshgrid(range(IM_T.shape[0]), range(IM_T.shape[1]), indexing='ij')
        
        Xp = padfTransform[0] + y_coords*padfTransform[1] + x_coords*padfTransform[2];
        Yp = padfTransform[3] + y_coords*padfTransform[4] + x_coords*padfTransform[5]
        
        # shift to the center of the pixel
        Xp -= padfTransform[5] / 2.0
        Yp -= padfTransform[1] / 2.0
        XpYp = np.dstack((Xp,Yp))
        IM_T = np.concatenate((IM_T, XpYp), axis=2)
        
    bands = IM_T.shape[-1]
    IM_VECT = IM_T.reshape([-1,bands])
    
    if keep_annotated and coordinates:
        IM_VECT = IM_VECT[IM_VECT[:,-3] > 0] # Keep only based on non zero class
    elif keep_annotated and not coordinates:
        IM_VECT = IM_VECT[IM_VECT[:,-1] > 0] # Keep only based on non zero class
        
    if cols_mapping:
        IM_df = pd.DataFrame({k:IM_VECT[:,v] for k, v in cols_mapping.items()})
    else:
        IM_df = pd.DataFrame(IM_VECT)
        
    if coordinates:
        IM_df['XCoords'] = IM_VECT[:,-2]
        IM_df['YCoords'] = IM_VECT[:,-1]
    
    IM_df.date = ds.GetMetadataItem("TIFFTAG_DATETIME")
    ds = None
    ds_conf = None
    ds_cl = None
    
    IM_df["Class"] = IM_df["Class"].apply(lambda x: rev_cat_mapping[x])
    IM_df['Confidence'] = IM_df['Confidence'].apply(lambda x: rev_conf_mapping[x])
    
    return IM_df

def main(options):
    
    # Which features?
    if options['type']=='s2':
        mapping = s2_mapping
        h5_prefix = 'dataset'
        
        # Get patches files without _cl and _conf associated files
        patches = glob(os.path.join(options['path'], 'patches', '*/*.tif'))
        
    elif options['type']=='indices':
        mapping = indexes_mapping
        h5_prefix = 'dataset_si'
        
        # Get patches files without _cl and _conf associated files
        patches = glob(os.path.join(options['path'], 'indices', '*/*.tif'))
        
    elif options['type']=='texture':
        mapping = texture_mapping
        h5_prefix = 'dataset_glcm'
        
        # Get patches files without _cl and _conf associated files
        patches = glob(os.path.join(options['path'], 'texture', '*/*.tif'))
        
    else:
        raise AssertionError("Wrong Type, select between s2, indices or texture")
        
    patches = [p for p in patches if ('_cl.tif' not in p) and ('_conf.tif' not in p)]

    # Read splits
    X_train = np.genfromtxt(os.path.join(options['path'], 'splits','train_X.txt'),dtype='str')
    
    X_val = np.genfromtxt(os.path.join(options['path'], 'splits','val_X.txt'),dtype='str')
    
    X_test = np.genfromtxt(os.path.join(options['path'], 'splits','test_X.txt'),dtype='str')
    
    dataset_name = os.path.join(options['path'], h5_prefix + '_nonindex.h5')
    hdf = pd.HDFStore(dataset_name, mode = 'w')
    
    # For each patch extract the spectral signatures and store them
    for im_name in tqdm(patches):

        # Get date_tile_image info

        img_name = '_'.join(os.path.basename(im_name).split('.tif')[0].split('_')[1:4])
        
        # Generate Dataframe from Image
        if img_name in X_train:
            split = 'train'
            temp = ImageToDataframe(im_name, mapping)
        elif img_name in X_val:
            split = 'val'
            temp = ImageToDataframe(im_name, mapping)
        elif img_name in X_test:
            split = 'test'
            temp = ImageToDataframe(im_name, mapping)
        else:
            raise AssertionError("Image not in train,val,test splits")
        
        # Update Satellite and Date info
        temp['Date'] = os.path.splitext(os.path.basename(im_name))[0].split('_')[1]
        temp['Tile'] = os.path.splitext(os.path.basename(im_name))[0].split('_')[2]
        temp['Image'] = os.path.splitext(os.path.basename(im_name))[0].split('_')[3]
        # Store data
        hdf.append(split, temp, format='table', data_columns=True, min_itemsize={'Class':27,
                                                                                 'Confidence':8,
                                                                                 'Date':8,
                                                                                 'Image':3,
                                                                                 'Tile':5})
    
    hdf.close()
    
    # Read the stored file and fix an indexing problem (indexes were not incremental and unique)
    hdf_old = pd.HDFStore(dataset_name, mode = 'r')
    
    df_train = hdf_old['train'].copy(deep=True)
    df_val = hdf_old['val'].copy(deep=True)
    df_test = hdf_old['test'].copy(deep=True)
    
    df_train.reset_index(drop = True, inplace = True)
    df_val.reset_index(drop = True, inplace = True)
    df_test.reset_index(drop = True, inplace = True)

    hdf_old.close()
    
    # Store the fixed table to a new dataset file
    dataset_name_fixed = os.path.join(options['path'], h5_prefix+'.h5')

    df_train.to_hdf(dataset_name_fixed, key='train', mode='a', format='table', data_columns=True)
    df_val.to_hdf(dataset_name_fixed, key='val', mode='a', format='table', data_columns=True)
    df_test.to_hdf(dataset_name_fixed, key='test', mode='a', format='table', data_columns=True)
    
    os.remove(dataset_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--path', default=os.path.join(root_path, 'data'), help='Path to dataset')
    parser.add_argument('--type', default='s2', type=str, help=' Select between s2, indices or texture for Spectral Signatures, Produced Indices or GLCM Textures, respectively')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    main(options)
