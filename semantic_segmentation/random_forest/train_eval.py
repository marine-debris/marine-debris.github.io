# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: train_eval.py includes the training and evaluation process for the
             pixel-level semantic segmentation with random forest.
'''

import os
import ast
import sys
import time
import json
import random
import logging
import rasterio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
from random_forest import rf_classifier, bands_mean

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from metrics import confusion_matrix
from assets import conf_mapping, rf_features, cat_mapping_vec

random.seed(0)
np.random.seed(0)

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(filename=os.path.join(root_path, 'logs','evaluation_rf.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)


###############################################################
# Training                                                    #
###############################################################

def main(options):
    
    # Load Spectral Signatures, Spectral Indices and GLCM texture features
    hdf_ss = pd.HDFStore(os.path.join(options['path'], 'dataset.h5'), mode = 'r')
    df_train_ss = hdf_ss.select('train')
    df_val_ss = hdf_ss.select('val')
    df_test_ss = hdf_ss.select('test')
    hdf_ss.close()
    
    hdf_si = pd.HDFStore(os.path.join(options['path'], 'dataset_si.h5'), mode = 'r')
    df_train_si = hdf_si.select('train')
    df_val_si = hdf_si.select('val')
    df_test_si = hdf_si.select('test')
    hdf_si.close()
    
    hdf_glcm = pd.HDFStore(os.path.join(options['path'], 'dataset_glcm.h5'), mode = 'r')
    df_train_glcm = hdf_glcm.select('train')
    df_val_glcm = hdf_glcm.select('val')
    df_test_glcm = hdf_glcm.select('test')
    hdf_glcm.close()
    
    # Join for each split the ss, si and glcm features
    df_train = df_train_ss.merge(df_train_si, left_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], 
                                 right_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], suffixes=('', '_si'))

    df_val = df_val_ss.merge(df_val_si, left_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], 
                                 right_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], suffixes=('', '_si'))

    df_test = df_test_ss.merge(df_test_si, left_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], 
                                 right_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], suffixes=('', '_si'))
    
    df_train = df_train.merge(df_train_glcm, left_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], 
                                 right_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], suffixes=('', '_glcm'))

    df_val = df_val.merge(df_val_glcm, left_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], 
                                 right_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], suffixes=('', '_glcm'))

    df_test = df_test.merge(df_test_glcm, left_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], 
                                 right_on=['Date', 'Tile', 'Image', 'XCoords', 'YCoords'], suffixes=('', '_glcm'))
    
    # Calculate weights for each sample on Train/ Val splits based on Confidence Level
    df_train['Weight'] = 1/df_train['Confidence'].apply(lambda x: conf_mapping[x])
    
    # Aggregate classes to Water Super class
    for agg_class in options['agg_to_water']:
        df_train.loc[df_train['Class'] == agg_class, 'Class'] = 'Marine Water'
        df_val.loc[df_val['Class'] == agg_class, 'Class'] = 'Marine Water'
        df_test.loc[df_test['Class'] == agg_class, 'Class'] = 'Marine Water'
    
    # Keep selected features and transform to numpy array
    X_train = df_train[rf_features].values
    y_train = df_train['Class'].values
    weight_train = df_train['Weight'].values
    
    if options['eval_set'] == 'test':
        X_test = df_test[rf_features].values
        y_test = df_test['Class'].values
        
    elif options['eval_set'] == 'val':
        X_test = df_val[rf_features].values
        y_test = df_val['Class'].values
        
    else:
        raise
    
    print('Number of Input features: ', X_train.shape[1])
    print('Train: ',X_train.shape[0])
    print('Test: ',X_test.shape[0])
    
    logging.info('Number of Input features: ' + str(X_train.shape[1]))
    logging.info('Train: ' + str(X_train.shape[0]))
    logging.info('Test: ' + str(X_test.shape[0]))
        
    # Training
    print('Started training')
    logging.info('Started training')
    
    start_time = time.time()
    rf_classifier.fit(X_train, y_train, **dict(rf__sample_weight=weight_train))
    
    print("Training finished after %s seconds" % (time.time() - start_time))
    logging.info("Training finished after %s seconds" % (time.time() - start_time))
    
    cl_path = os.path.join(up(os.path.abspath(__file__)), 'rf_classifier.joblib')
    dump(rf_classifier, cl_path)
    print("Classifier is saved at: " +str(cl_path))
    logging.info("Classifier is saved at: " +str(cl_path))
    
    print('\t\t Random Forest Results on '+options['eval_set']+' Set')
    conf_mat = confusion_matrix(y_test, rf_classifier.predict(X_test), rf_classifier.classes_)
    logging.info("Confusion Matrix:  \n" + str(conf_mat.to_string()))
    print("Confusion Matrix:  \n" + str(conf_mat.to_string()))
    
    if options['predict_masks']:
    
        path = os.path.join(options['path'], 'patches')
        ROIs = np.genfromtxt(os.path.join(options['path'], 'splits', 'test_X.txt'),dtype='str')
        
        impute_nan = np.tile(bands_mean, (256,256,1))
                    
        for roi in tqdm(ROIs):
        
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])             # Get Folder Name
            roi_name = '_'.join(['S2'] + roi.split('_'))                    # Get File Name
            roi_file = os.path.join(path, roi_folder,roi_name + '.tif')     # Get File path
        
            os.makedirs(options['gen_masks_path'], exist_ok=True)
        
            output_image = os.path.join(options['gen_masks_path'], os.path.basename(roi_file).split('.tif')[0] + '_rf.tif')
            
            # Load the image patch and metadata
            with rasterio.open(roi_file, mode ='r') as src:
                tags = src.tags().copy()
                meta = src.meta
                image = src.read()
                image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
                dtype = src.read(1).dtype
        
            # Update meta to reflect the number of layers
            meta.update(count = 1)
            
            # Preprocessing
            # Fill image nan with mean
            nan_mask = np.isnan(image)
            image[nan_mask] = impute_nan[nan_mask]
            
            sz1 = image.shape[0]
            sz2 = image.shape[1]
            image_features = np.reshape(image, (sz1*sz2, -1))
            
            # Load Indices
            si_filename = os.path.join(options['path'], 'indices', roi_folder,roi_name + '_si.tif')
            with rasterio.open(si_filename, mode ='r') as src:
                image_si = src.read()
                image_si = np.moveaxis(image_si, (0, 1, 2), (2, 0, 1))
                
                si_sz1 = image_si.shape[0]
                si_sz2 = image_si.shape[1]
                si_image_features = np.reshape(image_si, (si_sz1*si_sz2, -1))

                si_image_features = np.nan_to_num(si_image_features)
            
            # Load Texture
            glcm_filename = os.path.join(options['path'], 'texture', roi_folder,roi_name + '_glcm.tif')
            with rasterio.open(glcm_filename, mode ='r') as src:
                image_glcm = src.read()
                image_glcm = np.moveaxis(image_glcm, (0, 1, 2), (2, 0, 1))
                
                glcm_sz1 = image_glcm.shape[0]
                glcm_sz2 = image_glcm.shape[1]
                glcm_image_features = np.reshape(image_glcm, (glcm_sz1*glcm_sz2, -1))

                glcm_image_features = np.nan_to_num(glcm_image_features)
                
            # Concatenate all features
            image_features = np.concatenate([image_features, si_image_features, glcm_image_features], axis=1)  
        
            # Write it
            with rasterio.open(output_image, 'w', **meta) as dst:
                
                # use classifier to predict labels for the whole image
                predictions = rf_classifier.predict(image_features)  
    
                predicted_labels = np.reshape(predictions, (sz1,sz2))
    
                class_ind = cat_mapping_vec(predicted_labels).astype(dtype).copy()
                dst.write_band(1, class_ind) # In order to be in the same dtype
    
                dst.update_tags(**tags)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options

    # Evaluation/Checkpointing
    parser.add_argument('--path', default=os.path.join(root_path, 'data'), help='Path to dataset')

    # Produce Predicted Masks
    parser.add_argument('--eval_set', default='test', type=str, help="Set for the evaluation 'val' or 'test' for final testing")
    parser.add_argument('--predict_masks', default= True, type=bool, help='Generate test set prediction masks?')
    parser.add_argument('--gen_masks_path', default=os.path.join(root_path, 'data', 'predicted_rf'), help='Path to where to produce store predictions')

    parser.add_argument('--agg_to_water', default='["Mixed Water", "Wakes", "Cloud Shadows", "Waves"]', type=str, help='Specify the Classes that will aggregate with Marine Water')
    
    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    # agg_to_water list fix
    agg_to_water = ast.literal_eval(options['agg_to_water'])
    if type(agg_to_water) is list:
        pass
    elif type(agg_to_water) is str:
        agg_to_water = [agg_to_water]
    else:
        raise
        
    options['agg_to_water'] = agg_to_water
    
    logging.info('parsed input parameters:')
    logging.info(json.dumps(options, indent = 2))
    main(options)
