# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: evaluation.py includes the code in order to produce
             the evaluation for each patch in the test set.
'''

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(up(os.path.abspath(__file__)))
from resnet import ResNet
from dataloader import GenDEBRIS_ML, bands_mean, bands_std

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from metrics import Evaluation_ML, print_confusion_matrix_ML
from assets import labels

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(filename=os.path.join(root_path, 'logs','evaluating_resnet.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)

###############################################################
# Training
###############################################################

def main(options):
    # Transformations
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)
    
    # Construct Data loader
    
    dataset_test = GenDEBRIS_ML('test', transform=transform_test, standardization = standardization, agg_to_water = options['agg_to_water'])
    
    test_loader = DataLoader(dataset_test, 
                             batch_size = options['batch'], 
                             shuffle = False)
    
    global labels
    # Aggregate Distribution Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water
    if options['agg_to_water']:
        labels = labels[:-4] # Drop Mixed Water, Wakes, Cloud Shadows, Waves
                        
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = ResNet(input_bands = options['input_channels'], 
                   output_classes = options['output_channels'])
        
    model.to(device)
    
    # Load model from specific epoch to continue the training or start the evaluation
    model_file = options['model_path']
    logging.info('Loading model files from folder: %s' % model_file)

    checkpoint = torch.load(model_file, map_location = device)
    model.load_state_dict(checkpoint)

    del checkpoint  # dereference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_predicted = []
    predicted_probs = []
    
    with torch.no_grad():

        for (image, target) in tqdm(test_loader, desc="testing"):

            image = image.to(device)
            target = target.to(device)

            logits = model(image)                  
            
            probs = torch.sigmoid(logits).cpu().numpy()
            target = target.cpu().numpy()
            
            predicted_probs += list(probs)
            y_true += list(target)
                
        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= options['threshold']).astype(np.float32)
        y_true = np.asarray(y_true)
            
        y_predicted = np.asarray(y_predicted)
        y_true = np.asarray(y_true)
        
        ###############################################################
        # Store Scores
        ###############################################################
        
        # Store Results
        output = {'S2_'+ dataset_test.ROIs[i] + '.tif' :y_predicted[i,:].tolist() for i in range(y_predicted.shape[0])}
        with open(os.path.join(root_path, 'data', 'predicted_labels_mapping.txt'), 'w') as outfile:
            json.dump(output, outfile)
        
        # Start Evaluation
        acc = Evaluation_ML(y_predicted, predicted_probs, y_true)
        logging.info("\n")
        logging.info("STATISTICS : \n")
        
        print("Evaluation: " + str(acc))
        logging.info("Evaluation: " + str(acc))
        
        # Calculate Classification Report
        cl_report = classification_report(y_true, y_predicted, target_names=labels, digits=4, output_dict=False)
        print(cl_report)
        logging.info(cl_report)
        
        # Calculate Per Label Confusion Matrix
        vis_arr = multilabel_confusion_matrix(y_true, y_predicted)
        
        print("Per Label Confusion Matrix: \n")
        logging.info("Per Label Confusion Matrix: \n")
        
        print('Assigned or Not during annotation ("Ground-Truth") and Predicted or Not by the model. \n')
        logging.info('Assigned or Not during annotation ("Ground-Truth") and Predicted or Not by the model. \n')
        
        for cfs_matrix, label in zip(vis_arr, labels):
            df_cm = print_confusion_matrix_ML(cfs_matrix, label, ["Not Assigned", "Assigned"], ["No", "Yes"])
            print(df_cm.to_string() + ' \n')
            logging.info(df_cm.to_string() + ' \n')
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--agg_to_water', default=True, type=bool,  help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')
    parser.add_argument('--batch', default=5, type=int, help='Number of epochs to run')
    parser.add_argument('--threshold', default=0.5, type=int, help='threshold for evaluation')
    
    # ResNet parameters
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    
    # ResNet model path
    parser.add_argument('--model_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models', '18', 'model.pth'), help='Path to ResNet pytorch model')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    main(options)
