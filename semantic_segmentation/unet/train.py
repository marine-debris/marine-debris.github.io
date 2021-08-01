# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: train.py includes the training process for the
             pixel-level semantic segmentation.
'''

import os
import ast
import sys
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append(up(os.path.abspath(__file__)))
from unet import UNet
from dataloader import GenDEBRIS, bands_mean, bands_std, RandomRotationTransform , class_distr, gen_weights

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from metrics import Evaluation

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(filename=os.path.join(root_path, 'logs','log_unet.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)

def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    # DataLoader Workers Reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

###############################################################
# Training                                                    #
###############################################################

def main(options):
    # Reproducibility
    # Limit the number of sources of nondeterministic behavior 
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(root_path, 'logs', options['tensorboard']))
    
    # Transformations
    
    transform_train = transforms.Compose([transforms.ToTensor(),
                                    RandomRotationTransform([-90, 0, 90, 180]),
                                    transforms.RandomHorizontalFlip()])
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    standardization = transforms.Normalize(bands_mean, bands_std)
    
    # Construct Data loader
    
    if options['mode']=='train':
        
        dataset_train = GenDEBRIS('train', transform=transform_train, standardization = standardization, agg_to_water = options['agg_to_water'])
        dataset_test = GenDEBRIS('val', transform=transform_test, standardization = standardization, agg_to_water = options['agg_to_water'])
        
        train_loader = DataLoader(  dataset_train, 
                                    batch_size = options['batch'], 
                                    shuffle = True,
                                    num_workers = options['num_workers'],
                                    pin_memory = options['pin_memory'],
                                    prefetch_factor = options['prefetch_factor'],
                                    persistent_workers= options['persistent_workers'],
                                    worker_init_fn=seed_worker,
                                    generator=g)
        
        test_loader = DataLoader(   dataset_test, 
                                    batch_size = options['batch'], 
                                    shuffle = False,
                                    num_workers = options['num_workers'],
                                    pin_memory = options['pin_memory'],
                                    prefetch_factor = options['prefetch_factor'],
                                    persistent_workers= options['persistent_workers'],
                                    worker_init_fn=seed_worker,
                                    generator=g)
        
    elif options['mode']=='test':
        
        dataset_test = GenDEBRIS('test', transform=transform_test, standardization = standardization, agg_to_water = options['agg_to_water'])
    
        test_loader = DataLoader(   dataset_test, 
                                    batch_size = options['batch'], 
                                    shuffle = False,
                                    num_workers = options['num_workers'],
                                    pin_memory = options['pin_memory'],
                                    prefetch_factor = options['prefetch_factor'],
                                    persistent_workers= options['persistent_workers'],
                                    worker_init_fn=seed_worker,
                                    generator=g)
    else:
        raise                        
    
    # Use gpu or cpu
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = UNet(input_bands = options['input_channels'], 
                 output_classes = options['output_channels'], 
                 hidden_channels = options['hidden_channels'])

    model.to(device)

    # Load model from specific epoch to continue the training or start the evaluation
    if options['resume_from_epoch'] > 1:
        
        resume_model_dir = os.path.join(options['checkpoint_path'], str(options['resume_from_epoch']))
        model_file = os.path.join(resume_model_dir, 'model.pth')
        logging.info('Loading model files from folder: %s' % model_file)

        checkpoint = torch.load(model_file, map_location = device)
        model.load_state_dict(checkpoint)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global class_distr
    # Aggregate Distribution Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water
    if options['agg_to_water']:
        agg_distr = sum(class_distr[-4:]) # Density of Mixed Water, Wakes, Cloud Shadows, Waves
        class_distr[6] += agg_distr       # To Water
        class_distr = class_distr[:-4]    # Drop Mixed Water, Wakes, Cloud Shadows, Waves

    # Weighted Cross Entropy Loss & adam optimizer
    weight = gen_weights(class_distr, c = options['weight_param'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean', weight=weight.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay'])

    # Learning Rate scheduler
    if options['reduce_lr_on_plateau']==1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, options['lr_steps'], gamma=0.1, verbose=True)

    # Start training
    start = options['resume_from_epoch'] + 1
    epochs = options['epochs']
    eval_every = options['eval_every']

    
    # Write model-graph to Tensorboard
    if options['mode']=='train':
        dataiter = iter(train_loader)
        image_temp, _ = dataiter.next()
        writer.add_graph(model, image_temp.to(device))
        
        ###############################################################
        # Start Training                                              #
        ###############################################################
        model.train()
        
        for epoch in range(start, epochs+1):
            training_loss = []
            training_batches = 0
            
            i_board = 0
            for (image, target) in tqdm(train_loader, desc="training"):
                
                image = image.to(device)
                target = target.to(device)
    
                optimizer.zero_grad()
                
                logits = model(image)
                
                loss = criterion(logits, target)
    
                loss.backward()
    
                training_batches += target.shape[0]
    
                training_loss.append((loss.data*target.shape[0]).tolist())
                
                optimizer.step()
                
                # Write running loss
                writer.add_scalar('training loss', loss , (epoch - 1) * len(train_loader)+i_board)
                i_board+=1
            
            logging.info("Training loss was: " + str(sum(training_loss) / training_batches))
            
            ###############################################################
            # Start Evaluation                                            #
            ###############################################################
            
            if epoch % eval_every == 0 or epoch==1:
                model.eval()
    
                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []
                
                with torch.no_grad():
                    for (image, target) in tqdm(test_loader, desc="testing"):
    
                        image = image.to(device)
                        target = target.to(device)
    
                        logits = model(image)
                        
                        loss = criterion(logits, target)
                                    
                        # Accuracy metrics only on annotated pixels
                        logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                        logits = logits.reshape((-1,options['output_channels']))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]
                        
                        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                        target = target.cpu().numpy()
                        
                        test_batches += target.shape[0]
                        test_loss.append((loss.data*target.shape[0]).tolist())
                        y_predicted += probs.argmax(1).tolist()
                        y_true += target.tolist()
                            
                        
                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)
                    
                    ####################################################################
                    # Save Scores to the .log file and visualize also with tensorboard #
                    ####################################################################
                    
                    acc = Evaluation(y_predicted, y_true)
                    logging.info("\n")
                    logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
                    logging.info("STATISTICS AFTER EPOCH " +str(epoch) + ": \n")
                    logging.info("Evaluation: " + str(acc))
    
    
                    logging.info("Saving models")
                    model_dir = os.path.join(options['checkpoint_path'], str(epoch))
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
                    
                    writer.add_scalars('Loss per epoch', {'Test loss':sum(test_loss) / test_batches, 
                                                          'Train loss':sum(training_loss) / training_batches}, 
                                       epoch)
                    
                    writer.add_scalar('Precision/test macroPrec', acc["macroPrec"] , epoch)
                    writer.add_scalar('Precision/test microPrec', acc["microPrec"] , epoch)
                    writer.add_scalar('Precision/test weightPrec', acc["weightPrec"] , epoch)
                    
                    writer.add_scalar('Recall/test macroRec', acc["macroRec"] , epoch)
                    writer.add_scalar('Recall/test microRec', acc["microRec"] , epoch)
                    writer.add_scalar('Recall/test weightRec', acc["weightRec"] , epoch)
                    
                    writer.add_scalar('F1/test macroF1', acc["macroF1"] , epoch)
                    writer.add_scalar('F1/test microF1', acc["microF1"] , epoch)
                    writer.add_scalar('F1/test weightF1', acc["weightF1"] , epoch)
                    
                    writer.add_scalar('IoU/test MacroIoU', acc["IoU"] , epoch)
                    
    
                if options['reduce_lr_on_plateau'] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()
                    
                model.train()
               
    # CODE ONLY FOR EVALUATION - TESTING MODE !
    elif options['mode']=='test':
        
        model.eval()

        test_loss = []
        test_batches = 0
        y_true = []
        y_predicted = []
        
        with torch.no_grad():
            for (image, target) in tqdm(test_loader, desc="testing"):

                image = image.to(device)
                target = target.to(device)

                logits = model(image)
                
                loss = criterion(logits, target)

                # Accuracy metrics only on annotated pixels
                logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                logits = logits.reshape((-1,options['output_channels']))
                target = target.reshape(-1)
                mask = target != -1
                logits = logits[mask]
                target = target[mask]
                
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                target = target.cpu().numpy()
                
                test_batches += target.shape[0]
                test_loss.append((loss.data*target.shape[0]).tolist())
                y_predicted += probs.argmax(1).tolist()
                y_true += target.tolist()
                
            y_predicted = np.asarray(y_predicted)
            y_true = np.asarray(y_true)
            
            ####################################################################
            # Save Scores to the .log file                                     #
            ####################################################################
            acc = Evaluation(y_predicted, y_true)
            logging.info("\n")
            logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
            logging.info("STATISTICS: \n")
            logging.info("Evaluation: " + str(acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--agg_to_water', default=True, type=bool,  help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')
  
    parser.add_argument('--mode', default='train', help='select between train or test ')
    parser.add_argument('--epochs', default=45, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=5, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')
    
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    parser.add_argument('--hidden_channels', default=16, type=int, help='Number of hidden features')
    parser.add_argument('--weight_param', default=1.03, type=float, help='Weighting parameter for Loss Function')

    # Optimization
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='learning rate decay')
    parser.add_argument('--reduce_lr_on_plateau', default=0, type=int, help='reduce learning rate when no increase (0 or 1)')
    parser.add_argument('--lr_steps', default='[40]', type=str, help='Specify the steps that the lr will be reduced')

    # Evaluation/Checkpointing
    parser.add_argument('--checkpoint_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models'), help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    # misc
    parser.add_argument('--num_workers', default=1, type=int, help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=False, type=bool, help='Use pinned memory or not')
    parser.add_argument('--prefetch_factor', default=1, type=int, help='Number of sample loaded in advance by each worker')
    parser.add_argument('--persistent_workers', default=True, type=bool, help='This allows to maintain the workers Dataset instances alive.')
    parser.add_argument('--tensorboard', default='tsboard_segm', type=str, help='Name for tensorboard run')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    # lr_steps list or single float
    lr_steps = ast.literal_eval(options['lr_steps'])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise
        
    options['lr_steps'] = lr_steps
    
    logging.info('parsed input parameters:')
    logging.info(json.dumps(options, indent = 2))
    main(options)
