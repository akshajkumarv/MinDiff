import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from arguments import args
       
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##########################
### SETTINGS
##########################
# Hyperparameters
LEARNING_RATE = args.lr
NUM_STEPS = args.steps
RUN = args.run
FLOOD_LEVEL = args.flood_level
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
MIN_DIFF_WEIGHT = args.mindiff_weight
MIN_DIFF = args.mindiff
print(args)

DEVICE = 'cuda:0'

torch.manual_seed(RUN)
torch.cuda.manual_seed_all(RUN)
np.random.seed(RUN)
os.environ['PYTHONHASHSEED'] = str(RUN)

def kernel_product(a, b, l=0.1):
    a = a.view(-1)
    b = b.view(-1)
    a = torch.t(a).unsqueeze(1)
    b = b.unsqueeze(0)
    K = torch.mean(torch.exp(-((a-b)**2)/(l**2)))   
    return K

def adjust_learning_rate(optimizer, step, LEARNING_RATE):
    lr = LEARNING_RATE * (0.1 ** (step // 16000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for width in [5, 7, 9, 13, 19, 55, 64]:

    from data import train_dataset, train_blond_men_dataset, train_blond_women_dataset

    from architecture import make_resnet18k

    print("*"*10 + " width = ",str(width) + " " + "*"*10)

    if MIN_DIFF == False:    
        PATH = 'saved_models/batch_size/%s/run_%s/original_model_flooding_%s_weight_decay_%s/'%(BATCH_SIZE, RUN, FLOOD_LEVEL, WEIGHT_DECAY)  
        print(PATH) 
    if MIN_DIFF == True:
        PATH = 'saved_models/batch_size/%s/run_%s/min_diff_model_weight_%s_flooding_%s_weight_decay_%s/'%(BATCH_SIZE, RUN, MIN_DIFF_WEIGHT, FLOOD_LEVEL, WEIGHT_DECAY)
        print(PATH)

    if not os.path.exists(PATH+'width_%s/steps'%(width)):
        os.makedirs(PATH+'width_%s/steps'%(width))
    if not os.path.exists(PATH+'width_%s/epochs'%(width)):
        os.makedirs(PATH+'width_%s/epochs'%(width))

    print(torch.cuda.device_count())
    
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=8)

    train_blond_women_loader = DataLoader(dataset=train_blond_women_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=6)

    train_blond_men_loader = DataLoader(dataset=train_blond_men_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=6)

    model = make_resnet18k(k=width)
    ##########################
    ### COST AND OPTIMIZER
    ##########################
    #### DATA PARALLEL START ####
    model = nn.DataParallel(model, device_ids=[0])
    #### DATA PARALLEL END ####
    model.to(DEVICE)

    cost_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print('Length of Loader: ', len(train_loader))
    train_blond_women_iter = iter(train_blond_women_loader)
    train_blond_men_iter = iter(train_blond_men_loader)

    step = 1
    epoch = 1

    while step <= NUM_STEPS:
        for batch_idx, (features, _, targets) in enumerate(train_loader):

            model.train()

            adjust_learning_rate(optimizer, step, LEARNING_RATE)
            for param_group in optimizer.param_groups:
                step_lr = param_group['lr']
        
            optimizer.zero_grad()
        
            if MIN_DIFF == True:
                try:
                    train_blond_women_features, train_blond_women_attributes, train_blond_women_targets = next(train_blond_women_iter)
                except StopIteration: 
                    train_blond_women_iter = iter(train_blond_women_loader)
                    train_blond_women_features, train_blond_women_attributes, train_blond_women_targets = next(train_blond_women_iter)

                try:
                    train_blond_men_features, train_blond_men_attributes, train_blond_men_targets = next(train_blond_men_iter) 
                except StopIteration: 
                    train_blond_men_iter = iter(train_blond_men_loader)
                    train_blond_men_features, train_blond_men_attributes, train_blond_men_targets = next(train_blond_men_iter)

                inputs = torch.cat((features.float(), train_blond_women_features.float(), train_blond_men_features.float()))
                inputs = inputs.to(DEVICE)
                targets = targets.unsqueeze(1).float().to(DEVICE)
            
                logits = model(inputs)
                probas = torch.sigmoid(logits)

                primary_loss = cost_fn(probas[:targets.size(0)], targets)

                x1x1 = kernel_product(probas[targets.size(0):targets.size(0)+train_blond_women_targets.size(0)], probas[targets.size(0):targets.size(0)+train_blond_women_targets.size(0)])
                x1x2 = kernel_product(probas[targets.size(0):targets.size(0)+train_blond_women_targets.size(0)], probas[targets.size(0)+train_blond_women_targets.size(0):])
                x2x2 = kernel_product(probas[targets.size(0)+train_blond_women_targets.size(0):], probas[targets.size(0)+train_blond_women_targets.size(0):])
                mmd_loss = x1x1 - (2*x1x2) + x2x2
                mmd_loss = mmd_loss.to(DEVICE)    
  
                cost = (primary_loss-FLOOD_LEVEL).abs() + FLOOD_LEVEL + (MIN_DIFF_WEIGHT*mmd_loss)

                cost.backward()
                optimizer.step()

                ### LOGGING
                if (step == 1) or (step % 500 == 0):
                    print(step)
                    CHECKPOINT = PATH+'width_%s/steps/step_%s.pt'%(width, step)
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, CHECKPOINT)
            
            if MIN_DIFF == False:
                features = features.float().to(DEVICE)
                targets = targets.unsqueeze(1).float().to(DEVICE)

                logits = model(features)
                probas = torch.sigmoid(logits)
                     
                primary_loss = cost_fn(probas, targets)
                cost = primary_loss
                cost = (cost-FLOOD_LEVEL).abs() + FLOOD_LEVEL
            
                cost.backward()
                optimizer.step()

                ### LOGGING
                if (step == 1) or (step % 500 == 0):
                    print(step)
                    CHECKPOINT = PATH+'width_%s/steps/step_%s.pt'%(width, step)
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, CHECKPOINT)

            step += 1
            if step == (NUM_STEPS+1):
                break

        CHECKPOINT = PATH+'width_%s/epochs/epoch_%s.pt'%(width, epoch)
        torch.save({
            'epoch': epoch,
            'step': step-1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPOINT)
        epoch += 1

    CHECKPOINT = PATH+'width_%s/width_%s.pt'%(width, width)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, CHECKPOINT)

