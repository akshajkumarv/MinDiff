from arguments import args
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

def compute_loss_accuracy(net, loader, threshold, device):
    correct_pred, num_examples = 0.0, 0.0
    running_loss = 0.0
    cost_fn = torch.nn.BCELoss()

    with torch.no_grad():
        for i, (features, attributes, targets) in enumerate(loader):
            
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()

            logits = net(features)
            probas = F.sigmoid(logits)

            primary_loss = cost_fn(probas, targets)
            running_loss += primary_loss.item()*targets.size(0)

            predicted_labels = logits > threshold
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    loss = running_loss/num_examples
    accuracy = (correct_pred/num_examples)
    accuracy = accuracy.item()
    
    return loss, accuracy
    
def compute_confusion_matrix(net, loader, device):
    correct_pred, num_examples = 0.0, 0.0
    running_loss = 0.0

    logitslist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    threshold_list = []
    fnr_list = []
    tnr_list = []
    fpr_list = []
    tpr_list = []
    
    with torch.no_grad():
        for i, (features, attributes, targets) in enumerate(loader):
            
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()

            logits = net(features)
            
            logitslist = torch.cat([logitslist,logits.view(-1).cpu()])
            lbllist = torch.cat([lbllist,targets.view(-1).cpu()])

    num_examples = lbllist.shape[0]
    
    for threshold in np.linspace(-20.0, 20, num=10000):
        predicted_labels = logitslist > threshold

        correct_pred = (predicted_labels == lbllist).sum()
            
        # Confusion matrix
        conf_mat = confusion_matrix(lbllist.numpy(), predicted_labels.numpy())
        negative = np.sum(conf_mat, 1)[0]
        positive = np.sum(conf_mat, 1)[1]
        tpr = conf_mat[1, 1]/positive
        fpr = conf_mat[0, 1]/negative
        fnr = conf_mat[1, 0]/positive
        tnr = conf_mat[0, 0]/negative

        threshold_list.append(threshold)
        tnr_list.append(tnr)
        fnr_list.append(fnr)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return np.asarray(threshold_list), np.asarray(tnr_list), np.asarray(fnr_list), np.asarray(tpr_list), np.asarray(fpr_list)

DEVICE = 'cuda:0'
RUN = args.run
FLOOD_LEVEL = args.flood_level
WEIGHT_DECAY =args.weight_decay
BATCH_SIZE = args.batch_size
bs = 1024
FNR_GAP = args.fnr_gap
MIN_DIFF_WEIGHT = args.mindiff_weight
MIN_DIFF = args.mindiff
N = [5, 7, 9, 13, 19, 55, 64]
print(args)

if MIN_DIFF == False:    
    PATH = 'saved_models/batch_size/%s/run_%s/original_model_flooding_%s_weight_decay_%s/'%(BATCH_SIZE, RUN, FLOOD_LEVEL, WEIGHT_DECAY)  
if MIN_DIFF == True:
    PATH = 'saved_models/batch_size/%s/run_%s/min_diff_model_weight_%s_flooding_%s_weight_decay_%s/'%(BATCH_SIZE, RUN, MIN_DIFF_WEIGHT, FLOOD_LEVEL, WEIGHT_DECAY)

train_accuracy_equal_threshold_list = []
valid_max_accuracy_list = []
valid_max_accuracy_threshold_list = []
valid_accuracy_equal_threshold_list = []
valid_fnr_gap_list = []
valid_equal_threshold_fnr_gap_list = []
test_max_accuracy_list = []
test_accuracy_equal_threshold_list = []
test_fnr_gap_list = []
test_equal_threshold_fnr_gap_list = []

for n in N:

    from architecture import make_resnet18k

    print("*"*10 + " width = ",str(n) + " " + "*"*10)
    
    model = make_resnet18k(k=n)
    load_path = PATH + 'width_%s/width_%s.pt'%(n, n)
    print('Loading Model Checkpoint: ', load_path)
    model = nn.DataParallel(model, device_ids=[0])
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    from data import train_dataset, valid_dataset, valid_men_dataset, valid_women_dataset, valid_blond_men_dataset, valid_blond_women_dataset, valid_brunette_men_dataset, valid_brunette_women_dataset, test_dataset, test_blond_men_dataset, test_blond_women_dataset, test_brunette_men_dataset, test_brunette_women_dataset

    valid_men_loader = DataLoader(dataset=valid_men_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    valid_women_loader = DataLoader(dataset=valid_women_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)

    threshold, tnr_men, fnr_men, tpr_men, fpr_men = compute_confusion_matrix(model, valid_men_loader, DEVICE)
    _, tnr_women, fnr_women, tpr_women, fpr_women = compute_confusion_matrix(model, valid_women_loader, DEVICE)

    count_men = 0
    max_acc = 0
    max_acc_threshold_men = 0
    max_acc_threshold_women = 0
    for i in fnr_men:
        count_women = 0
        for j in fnr_women:            
            if np.absolute(i-j) <= FNR_GAP:
                accuracy = (182*tpr_men[count_men]/19867) + (2874*tpr_women[count_women]/19867) + (8276*tnr_men[count_men]/19867) + (8535*tnr_women[count_women]/19867)
                if accuracy >= max_acc:
                    max_acc = accuracy                
                    max_acc_threshold_men = threshold[count_men]
                    max_acc_threshold_women = threshold[count_women]
            count_women += 1
        count_men += 1
    valid_max_accuracy_threshold_list.append((max_acc_threshold_men, max_acc_threshold_women))
    valid_blond_men_loader = DataLoader(dataset=valid_blond_men_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    valid_blond_women_loader = DataLoader(dataset=valid_blond_women_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    valid_brunette_men_loader = DataLoader(dataset=valid_brunette_men_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    valid_brunette_women_loader = DataLoader(dataset=valid_brunette_women_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    _, s1 = compute_loss_accuracy(model, valid_blond_men_loader, max_acc_threshold_men, DEVICE)
    _, s2 = compute_loss_accuracy(model, valid_blond_women_loader, max_acc_threshold_women, DEVICE)
    _, s3 = compute_loss_accuracy(model, valid_brunette_men_loader, max_acc_threshold_men, DEVICE)
    _, s4 = compute_loss_accuracy(model, valid_brunette_women_loader, max_acc_threshold_women, DEVICE)
    acc = (182*s1/19867) + (2874*s2/19867) + (8276*s3/19867) + (8535*s4/19867)
    valid_max_accuracy_list.append(acc)
    valid_fnr_gap_list.append((np.absolute(s2-s1)))
    print(max_acc, acc)
    
    test_blond_men_loader = DataLoader(dataset=test_blond_men_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    test_blond_women_loader = DataLoader(dataset=test_blond_women_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    test_brunette_men_loader = DataLoader(dataset=test_brunette_men_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    test_brunette_women_loader = DataLoader(dataset=test_brunette_women_dataset,
                     batch_size=bs,
                     shuffle=False,
                     num_workers=4)
    _, s1 = compute_loss_accuracy(model, test_blond_men_loader, max_acc_threshold_men, DEVICE)
    _, s2 = compute_loss_accuracy(model, test_blond_women_loader, max_acc_threshold_women, DEVICE)
    _, s3 = compute_loss_accuracy(model, test_brunette_men_loader, max_acc_threshold_men, DEVICE)
    _, s4 = compute_loss_accuracy(model, test_brunette_women_loader, max_acc_threshold_women, DEVICE)
    acc = (180*s1/19962) + (2480*s2/19962) + (7535*s3/19962) + (9767*s4/19962)
    test_max_accuracy_list.append(acc)
    test_fnr_gap_list.append((np.absolute(s2-s1)))

    if FNR_GAP == 0.01:
        train_loader = DataLoader(dataset=train_dataset,
                      batch_size=bs,
                      shuffle=True,
                      num_workers=4)
        _, acc = compute_loss_accuracy(model, train_loader, 0, DEVICE)
        train_accuracy_equal_threshold_list.append(acc)

        _, s1 = compute_loss_accuracy(model, valid_blond_men_loader, 0, DEVICE)
        _, s2 = compute_loss_accuracy(model, valid_blond_women_loader, 0, DEVICE)
        _, s3 = compute_loss_accuracy(model, valid_brunette_men_loader, 0, DEVICE)
        _, s4 = compute_loss_accuracy(model, valid_brunette_women_loader, 0, DEVICE)
        acc = (182*s1/19867) + (2874*s2/19867) + (8276*s3/19867) + (8535*s4/19867)
        valid_accuracy_equal_threshold_list.append(acc)
        valid_equal_threshold_fnr_gap_list.append((np.absolute(s2-s1)))
        valid_loader = DataLoader(dataset=valid_dataset,
                      batch_size=bs,
                      shuffle=True,
                      num_workers=4)
        print(acc, compute_loss_accuracy(model, valid_loader, 0, DEVICE))

        
        _, s1 = compute_loss_accuracy(model, test_blond_men_loader, 0, DEVICE)
        _, s2 = compute_loss_accuracy(model, test_blond_women_loader, 0, DEVICE)
        _, s3 = compute_loss_accuracy(model, test_brunette_men_loader, 0, DEVICE)
        _, s4 = compute_loss_accuracy(model, test_brunette_women_loader, 0, DEVICE)
        acc = (180*s1/19962) + (2480*s2/19962) + (7535*s3/19962) + (9767*s4/19962)
        test_accuracy_equal_threshold_list.append(acc)
        test_equal_threshold_fnr_gap_list.append((np.absolute(s2-s1)))
        test_loader = DataLoader(dataset=test_dataset,
                      batch_size=bs,
                      shuffle=True,
                      num_workers=4)
        print(acc, compute_loss_accuracy(model, test_loader, 0, DEVICE))


if FNR_GAP == 0.01:    
    with open(PATH+'train_accuracy_equal_threshold_list.txt', 'w') as filehandle:
        json.dump(train_accuracy_equal_threshold_list, filehandle)
    with open(PATH+'valid_accuracy_equal_threshold_list.txt', 'w') as filehandle:
        json.dump(valid_accuracy_equal_threshold_list, filehandle)
    with open(PATH+'test_accuracy_equal_threshold_list.txt', 'w') as filehandle:
        json.dump(test_accuracy_equal_threshold_list, filehandle)
    with open(PATH+'valid_equal_threshold_fnr_gap_list.txt', 'w') as filehandle:
        json.dump(valid_equal_threshold_fnr_gap_list, filehandle)
    with open(PATH+'test_equal_threshold_fnr_gap_list.txt', 'w') as filehandle:
        json.dump(test_equal_threshold_fnr_gap_list, filehandle)
    
if not os.path.exists(PATH+'fnr_gap_%s'%(FNR_GAP)):
    os.makedirs(PATH+'fnr_gap_%s'%(FNR_GAP))

with open(PATH+'fnr_gap_%s/valid_max_accuracy_list.txt'%(FNR_GAP), 'w') as filehandle:
    json.dump(valid_max_accuracy_list, filehandle)
with open(PATH+'fnr_gap_%s/valid_max_accuracy_threshold_list.txt'%(FNR_GAP), 'w') as filehandle:
    json.dump(valid_max_accuracy_threshold_list, filehandle)
with open(PATH+'fnr_gap_%s/test_max_accuracy_list.txt'%(FNR_GAP), 'w') as filehandle:
    json.dump(test_max_accuracy_list, filehandle)
with open(PATH+'fnr_gap_%s/valid_fnr_gap_list.txt'%(FNR_GAP), 'w') as filehandle:
    json.dump(valid_fnr_gap_list, filehandle)
with open(PATH+'fnr_gap_%s/test_fnr_gap_list.txt'%(FNR_GAP), 'w') as filehandle:
    json.dump(test_fnr_gap_list, filehandle)

