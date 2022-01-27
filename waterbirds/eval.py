from arguments import args
import os

import time

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import h5py
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_inputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_inputs, 1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class PreDataset(Dataset):
    def __init__(self, x, a, y, transform=None):
 
        self.img = x
        self.attri = a
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.img[index]
        if self.transform is not None:
            img = self.transform(img)
        attri = self.attri[index]
        label = self.y[index]
        
        return img, attri, label

    def __len__(self):
        return self.y.shape[0]
        
def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x = np.array(data['inputs'])
    a = np.array(data['attributes'])
    y = np.array(data['targets'])
 
    return x, a, y
        
def get_projections(x, W):
    # Skipped normalization because I normalized W instead
    return np.maximum(x @ W, 0)


def get_random_features(W, train_x, train_waterbkgd_x, train_landbkgd_x, train_waterbirds_waterbkgd_x, train_waterbirds_landbkgd_x, train_landbirds_waterbkgd_x, train_landbirds_landbkgd_x, valid_x, valid_waterbkgd_x, valid_landbkgd_x, valid_waterbirds_waterbkgd_x, valid_waterbirds_landbkgd_x, valid_landbirds_waterbkgd_x, valid_landbirds_landbkgd_x, test_x, test_waterbkgd_x, test_landbkgd_x, test_waterbirds_waterbkgd_x, test_waterbirds_landbkgd_x, test_landbirds_waterbkgd_x, test_landbirds_landbkgd_x, N):
    train_x = (train_x,)
    train_waterbkgd_x = (train_waterbkgd_x,)
    train_landbkgd_x = (train_landbkgd_x,)
    train_waterbirds_waterbkgd_x = (train_waterbirds_waterbkgd_x,)
    train_waterbirds_landbkgd_x = (train_waterbirds_landbkgd_x,)
    train_landbirds_waterbkgd_x = (train_landbirds_waterbkgd_x,)
    train_landbirds_landbkgd_x = (train_landbirds_landbkgd_x,)
    valid_x = (valid_x,)
    valid_waterbkgd_x = (valid_waterbkgd_x,)
    valid_landbkgd_x = (valid_landbkgd_x,)
    valid_waterbirds_waterbkgd_x = (valid_waterbirds_waterbkgd_x,)
    valid_waterbirds_landbkgd_x = (valid_waterbirds_landbkgd_x,)
    valid_landbirds_waterbkgd_x = (valid_landbirds_waterbkgd_x,)
    valid_landbirds_landbkgd_x = (valid_landbirds_landbkgd_x,)
    test_x = (test_x,)
    test_waterbkgd_x = (test_waterbkgd_x,)
    test_landbkgd_x = (test_landbkgd_x,)
    test_waterbirds_waterbkgd_x = (test_waterbirds_waterbkgd_x,)
    test_waterbirds_landbkgd_x = (test_waterbirds_landbkgd_x,)
    test_landbirds_waterbkgd_x = (test_landbirds_waterbkgd_x,)
    test_landbirds_landbkgd_x = (test_landbirds_landbkgd_x,)
    N = (N,)

    train_x_projected, train_waterbkgd_x_projected, train_landbkgd_x_projected, train_waterbirds_waterbkgd_x_projected, train_waterbirds_landbkgd_x_projected, train_landbirds_waterbkgd_x_projected, train_landbirds_landbkgd_x_projected = [], [], [], [], [], [], []
    valid_x_projected, valid_waterbkgd_x_projected, valid_landbkgd_x_projected, valid_waterbirds_waterbkgd_x_projected, valid_waterbirds_landbkgd_x_projected, valid_landbirds_waterbkgd_x_projected, valid_landbirds_landbkgd_x_projected = [], [], [], [], [], [], []
    test_x_projected, test_waterbkgd_x_projected, test_landbkgd_x_projected, test_waterbirds_waterbkgd_x_projected, test_waterbirds_landbkgd_x_projected, test_landbirds_waterbkgd_x_projected, test_landbirds_landbkgd_x_projected = [], [], [], [], [], [], []
    
    for train_x_component, train_waterbkgd_x_component, train_landbkgd_x_component, train_waterbirds_waterbkgd_x_component, train_waterbirds_landbkgd_x_component, train_landbirds_waterbkgd_x_component, train_landbirds_landbkgd_x_component, valid_x_component, valid_waterbkgd_x_component, valid_landbkgd_x_component, valid_waterbirds_waterbkgd_x_component, valid_waterbirds_landbkgd_x_component, valid_landbirds_waterbkgd_x_component, valid_landbirds_landbkgd_x_component, test_x_component, test_waterbkgd_x_component, test_landbkgd_x_component, test_waterbirds_waterbkgd_x_component, test_waterbirds_landbkgd_x_component, test_landbirds_waterbkgd_x_component, test_landbirds_landbkgd_x_component, N_component in zip(train_x, train_waterbkgd_x, train_landbkgd_x, train_waterbirds_waterbkgd_x, train_waterbirds_landbkgd_x, train_landbirds_waterbkgd_x, train_landbirds_landbkgd_x, valid_x, valid_waterbkgd_x, valid_landbkgd_x, valid_waterbirds_waterbkgd_x, valid_waterbirds_landbkgd_x, valid_landbirds_waterbkgd_x, valid_landbirds_landbkgd_x, test_x, test_waterbkgd_x, test_landbkgd_x, test_waterbirds_waterbkgd_x, test_waterbirds_landbkgd_x, test_landbirds_waterbkgd_x, test_landbirds_landbkgd_x, N):
        d = train_x_component.shape[1] # number of original features
        # Train
        train_x_projected.append(get_projections(train_x_component, W))
        train_waterbkgd_x_projected.append(get_projections(train_waterbkgd_x_component, W))
        train_landbkgd_x_projected.append(get_projections(train_landbkgd_x_component, W))
        train_waterbirds_waterbkgd_x_projected.append(get_projections(train_waterbirds_waterbkgd_x_component, W))
        train_waterbirds_landbkgd_x_projected.append(get_projections(train_waterbirds_landbkgd_x_component, W))
        train_landbirds_waterbkgd_x_projected.append(get_projections(train_landbirds_waterbkgd_x_component, W))
        train_landbirds_landbkgd_x_projected.append(get_projections(train_landbirds_landbkgd_x_component, W))
        # Valid
        valid_x_projected.append(get_projections(valid_x_component, W))
        valid_waterbkgd_x_projected.append(get_projections(valid_waterbkgd_x_component, W))
        valid_landbkgd_x_projected.append(get_projections(valid_landbkgd_x_component, W))
        valid_waterbirds_waterbkgd_x_projected.append(get_projections(valid_waterbirds_waterbkgd_x_component, W))
        valid_waterbirds_landbkgd_x_projected.append(get_projections(valid_waterbirds_landbkgd_x_component, W))
        valid_landbirds_waterbkgd_x_projected.append(get_projections(valid_landbirds_waterbkgd_x_component, W))
        valid_landbirds_landbkgd_x_projected.append(get_projections(valid_landbirds_landbkgd_x_component, W))
        # Test
        test_x_projected.append(get_projections(test_x_component, W))
        test_waterbkgd_x_projected.append(get_projections(test_waterbkgd_x_component, W))
        test_landbkgd_x_projected.append(get_projections(test_landbkgd_x_component, W))
        test_waterbirds_waterbkgd_x_projected.append(get_projections(test_waterbirds_waterbkgd_x_component, W))
        test_waterbirds_landbkgd_x_projected.append(get_projections(test_waterbirds_landbkgd_x_component, W))
        test_landbirds_waterbkgd_x_projected.append(get_projections(test_landbirds_waterbkgd_x_component, W))
        test_landbirds_landbkgd_x_projected.append(get_projections(test_landbirds_landbkgd_x_component, W))
        
    return np.hstack(train_x_projected), np.hstack(train_waterbkgd_x_projected), np.hstack(train_landbkgd_x_projected), np.hstack(train_waterbirds_waterbkgd_x_projected), np.hstack(train_waterbirds_landbkgd_x_projected), np.hstack(train_landbirds_waterbkgd_x_projected), np.hstack(train_landbirds_landbkgd_x_projected), np.hstack(valid_x_projected), np.hstack(valid_waterbkgd_x_projected), np.hstack(valid_landbkgd_x_projected), np.hstack(valid_waterbirds_waterbkgd_x_projected), np.hstack(valid_waterbirds_landbkgd_x_projected), np.hstack(valid_landbirds_waterbkgd_x_projected), np.hstack(valid_landbirds_landbkgd_x_projected), np.hstack(test_x_projected), np.hstack(test_waterbkgd_x_projected), np.hstack(test_landbkgd_x_projected), np.hstack(test_waterbirds_waterbkgd_x_projected), np.hstack(test_waterbirds_landbkgd_x_projected), np.hstack(test_landbirds_waterbkgd_x_projected), np.hstack(test_landbirds_landbkgd_x_projected)
    
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

DEVICE = 'cpu'
RUN = args.run
FLOOD_LEVEL = args.flood_level
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
FNR_GAP = args.fnr_gap
MIN_DIFF_WEIGHT = args.mindiff_weight
MIN_DIFF = args.mindiff
N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
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

    print('='*15, 'N =', n, '='*15)

    model = LogisticRegression(num_inputs=n)
    load_path = PATH + 'width_%s/width_%s.pt'%(n, n)
    print('Loading Model Checkpoint: ', load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    W = checkpoint['W']
    model.to(DEVICE)
    model.eval()

    features = np.load('extracted_features.npy')
    metadata = pd.read_csv('./waterbird_complete95_forest2water2/metadata.csv')
    waterbirds_mask = metadata['y']==1
    landbirds_mask = metadata['y']==0
    waterbkgd_mask = metadata['place']==1
    landbkgd_mask = metadata['place']==0
    # Train
    train_mask = metadata['split']==0
    train_y = metadata[train_mask]['y'].values
    train_a = metadata[train_mask]['place'].values
    train_x = features[train_mask,:]
    train_waterbkgd_x, train_waterbkgd_a, train_waterbkgd_y = features[train_mask & waterbkgd_mask,:], metadata[train_mask & waterbkgd_mask]['place'].values, metadata[train_mask & waterbkgd_mask]['y'].values
    train_landbkgd_x, train_landbkgd_a, train_landbkgd_y = features[train_mask & landbkgd_mask,:], metadata[train_mask & landbkgd_mask]['place'].values, metadata[train_mask & landbkgd_mask]['y'].values
    train_waterbirds_waterbkgd_x, train_waterbirds_waterbkgd_a, train_waterbirds_waterbkgd_y = features[train_mask & waterbirds_mask & waterbkgd_mask,:], metadata[train_mask & waterbirds_mask & waterbkgd_mask]['place'].values, metadata[train_mask & waterbirds_mask & waterbkgd_mask]['y'].values
    train_waterbirds_landbkgd_x, train_waterbirds_landbkgd_a, train_waterbirds_landbkgd_y = features[train_mask & waterbirds_mask & landbkgd_mask,:], metadata[train_mask & waterbirds_mask & landbkgd_mask]['place'].values, metadata[train_mask & waterbirds_mask & landbkgd_mask]['y'].values
    train_landbirds_waterbkgd_x, train_landbirds_waterbkgd_a, train_landbirds_waterbkgd_y = features[train_mask & landbirds_mask & waterbkgd_mask,:], metadata[train_mask & landbirds_mask & waterbkgd_mask]['place'].values, metadata[train_mask & landbirds_mask & waterbkgd_mask]['y'].values
    train_landbirds_landbkgd_x, train_landbirds_landbkgd_a, train_landbirds_landbkgd_y = features[train_mask & landbirds_mask & landbkgd_mask,:], metadata[train_mask & landbirds_mask & landbkgd_mask]['place'].values, metadata[train_mask & landbirds_mask & landbkgd_mask]['y'].values
    # Valid
    valid_mask = metadata['split']==1
    valid_y = metadata[valid_mask]['y'].values
    valid_a = metadata[valid_mask]['place'].values
    valid_x = features[valid_mask,:]
    valid_waterbkgd_x, valid_waterbkgd_a, valid_waterbkgd_y = features[valid_mask & waterbkgd_mask,:], metadata[valid_mask & waterbkgd_mask]['place'].values, metadata[valid_mask & waterbkgd_mask]['y'].values
    valid_landbkgd_x, valid_landbkgd_a, valid_landbkgd_y = features[valid_mask & landbkgd_mask,:], metadata[valid_mask & landbkgd_mask]['place'].values, metadata[valid_mask & landbkgd_mask]['y'].values
    valid_waterbirds_waterbkgd_x, valid_waterbirds_waterbkgd_a, valid_waterbirds_waterbkgd_y = features[valid_mask & waterbirds_mask & waterbkgd_mask,:], metadata[valid_mask & waterbirds_mask & waterbkgd_mask]['place'].values, metadata[valid_mask & waterbirds_mask & waterbkgd_mask]['y'].values
    valid_waterbirds_landbkgd_x, valid_waterbirds_landbkgd_a, valid_waterbirds_landbkgd_y = features[valid_mask & waterbirds_mask & landbkgd_mask,:], metadata[valid_mask & waterbirds_mask & landbkgd_mask]['place'].values, metadata[valid_mask & waterbirds_mask & landbkgd_mask]['y'].values
    valid_landbirds_waterbkgd_x, valid_landbirds_waterbkgd_a, valid_landbirds_waterbkgd_y = features[valid_mask & landbirds_mask & waterbkgd_mask,:], metadata[valid_mask & landbirds_mask & waterbkgd_mask]['place'].values, metadata[valid_mask & landbirds_mask & waterbkgd_mask]['y'].values
    valid_landbirds_landbkgd_x, valid_landbirds_landbkgd_a, valid_landbirds_landbkgd_y = features[valid_mask & landbirds_mask & landbkgd_mask,:], metadata[valid_mask & landbirds_mask & landbkgd_mask]['place'].values, metadata[valid_mask & landbirds_mask & landbkgd_mask]['y'].values
    # Test
    test_mask = metadata['split']==2
    test_y = metadata[test_mask]['y'].values
    test_a = metadata[test_mask]['place'].values
    test_x = features[test_mask,:]
    test_waterbkgd_x, test_waterbkgd_a, test_waterbkgd_y = features[test_mask & waterbkgd_mask,:], metadata[test_mask & waterbkgd_mask]['place'].values, metadata[test_mask & waterbkgd_mask]['y'].values
    test_landbkgd_x, test_landbkgd_a, test_landbkgd_y = features[test_mask & landbkgd_mask,:], metadata[test_mask & landbkgd_mask]['place'].values, metadata[test_mask & landbkgd_mask]['y'].values
    test_waterbirds_waterbkgd_x, test_waterbirds_waterbkgd_a, test_waterbirds_waterbkgd_y = features[test_mask & waterbirds_mask & waterbkgd_mask,:], metadata[test_mask & waterbirds_mask & waterbkgd_mask]['place'].values, metadata[test_mask & waterbirds_mask & waterbkgd_mask]['y'].values
    test_waterbirds_landbkgd_x, test_waterbirds_landbkgd_a, test_waterbirds_landbkgd_y = features[test_mask & waterbirds_mask & landbkgd_mask,:], metadata[test_mask & waterbirds_mask & landbkgd_mask]['place'].values, metadata[test_mask & waterbirds_mask & landbkgd_mask]['y'].values
    test_landbirds_waterbkgd_x, test_landbirds_waterbkgd_a, test_landbirds_waterbkgd_y = features[test_mask & landbirds_mask & waterbkgd_mask,:], metadata[test_mask & landbirds_mask & waterbkgd_mask]['place'].values, metadata[test_mask & landbirds_mask & waterbkgd_mask]['y'].values
    test_landbirds_landbkgd_x, test_landbirds_landbkgd_a, test_landbirds_landbkgd_y = features[test_mask & landbirds_mask & landbkgd_mask,:], metadata[test_mask & landbirds_mask & landbkgd_mask]['place'].values, metadata[test_mask & landbirds_mask & landbkgd_mask]['y'].values
    
    proj_train_x, proj_train_waterbkgd_x, proj_train_landbkgd_x, proj_train_waterbirds_waterbkgd_x, proj_train_waterbirds_landbkgd_x, proj_train_landbirds_waterbkgd_x, proj_train_landbirds_landbkgd_x, proj_valid_x, proj_valid_waterbkgd_x, proj_valid_landbkgd_x, proj_valid_waterbirds_waterbkgd_x, proj_valid_waterbirds_landbkgd_x, proj_valid_landbirds_waterbkgd_x, proj_valid_landbirds_landbkgd_x, proj_test_x, proj_test_waterbkgd_x, proj_test_landbkgd_x, proj_test_waterbirds_waterbkgd_x, proj_test_waterbirds_landbkgd_x, proj_test_landbirds_waterbkgd_x, proj_test_landbirds_landbkgd_x = get_random_features(W, train_x, train_waterbkgd_x, train_landbkgd_x, train_waterbirds_waterbkgd_x, train_waterbirds_landbkgd_x, train_landbirds_waterbkgd_x, train_landbirds_landbkgd_x, valid_x, valid_waterbkgd_x, valid_landbkgd_x, valid_waterbirds_waterbkgd_x, valid_waterbirds_landbkgd_x, valid_landbirds_waterbkgd_x, valid_landbirds_landbkgd_x, test_x, test_waterbkgd_x, test_landbkgd_x, test_waterbirds_waterbkgd_x, test_waterbirds_landbkgd_x, test_landbirds_waterbkgd_x, test_landbirds_landbkgd_x, n)

    train_dataset = PreDataset(proj_train_x, train_a, train_y)
    train_waterbkgd_dataset = PreDataset(proj_train_waterbkgd_x, train_waterbkgd_a, train_waterbkgd_y)
    train_landbkgd_dataset = PreDataset(proj_train_landbkgd_x, train_landbkgd_a, train_landbkgd_y)
    train_waterbirds_waterbkgd_dataset = PreDataset(proj_train_waterbirds_waterbkgd_x, train_waterbirds_waterbkgd_a, train_waterbirds_waterbkgd_y)
    train_waterbirds_landbkgd_dataset = PreDataset(proj_train_waterbirds_landbkgd_x, train_waterbirds_landbkgd_a, train_waterbirds_landbkgd_y)
    train_landbirds_waterbkgd_dataset = PreDataset(proj_train_landbirds_waterbkgd_x, train_landbirds_waterbkgd_a, train_landbirds_waterbkgd_y)
    train_landbirds_landbkgd_dataset = PreDataset(proj_train_landbirds_landbkgd_x, train_landbirds_landbkgd_a, train_landbirds_landbkgd_y)
    valid_dataset = PreDataset(proj_valid_x, valid_a, valid_y)
    valid_waterbkgd_dataset = PreDataset(proj_valid_waterbkgd_x, valid_waterbkgd_a, valid_waterbkgd_y)
    valid_landbkgd_dataset = PreDataset(proj_valid_landbkgd_x, valid_landbkgd_a, valid_landbkgd_y)
    valid_waterbirds_waterbkgd_dataset = PreDataset(proj_valid_waterbirds_waterbkgd_x, valid_waterbirds_waterbkgd_a, valid_waterbirds_waterbkgd_y)
    valid_waterbirds_landbkgd_dataset = PreDataset(proj_valid_waterbirds_landbkgd_x, valid_waterbirds_landbkgd_a, valid_waterbirds_landbkgd_y)
    valid_landbirds_waterbkgd_dataset = PreDataset(proj_valid_landbirds_waterbkgd_x, valid_landbirds_waterbkgd_a, valid_landbirds_waterbkgd_y)
    valid_landbirds_landbkgd_dataset = PreDataset(proj_valid_landbirds_landbkgd_x, valid_landbirds_landbkgd_a, valid_landbirds_landbkgd_y)
    test_dataset = PreDataset(proj_test_x, test_a, test_y)
    test_waterbkgd_dataset = PreDataset(proj_test_waterbkgd_x, test_waterbkgd_a, test_waterbkgd_y)
    test_landbkgd_dataset = PreDataset(proj_test_landbkgd_x, test_landbkgd_a, test_landbkgd_y)
    test_waterbirds_waterbkgd_dataset = PreDataset(proj_test_waterbirds_waterbkgd_x, test_waterbirds_waterbkgd_a, test_waterbirds_waterbkgd_y)
    test_waterbirds_landbkgd_dataset = PreDataset(proj_test_waterbirds_landbkgd_x, test_waterbirds_landbkgd_a, test_waterbirds_landbkgd_y)
    test_landbirds_waterbkgd_dataset = PreDataset(proj_test_landbirds_waterbkgd_x, test_landbirds_waterbkgd_a, test_landbirds_waterbkgd_y)
    test_landbirds_landbkgd_dataset = PreDataset(proj_test_landbirds_landbkgd_x, test_landbirds_landbkgd_a, test_landbirds_landbkgd_y)


    valid_waterbkgd_loader = DataLoader(dataset=valid_waterbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    valid_landbkgd_loader = DataLoader(dataset=valid_landbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    valid_waterbirds_waterbkgd_loader = DataLoader(dataset=valid_waterbirds_waterbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    valid_waterbirds_landbkgd_loader = DataLoader(dataset=valid_waterbirds_landbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    valid_landbirds_waterbkgd_loader = DataLoader(dataset=valid_landbirds_waterbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    valid_landbirds_landbkgd_loader = DataLoader(dataset=valid_landbirds_landbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)                 
    test_waterbirds_waterbkgd_loader = DataLoader(dataset=test_waterbirds_waterbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    test_waterbirds_landbkgd_loader = DataLoader(dataset=test_waterbirds_landbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    test_landbirds_waterbkgd_loader = DataLoader(dataset=test_landbirds_waterbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)
    test_landbirds_landbkgd_loader = DataLoader(dataset=test_landbirds_landbkgd_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=2)

    threshold, tnr_waterbkgd, fnr_waterbkgd, tpr_waterbkgd, fpr_waterbkgd = compute_confusion_matrix(model, valid_waterbkgd_loader, DEVICE)
    _, tnr_landbkgd, fnr_landbkgd, tpr_landbkgd, fpr_landbkgd = compute_confusion_matrix(model, valid_landbkgd_loader, DEVICE)

    count_waterbkgd = 0
    max_acc = 0
    max_acc_threshold_waterbkgd = 0
    max_acc_threshold_landbkgd = 0
    for i in fnr_waterbkgd:
        count_landbkgd = 0
        for j in fnr_landbkgd:            
            if np.absolute(i-j) <= FNR_GAP:
                accuracy = (1057*tpr_waterbkgd[count_waterbkgd]/4795) + (56*tpr_landbkgd[count_landbkgd]/4795) + (184*tnr_waterbkgd[count_waterbkgd]/4795) + (3498*tnr_landbkgd[count_landbkgd]/4795)
                if accuracy >= max_acc:
                    max_acc = accuracy                
                    max_acc_threshold_waterbkgd = threshold[count_waterbkgd]
                    max_acc_threshold_landbkgd = threshold[count_landbkgd]
            count_landbkgd += 1
        count_waterbkgd += 1
    
    if max_acc == 0:
        valid_max_accuracy_list.append(np.nan)
        valid_max_accuracy_threshold_list.append((np.nan, np.nan))
        valid_fnr_gap_list.append(np.nan)
        test_max_accuracy_list.append(np.nan)
        test_fnr_gap_list.append(np.nan) 
        
    if max_acc != 0:
        _, s1 = compute_loss_accuracy(model, valid_waterbirds_waterbkgd_loader, max_acc_threshold_waterbkgd, DEVICE)
        _, s2 = compute_loss_accuracy(model, valid_waterbirds_landbkgd_loader, max_acc_threshold_landbkgd, DEVICE)
        _, s3 = compute_loss_accuracy(model, valid_landbirds_waterbkgd_loader, max_acc_threshold_waterbkgd, DEVICE)
        _, s4 = compute_loss_accuracy(model, valid_landbirds_landbkgd_loader, max_acc_threshold_landbkgd, DEVICE)
        acc = (1057*s1/4795) + (56*s2/4795) + (184*s3/4795) + (3498*s4/4795)
        valid_max_accuracy_list.append(acc)
        valid_max_accuracy_threshold_list.append((max_acc_threshold_waterbkgd, max_acc_threshold_landbkgd))
        valid_fnr_gap_list.append((np.absolute(s2-s1)))
        
        _, s1 = compute_loss_accuracy(model, test_waterbirds_waterbkgd_loader, max_acc_threshold_waterbkgd, DEVICE)
        _, s2 = compute_loss_accuracy(model, test_waterbirds_landbkgd_loader, max_acc_threshold_landbkgd, DEVICE)
        _, s3 = compute_loss_accuracy(model, test_landbirds_waterbkgd_loader, max_acc_threshold_waterbkgd, DEVICE)
        _, s4 = compute_loss_accuracy(model, test_landbirds_landbkgd_loader, max_acc_threshold_landbkgd, DEVICE)
        acc = (1057*s1/4795) + (56*s2/4795) + (184*s3/4795) + (3498*s4/4795)
        test_max_accuracy_list.append(acc)
        test_fnr_gap_list.append((np.absolute(s2-s1)))
        
    if FNR_GAP == 0.01:
        train_loader = DataLoader(dataset=train_dataset,
                      batch_size=1024,
                      shuffle=True,
                      num_workers=2)
        _, acc = compute_loss_accuracy(model, train_loader, 0, DEVICE)
        train_accuracy_equal_threshold_list.append(acc)


        _, s1 = compute_loss_accuracy(model, valid_waterbirds_waterbkgd_loader, 0, DEVICE)
        _, s2 = compute_loss_accuracy(model, valid_waterbirds_landbkgd_loader, 0, DEVICE)
        _, s3 = compute_loss_accuracy(model, valid_landbirds_waterbkgd_loader, 0, DEVICE)
        _, s4 = compute_loss_accuracy(model, valid_landbirds_landbkgd_loader, 0, DEVICE)
        acc = (1057*s1/4795) + (56*s2/4795) + (184*s3/4795) + (3498*s4/4795)
        valid_accuracy_equal_threshold_list.append(acc)
        valid_equal_threshold_fnr_gap_list.append((np.absolute(s2-s1)))
    
        _, s1 = compute_loss_accuracy(model, test_waterbirds_waterbkgd_loader, 0, DEVICE)
        _, s2 = compute_loss_accuracy(model, test_waterbirds_landbkgd_loader, 0, DEVICE)
        _, s3 = compute_loss_accuracy(model, test_landbirds_waterbkgd_loader, 0, DEVICE)
        _, s4 = compute_loss_accuracy(model, test_landbirds_landbkgd_loader, 0, DEVICE)
        acc = (1057*s1/4795) + (56*s2/4795) + (184*s3/4795) + (3498*s4/4795)
        test_accuracy_equal_threshold_list.append(acc)
        test_equal_threshold_fnr_gap_list.append((np.absolute(s2-s1)))

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

