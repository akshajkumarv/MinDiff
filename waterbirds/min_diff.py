from arguments import args
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=args.device_id

import time

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        
def sample_from_sphere(n, d):
    x = np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    x = (x.T/np.linalg.norm(x, axis=1)).T
    return x

def get_projections(x, W):
    # Skipped normalization because I normalized W instead
    return np.maximum(x @ W, 0)

def get_random_features(train_x, train_waterbirds_waterbkgd_x, train_waterbirds_landbkgd_x, train_landbirds_waterbkgd_x, train_landbirds_landbkgd_x, N):
    if isinstance(train_x, tuple):
        assert len(N)==len(train_x)
    elif N==0:
        return train_x, train_waterbirds_waterbkgd_x, train_waterbirds_landbkgd_x, train_landbirds_waterbkgd_x, train_landbirds_landbkgd_x
    else:
        train_x = (train_x,)
        train_waterbirds_waterbkgd_x = (train_waterbirds_waterbkgd_x,)
        train_waterbirds_landbkgd_x = (train_waterbirds_landbkgd_x,)
        train_landbirds_waterbkgd_x = (train_landbirds_waterbkgd_x,)
        train_landbirds_landbkgd_x = (train_landbirds_landbkgd_x,)
        N = (N,)

    train_x_projected, train_waterbirds_waterbkgd_x_projected, train_waterbirds_landbkgd_x_projected, train_landbirds_waterbkgd_x_projected, train_landbirds_landbkgd_x_projected = [], [], [], [], []
    
    for train_x_component, train_waterbirds_waterbkgd_x_component, train_waterbirds_landbkgd_x_component, train_landbirds_waterbkgd_x_component, train_landbirds_landbkgd_x_component, N_component in zip(train_x, train_waterbirds_waterbkgd_x, train_waterbirds_landbkgd_x, train_landbirds_waterbkgd_x, train_landbirds_landbkgd_x, N):
        if N_component == 0:
            continue
        d = train_x_component.shape[1] # number of original features
        W = sample_from_sphere(d, N_component)
        # Train
        train_x_projected.append(get_projections(train_x_component, W))
        train_waterbirds_waterbkgd_x_projected.append(get_projections(train_waterbirds_waterbkgd_x_component, W))
        train_waterbirds_landbkgd_x_projected.append(get_projections(train_waterbirds_landbkgd_x_component, W))
        train_landbirds_waterbkgd_x_projected.append(get_projections(train_landbirds_waterbkgd_x_component, W))
        train_landbirds_landbkgd_x_projected.append(get_projections(train_landbirds_landbkgd_x_component, W))
        
    return W, np.hstack(train_x_projected), np.hstack(train_waterbirds_waterbkgd_x_projected), np.hstack(train_waterbirds_landbkgd_x_projected), np.hstack(train_landbirds_waterbkgd_x_projected), np.hstack(train_landbirds_landbkgd_x_projected)

def compute_loss_accuracy(net, loader, device):
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

            predicted_labels = probas > 0.5
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    loss = running_loss/num_examples
    accuracy = (correct_pred/num_examples) * 100
    accuracy = accuracy.item()
    
    return loss, accuracy

def kernel_product(a, b, l=0.1):
    a = a.view(-1)
    b = b.view(-1)
    a = torch.t(a).unsqueeze(1)
    b = b.unsqueeze(0)
    K = torch.mean(torch.exp(-((a-b)**2)/(l**2)))   
    return K

def adjust_learning_rate(optimizer, step, LEARNING_RATE):
    lr = LEARNING_RATE * (0.1 ** (step // 10000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


DEVICE = 'cpu'#'cuda:0'
LEARNING_RATE = args.lr
NUM_STEPS = args.steps
RUN = args.run
FLOOD_LEVEL = args.flood_level
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
MIN_DIFF_WEIGHT = args.mindiff_weight
MIN_DIFF = args.mindiff
N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
print(args)

torch.manual_seed(RUN)
torch.cuda.manual_seed_all(RUN)
np.random.seed(RUN)
os.environ['PYTHONHASHSEED'] = str(RUN)


for n in N:

    print('='*15, 'N =', n, '='*15)

    if MIN_DIFF == False:    
        PATH = 'saved_models/batch_size/%s/run_%s/original_model_flooding_%s_weight_decay_%s/'%(BATCH_SIZE, RUN, FLOOD_LEVEL, WEIGHT_DECAY)  
        print(PATH) 
    if MIN_DIFF == True:
        PATH = 'saved_models/batch_size/%s/run_%s/min_diff_model_weight_%s_flooding_%s_weight_decay_%s/'%(BATCH_SIZE, RUN, MIN_DIFF_WEIGHT, FLOOD_LEVEL, WEIGHT_DECAY)
        print(PATH)

    if not os.path.exists(PATH+'width_%s/steps'%(n)):
        os.makedirs(PATH+'width_%s/steps'%(n))
    if not os.path.exists(PATH+'width_%s/epochs'%(n)):
        os.makedirs(PATH+'width_%s/epochs'%(n))

    import pandas as pd
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
    train_waterbirds_waterbkgd_x, train_waterbirds_waterbkgd_a, train_waterbirds_waterbkgd_y = features[train_mask & waterbirds_mask & waterbkgd_mask,:], metadata[train_mask & waterbirds_mask & waterbkgd_mask]['place'].values, metadata[train_mask & waterbirds_mask & waterbkgd_mask]['y'].values
    train_waterbirds_landbkgd_x, train_waterbirds_landbkgd_a, train_waterbirds_landbkgd_y = features[train_mask & waterbirds_mask & landbkgd_mask,:], metadata[train_mask & waterbirds_mask & landbkgd_mask]['place'].values, metadata[train_mask & waterbirds_mask & landbkgd_mask]['y'].values
    train_landbirds_waterbkgd_x, train_landbirds_waterbkgd_a, train_landbirds_waterbkgd_y = features[train_mask & landbirds_mask & waterbkgd_mask,:], metadata[train_mask & landbirds_mask & waterbkgd_mask]['place'].values, metadata[train_mask & landbirds_mask & waterbkgd_mask]['y'].values
    train_landbirds_landbkgd_x, train_landbirds_landbkgd_a, train_landbirds_landbkgd_y = features[train_mask & landbirds_mask & landbkgd_mask,:], metadata[train_mask & landbirds_mask & landbkgd_mask]['place'].values, metadata[train_mask & landbirds_mask & landbkgd_mask]['y'].values
    
    W, proj_train_x, proj_train_waterbirds_waterbkgd_x, proj_train_waterbirds_landbkgd_x, proj_train_landbirds_waterbkgd_x, proj_train_landbirds_landbkgd_x = get_random_features(train_x, train_waterbirds_waterbkgd_x, train_waterbirds_landbkgd_x, train_landbirds_waterbkgd_x, train_landbirds_landbkgd_x, n)

    train_dataset = PreDataset(proj_train_x, train_a, train_y)
    train_waterbirds_waterbkgd_dataset = PreDataset(proj_train_waterbirds_waterbkgd_x, train_waterbirds_waterbkgd_a, train_waterbirds_waterbkgd_y)
    train_waterbirds_landbkgd_dataset = PreDataset(proj_train_waterbirds_landbkgd_x, train_waterbirds_landbkgd_a, train_waterbirds_landbkgd_y)
    train_landbirds_waterbkgd_dataset = PreDataset(proj_train_landbirds_waterbkgd_x, train_landbirds_waterbkgd_a, train_landbirds_waterbkgd_y)
    train_landbirds_landbkgd_dataset = PreDataset(proj_train_landbirds_landbkgd_x, train_landbirds_landbkgd_a, train_landbirds_landbkgd_y)
   
    train_loader = DataLoader(dataset=train_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=True)

    train_waterbirds_waterbkgd_loader = DataLoader(dataset=train_waterbirds_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=2,
                     pin_memory=True)

    train_waterbirds_landbkgd_loader = DataLoader(dataset=train_waterbirds_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=2,
                     pin_memory=True)

    model = LogisticRegression(num_inputs=n)
    model.to(DEVICE)

    cost_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_waterbirds_waterbkgd_iter = iter(train_waterbirds_waterbkgd_loader)
    train_waterbirds_landbkgd_iter = iter(train_waterbirds_landbkgd_loader)

    step = 0
    epoch = 0

    while step < NUM_STEPS:
        for batch_idx, (features, _, targets) in enumerate(train_loader):
    
            model.train()

            adjust_learning_rate(optimizer, step, LEARNING_RATE)
            
            optimizer.zero_grad()

            if MIN_DIFF == True:
                if step % 5000 == 0:
                    print(step)    

                try:
                    train_waterbirds_waterbkgd_features, _, train_waterbirds_waterbkgd_targets = next(train_waterbirds_waterbkgd_iter)
                except StopIteration: 
                    train_waterbirds_waterbkgd_iter = iter(train_waterbirds_waterbkgd_loader)
                    train_waterbirds_waterbkgd_features, _, train_waterbirds_waterbkgd_targets = next(train_waterbirds_waterbkgd_iter)

                try:
                    train_waterbirds_landbkgd_features, _, train_waterbirds_landbkgd_targets = next(train_waterbirds_landbkgd_iter) 
                except StopIteration: 
                    train_waterbirds_landbkgd_iter = iter(train_waterbirds_landbkgd_loader)
                    train_waterbirds_landbkgd_features, _, train_waterbirds_landbkgd_targets = next(train_waterbirds_landbkgd_iter)

                inputs = torch.cat((features.float(), train_waterbirds_waterbkgd_features.float(), train_waterbirds_landbkgd_features.float()))
                inputs = inputs.to(DEVICE)
                targets = targets.unsqueeze(1).float().to(DEVICE)
        
                logits = model(inputs)
                probas = F.sigmoid(logits)

                primary_loss = cost_fn(probas[:targets.size(0)], targets)

                x1x1 = kernel_product(probas[targets.size(0):targets.size(0)+train_waterbirds_waterbkgd_targets.size(0)], probas[targets.size(0):targets.size(0)+train_waterbirds_waterbkgd_targets.size(0)])
                x1x2 = kernel_product(probas[targets.size(0):targets.size(0)+train_waterbirds_waterbkgd_targets.size(0)], probas[targets.size(0)+train_waterbirds_waterbkgd_targets.size(0):])
                x2x2 = kernel_product(probas[targets.size(0)+train_waterbirds_waterbkgd_targets.size(0):], probas[targets.size(0)+train_waterbirds_waterbkgd_targets.size(0):])
                mmd_loss = x1x1 - (2*x1x2) + x2x2
                mmd_loss = mmd_loss.to(DEVICE)    
  
                cost = (primary_loss-FLOOD_LEVEL).abs() + FLOOD_LEVEL + (MIN_DIFF_WEIGHT*mmd_loss)
                
            if MIN_DIFF == False:
                features = features.float().to(DEVICE)
                targets = targets.unsqueeze(1).float().to(DEVICE)
        
                logits = model(features)
                probas = F.sigmoid(logits)
                     
                primary_loss = cost_fn(probas, targets)
                cost = (primary_loss-FLOOD_LEVEL).abs() + FLOOD_LEVEL

            cost.backward()
            optimizer.step()

            step += 1
            if step == (NUM_STEPS+1):
                break    

            if (step == 1) or (step % 500 == 0):
                CHECKPOINT = PATH+'width_%s/steps/step_%s.pt'%(n, step)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, CHECKPOINT)

        epoch += 1
        CHECKPOINT = PATH+'width_%s/epochs/epoch_%s.pt'%(n, epoch)
        torch.save({
            'epoch': epoch,
            'step': step-1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPOINT)
        
    CHECKPOINT = PATH+'width_%s/width_%s.pt'%(n, n)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'W': W,
        }, CHECKPOINT)

