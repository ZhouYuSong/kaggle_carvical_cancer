from __future__ import division, print_function

import glob
import os
import time
import copy
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from train_val_split import train_val_split_bag
from inference_to_predict import inference_to_predict

np.random.seed(369258)

DIR_BASE = '/home/omewangyx/kaggle/carvical_cancer/zhouyusong_code/data/add_seg_bag'
use_gpu = torch.cuda.is_available()
BATCH_SIZE = 128
n_models = 30

model_name_base = 'resnet18_ft_bag_{}'

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(300),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25, kid=0):
    since = time.time()
    
    best_acc = 0.0
    best_loss = 1.0e9
    
    stop_flag = False

    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch + 1, num_epochs), end=' | ')
        # print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                try:
                    optimizer = lr_scheduler(optimizer, epoch)
                except Exception as E:
                    print(E)
                    stop_flag = True
                    break
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for data in train_val_loader[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.data[0] * inputs.size()[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_val_size[phase]
            epoch_acc = running_corrects / train_val_size[phase]

            print('{} Loss: {:.5f} Acc: {:.5f}'.format(
                phase, epoch_loss, epoch_acc), end='   ')
            if phase == 'val':
                print()

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model)
                torch.save(best_model, model_name_base.format(kid))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                
        if stop_flag:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}  Best val logloss: {:.4f}'.format(best_acc, best_loss))

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=20, min_lr=.99e-8):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    
    if lr < min_lr:
        raise Exception('Learning rate {} is smaller than min_lr({})'.format(lr, min_lr))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def start_train(kid, n_epoch):

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=n_epoch, kid=kid)


for i in range(30, n_models+30):
    print()
    print('*'*20, 'Start training model {}/{}'.format(i+1-30, n_models), '*'*20)

    train_val_split_bag(max_sample=0.8)

    train_val_folder = {x: datasets.ImageFolder(os.path.join(DIR_BASE, x), data_transforms[x]) 
                    for x in ['train', 'val']}
    train_val_loader = {x: torch.utils.data.DataLoader(train_val_folder[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=12) 
                        for x in ['train', 'val']}
    train_val_size = {x: len(train_val_folder[x]) for x in ['train', 'val']}
    classes = train_val_folder['train'].classes
    # print('There are {} classes, {} images in train set and {} images in validation set.'
    #       .format(len(classes), train_val_size['train'], train_val_size['val']))

    start_train(kid=i, n_epoch=50)

    inference_to_predict(model_name_base.format(i), BATCH_SIZE)


