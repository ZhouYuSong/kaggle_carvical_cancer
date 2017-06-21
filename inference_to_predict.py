from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import os

def make_pred(model, testloader):
    outputs = None
    predict = None
    
    for data in testloader:
        inputs, _ = data
        inputs = Variable(inputs.cuda())
        
        out = model(inputs)
        _, pred = torch.max(out.data, 1)
        
        if outputs is None:
            outputs = out.data.cpu().numpy()
        else:
            outputs = np.vstack([outputs, out.data.cpu().numpy()])
        
        if predict is None:
            predict = pred.cpu().numpy()
        else:
            predict = np.vstack([predict, pred.cpu().numpy()])
    
    outputs = nn.functional.softmax(Variable(torch.from_numpy(outputs)))
    outputs = outputs.data.numpy()
    return outputs, predict

def to_csv(fname, dfolder, outs):
    imgs = dfolder.imgs
    imgs = [img[0].split('/')[-1] for img in imgs]
    df = pd.DataFrame(outs, index=imgs, columns=['Type_1', 'Type_2', 'Type_3'])
    df.index.name = 'image_name'
    df.sort_index(inplace=True)
    df.head()
    df.to_csv(fname)


def inference_to_predict(model_name,  BATCH_SIZE=64):

    model_ft = torch.load(model_name)
    model_ft = model_ft.cuda()

    DIR_BASE = '/home/omewangyx/kaggle/carvical_cancer/zhouyusong_code/data/test_data'

    test_data_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_folder = datasets.ImageFolder(os.path.join(DIR_BASE, "step_2"), test_data_transforms)
    test_loader = torch.utils.data.DataLoader(test_folder, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    test_size = len(test_folder)

    print('Testing and saving csv file .... {} images in test set.'.format(test_size))

    outputs, predictions = make_pred(model_ft, test_loader)
    to_csv('sub'+model_name+'.csv', test_folder, outputs)



