from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
        'TLtest': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

data_dir = 'flowersTransferLearning'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                                                  data_transforms[x])
                                  for x in ['TLtest']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                                                         shuffle=False, num_workers=4)
                                  for x in ['TLtest']}
use_gpu = torch.cuda.is_available()

# Get a batch of training data
#inputs = next(iter(dataloaders['test.txt']))

# Make a grid from batch
def train_model(model, file):
    since = time.time()
    model.train(False)
    for data in dataloaders['TLtest']:
                inputs,labels = data
                if use_gpu:
                   inputs= Variable(inputs.cuda())
                else:
                   inputs = Variable(inputs)
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                print(preds)
                file.write(str(preds)+'\n')
#####################################################
# Test model
def restore_cnn():
        w_file = open('project2_20459219', 'w')
        cnnTL = torch.load('cnn.pkl')
        if use_gpu:
            cnnTL = cnnTL.cuda()
        train_model(cnnTL, w_file)
        w_file.close()
restore_cnn()
