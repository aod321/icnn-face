from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from dataset import HelenDataset

from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, LabelsToOneHot
import time
import os
import copy
import numpy as np

from model import FaceModel

import argparse


# Argument Parser Part
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",default=10,type=int,help="Batch size to use during training")
parser.add_argument("--lr",default=0.01,type=float,help="Learning rate for optimizer")
parser.add_argument("--epochs",default=10,type=int,help="Number of epochs to train")
args = parser.parse_args()
print(args)

# Initiation Part
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initiation For Normalize (calculated by mean_std_calc.py)
mean = [0.369, 0.314, 0.282]
std = [0.282, 0.251, 0.238]

# model initiation
model = FaceModel()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
criterion = criterion.to(device)

# Dataset Read_in Part
# root_dir = "/data1/yinzi/datas"
root_dir = '/home/yinzi/Downloads/datas'

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            RandomRotation(15),
            RandomResizedCrop((255, 255), scale=(0.9, 1.1)),
            Resize((64, 64)),
            ToTensor(),
            LabelsToOneHot(),
            Normalize(mean=mean,
                      std=std)
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor(),
            LabelsToOneHot(),
            Normalize(mean=mean,
                      std=std)
        ])
}

image_datasets = {x: HelenDataset(txt_file=txt_file_names[x],
                                  root_dir=root_dir,
                                  transform=transforms_list[x]
                                  )
                  for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


# Training part


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training moddataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}e
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, batch in enumerate(dataloaders[phase]):
                inputs, labels = batch['image'].to(device), batch['labels'].to(device)
                inputs, labels = inputs.float(), labels.float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs,  labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




# Start Train


train_model(model, criterion, optimizer, scheduler, num_epochs=25)


