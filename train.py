from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms
from dataset import HelenDataset

from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, CenterCrop
import time
import copy
from distutils.version import LooseVersion
import numpy as np

from model import FaceModel

from visualize import tensor_imshow
import argparse


# Argument Parser Part
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size to use during training")
parser.add_argument("--df", default=10, type=float, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs to train")
args = parser.parse_args()
print(args)

# Initiation Part
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initiation For Normalize (calculated by mean_std_calc.py)
mean = [0.369, 0.314, 0.282]
std = [0.282, 0.251, 0.238]

# model initiation
model = FaceModel()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
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
            Resize((64, 64)),
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor(),
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
    best_loss = 99999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phase = 'train'
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training moddataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}e
            else:
                model.eval()  # Set model to evaluate mode

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
                    loss = 0.0
                    for r in range(9):
                        loss += criterion(outputs[:, r, :, :], labels[:, r, :, :])

                        # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if i % args.df == 0:
                    time_now = time.time() - since
                    # for j in range(9):
                    # tensor_imshow(outputs[i][j])
                    print('{}: Epoch:{}/{} Iterate:{} Loss: {:.4f}'.format(
                        phase, epoch, num_epochs - 1, i, loss))

                    # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Epoch Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Start Train
trained = train_model(model, criterion, optimizer, scheduler, num_epochs=args.epochs)

torch.save(trained, 'trained_net.pkl')
# torch.save(trained.state_dict(), 'trained_net_params.pkl')


