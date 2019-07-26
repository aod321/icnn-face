from template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
from model_1 import FaceModel
# from es_model import ICNN
from torch.utils.data import DataLoader, ConcatDataset
from dataset import HelenDataset, SinglePart
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, CenterCrop, LabelsToOneHot
from torchvision import transforms
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboardX as tb
import os.path as osp
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=0, type=int, help="eval_per_epoch ")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset Read_in Part
# root_dir = "/data1/yinzi/facial_parts"
root_dir = '/home/yinzi/data/facial_parts'
txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            LabelsToOneHot(),
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            LabelsToOneHot(),
            ToTensor()
        ])
}

# _____________________________________
eyebrow1_dataset = {x: SinglePart(txt_file=txt_file_names[x],
                                           root_dir=root_dir,
                                           transform=transforms_list[x]
                                  ).set_part(name='eyebrow1',range=range(2,3),label_numbers=1)
                      for x in ['train', 'val']}
eyebrow2_dataset = {x: SinglePart(txt_file=txt_file_names[x],
                                           root_dir=root_dir,
                                           transform=transforms_list[x]
                                  ).set_part(name='eyebrow2',range=range(3,4),label_numbers=1)
                      for x in ['train', 'val']}
eye1_dataset = {x: SinglePart(txt_file=txt_file_names[x],
                              root_dir=root_dir,
                              transform=transforms_list[x]
                              ).set_part(name='eye1', range=range(4, 5), label_numbers=1)
                      for x in ['train', 'val']}
eye2_dataset = {x: SinglePart(txt_file=txt_file_names[x],
                                           root_dir=root_dir,
                                           transform=transforms_list[x]
                                  ).set_part(name='eye2',range=range(5,6),label_numbers=1)
                      for x in ['train', 'val']}
nose_dataset = {x: SinglePart(txt_file=txt_file_names[x],
                                           root_dir = root_dir,
                                           transform = transforms_list[x]
                                  ).set_part(name='nose', range=range(6,7),label_numbers=1)
                      for x in ['train', 'val']}
mouth_dataset = {x: SinglePart(txt_file=txt_file_names[x],
                                           root_dir = root_dir,
                                           transform = transforms_list[x]
                                  ).set_part(name='mouth', range=range(7,10),label_numbers=3)
                      for x in ['train', 'val']}
# _____________________________________

eyes_dataset = {x: ConcatDataset([eye1_dataset[x], eye2_dataset[x]])
                      for x in ['train', 'val']}
eyebrows_dataset = {x: ConcatDataset([eyebrow1_dataset[x], eyebrow2_dataset[x]])
                      for x in ['train', 'val']}
# _____________________________________

eyes_train_loader = {x: DataLoader(eyes_dataset[x], batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
eyebrows_train_loader = {x: DataLoader(eyebrows_dataset[x], batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
                    for x in ['train', 'val']}

nose_train_loader = {x: DataLoader(nose_dataset[x], batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
                    for x in ['train', 'val']}

mouth_train_loader = {x: DataLoader(mouth_dataset[x], batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
                    for x in ['train', 'val']}

model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']
data_lodaers = {model_name_list[0]: eyebrows_train_loader,
                 model_name_list[1]: eyes_train_loader,
                 model_name_list[2]: nose_train_loader,
                 model_name_list[3]: mouth_train_loader
                 }


class Stage2FaceModel(FaceModel):

    def set_label_channels(self, lable_channel_size):
        self.lable_channel_size = lable_channel_size
        # last conv layer1     input channels:8     output channels:2L+8
        self.last_conv1 = nn.Conv2d(in_channels=self.first_channels_size[0],
                                    out_channels=2 * self.lable_channel_size + 8,
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=self.kernel_size // 2)
        self.last_con1_bnm = nn.BatchNorm2d(2 * self.lable_channel_size + 8)

        # last conv layer2     input channels:2L+8  output channels:L channels
        self.last_conv2 = nn.Conv2d(in_channels=2 * self.lable_channel_size + 8,
                                    out_channels=self.lable_channel_size,
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=self.kernel_size // 2)
        self.last_con2_bnm = nn.BatchNorm2d(self.lable_channel_size)

        # last conv layer3     input channels:L  output channels:L channels
        self.last_conv3 = nn.Conv2d(in_channels=self.lable_channel_size,
                                    out_channels=self.lable_channel_size,
                                    kernel_size=self.last_kernel_size,
                                    stride=1,
                                    padding=self.last_kernel_size // 2)
        self.last_conv3_bnm = nn.BatchNorm2d(self.lable_channel_size)


class EyebrowTrain(TemplateModel):
    def __init__(self, argus=args):
        super(EyebrowTrain, self).__init__()
        self.label_channels = 2
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel().to(self.device)
        self.model.set_label_channels(self.label_channels)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['eyebrows']['train']
        self.eval_loader = data_lodaers['eyebrows']['val']

        self.ckpt_dir = "checkpoints_eyebrow"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


class EyeTrain(TemplateModel):
    def __init__(self, argus=args):
        super(EyeTrain, self).__init__()
        self.label_channels = 2
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel().to(self.device)
        self.model.set_label_channels(self.label_channels)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['eyes']['train']
        self.eval_loader = data_lodaers['eyes']['val']

        self.ckpt_dir = "checkpoints_eyes"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


class NoseTrain(TemplateModel):
    def __init__(self, argus=args):
        super(NoseTrain, self).__init__()
        self.label_channels = 2
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel().to(self.device)
        self.model.set_label_channels(self.label_channels)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['nose']['train']
        self.eval_loader = data_lodaers['nose']['val']

        self.ckpt_dir = "checkpoints_nose"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


class MouthTrain(TemplateModel):
    def __init__(self, argus=args):
        super(MouthTrain, self).__init__()
        self.label_channels = 6
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel().to(self.device)
        self.model.set_label_channels(self.label_channels)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['mouth']['train']
        self.eval_loader = data_lodaers['mouth']['val']

        self.ckpt_dir = "checkpoints_mouth"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


def start_train():
    train = {model_name_list[0]: EyebrowTrain(args),
             model_name_list[1]: EyeTrain(args),
             model_name_list[2]: NoseTrain(args),
             model_name_list[3]: MouthTrain(args)
    }
    for x in model_name_list:
        for epoch in range(args.epochs):
            # train[x].scheduler.step()
            train[x].train()
            if (epoch + 1) % args.eval_per_epoch == 0:
                train[x].eval()

    print('Done!!!')


start_train()
