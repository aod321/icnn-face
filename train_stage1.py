from utils.template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.model_1 import FaceModel
# from es_model import ICNN
from torch.utils.data import DataLoader
from datasets.dataset import HelenDataset
from datasets.Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
    RandomResizedCrop, CenterCrop, LabelsToOneHot
from torchvision import transforms
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboardX as tb
from datasets.dataset import Stage1Augmentation

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"
# root_dir = '/home/yinzi/Downloads/datas'

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            # RandomRotation(15),
            # RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
            # CenterCrop((512,512)),
            Resize((64, 64)),
            LabelsToOneHot(),
            ToTensor()
            # Normalize()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            LabelsToOneHot(),
            ToTensor()
            # Normalize()
        ])
}
# Stage 1 augmentation
stage1_augmentation = Stage1Augmentation(dataset=HelenDataset,
                                         txt_file=txt_file_names,
                                         root_dir=root_dir,
                                         resize=(64, 64)
                                         )
enhaced_stage1_datasets = stage1_augmentation.get_dataset()
stage1_dataloaders = {x: DataLoader(enhaced_stage1_datasets[x], batch_size=args.batch_size,
                                    shuffle=True, num_workers=4)
                      for x in ['train', 'val']}

stage1_dataset_sizes = {x: len(enhaced_stage1_datasets[x]) for x in ['train', 'val']}


class TrainModel(TemplateModel):

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.label_channels = 9
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = FaceModel().to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = stage1_dataloaders['train']
        self.eval_loader = stage1_dataloaders['val']

        self.ckpt_dir = "checkpoints"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        x, y = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)

        pred = self.model(x)
        # loss = self.criterion(pred, y)
        loss = self.criterion(pred, y.argmax(dim=1, keepdim=False))
        # loss /= self.args.batch_size
        return loss, None

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            x, y = batch['image'].to(self.device), batch['labels'].to(self.device)
            pred = self.model(x)
            error = self.metric(pred, y.argmax(dim=1, keepdim=False))
            # error = self.criterion(pred, y)

            loss_list.append(error.item())

        return np.mean(loss_list), None


def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.scheduler.step()
        train.train()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
