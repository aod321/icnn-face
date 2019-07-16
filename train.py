from template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
from model_1 import FaceModel
from es_model import ICNN
from torch.utils.data import DataLoader
from dataset import HelenDataset
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, CenterCrop, LabelsToOneHot
from torchvision import transforms
import argparse

from tensorboardX import SummaryWriter
import tensorboardX as tb

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)


# Initiation For Normalize (calculated by mean_std_calc.py)
mean = [0.369, 0.314, 0.282]
std = [0.282, 0.251, 0.238]

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
            RandomRotation(15),
            RandomResizedCrop((200, 200), scale=(0.9, 1.1)),
            CenterCrop((128,128)),
            Resize((64, 64)),
            # LabelsToOneHot(),
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            # LabelsToOneHot(),
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


class TrainModel(TemplateModel):

    def __init__(self , argus = args):
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
        self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)

        self.train_loader = dataloaders['train']
        self.eval_loader = dataloaders['val']

        self.ckpt_dir = "checkpoints"
        self.display_freq = args.display_freq



        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        x, y = batch['image'].float(), batch['labels'].float()

        x = x.to(self.device)
        y = y.to(self.device)
        pred = self.model(x)
        loss = self.criterion(pred, y.argmax(dim=1, keepdim=False))
        # loss /= self.args.batch_size
        return loss, None


    def eval_error(self):
        xs, ys, preds = [], [], []
        for batch in self.eval_loader:
            x, y = batch['image'], batch['labels']
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)

            xs.append(x.cpu())
            ys.append(y.cpu())
            preds.append(pred.cpu())

        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        preds = torch.cat(preds, dim=0)

        error = self.metric(preds, ys.argmax(dim=1, keepdim=False))

        return error, None


def start_train():
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.scheduler.step()
        train.train()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
