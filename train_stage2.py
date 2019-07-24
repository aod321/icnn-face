from template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
from model_1 import FaceModel
# from es_model import ICNN
from torch.utils.data import DataLoader
from dataset import HelenDataset, FacialPartDataset, MouthDataset
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, CenterCrop, LabelsToOneHot, Stage2_ToTensor, Stage2_ToPILImage
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


face_transforms_list = {
    'train':
        transforms.Compose([
            Stage2_ToPILImage(),
            # RandomRotation(15),
            # RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
            # CenterCrop((512,512)),
            # Resize((64, 64)),
            LabelsToOneHot(),
            Stage2_ToTensor()
        ]),
    'val':
        transforms.Compose([
            Stage2_ToPILImage(),
            # Resize((64, 64)),
            LabelsToOneHot(),
            Stage2_ToTensor()
        ])
}

mouth_transforms_list =  {
    'train':
        transforms.Compose([
            Stage2_ToPILImage(),
            # RandomRotation(15),
            # RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
            # CenterCrop((512,512)),
            # Resize((80, 80)),
            LabelsToOneHot(),
            Stage2_ToTensor()
        ]),
    'val':
        transforms.Compose([
            Stage2_ToPILImage(),
            # Resize((64, 64)),
            LabelsToOneHot(),
            Stage2_ToTensor()
        ])
}

transforms_list = {'face':face_transforms_list,
                   'mouth':mouth_transforms_list
                   }

Facialpart_dataset = {x: FacialPartDataset(txt_file=txt_file_names[x],
                                           root_dir=root_dir,
                                           transform=transforms_list['face'][x]
                                  )
                      for x in ['train', 'val']}

Mouth_dataset = {x: MouthDataset(txt_file=txt_file_names[x],
                                 root_dir=root_dir,
                                 transform=transforms_list['mouth'][x]
                                  )
                 \


                 for x in ['train', 'val']}

face_dataloaders = {x: DataLoader(Facialpart_dataset[x], batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
                    for x in ['train', 'val']}

mouth_dataloaders = {x: DataLoader(Mouth_dataset[x], batch_size=args.batch_size,
                                   shuffle=True, num_workers=4)
                     for x in ['train', 'val']}

dataloaders = {'face': face_dataloaders,
               'mouth': mouth_dataloaders}

dataset1_sizes = {x: len(Facialpart_dataset[x]) for x in ['train', 'val']}
dataset2_sizes = {x: len(Mouth_dataset[x]) for x in ['train', 'val']}


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


class TrainStage2(TemplateModel):
    def __init__(self, argus=args):
        super(TrainStage2, self).__init__()
        self.label_channels = 6
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = {x: float('Inf')
                           for x in ['face', 'mouth']}

        self.device = torch.device("cuda:%d" % self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = {x: Stage2FaceModel().to(self.device)
                      for x in ['face', 'mouth']}
        self.model['face'].set_label_channels(5)
        self.model['mouth'].set_label_channels(1)

        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = {x:optim.Adam(self.model[x].parameters(), self.args.lr)
                          for x in ['face', 'mouth']}
        self.criterion = {x: nn.BCEWithLogitsLoss()
                          for x in ['face', 'mouth']}
        self.metric = nn.CrossEntropyLoss()

        self.scheduler = {x: optim.lr_scheduler.StepLR(self.optimizer[x], step_size=15, gamma=0.5)
                          for x in ['face', 'mouth']}

        self.train_loader = dataloaders
        self.eval_loader = dataloaders
        # self.mouth_train_loader = mouth_dataloaders['train']
        # self.mouth_eval_loader = mouth_dataloaders['val']

        self.ckpt_dir = {x: "checkpoints_%s" % x
                         for x in ['face', 'mouth']}
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

        self.x = None

    def train(self):
        self.epoch += 1
        self.model[self.x].train()
        iters = 0
        for batch in self.train_loader[self.x]['train']:
                image, labels = batch['image'].to(self.device), batch['labels'].to(self.device)
                self.step += 1
                iters += 1
                self.optimizer[self.x].zero_grad()

                pred = self.model[self.x](image)
                loss = self.criterion[self.x](pred, labels)

                loss.backward()
                self.optimizer[self.x].step()

                if self.step % self.display_freq == 0:
                    self.writer.add_scalar('loss_%s' % x, loss.item(), self.step)
                    print('mode {}\tepoch {}\tstep {}\titers{}/{}\t loss {:.3}'.format(
                             self.x, self.epoch, self.step, len(self.train_loader[self.x]), iters, loss.item())
                          )



    def check_init(self):
        assert self.model
        assert self.optimizer
        assert self.criterion
        assert self.metric
        assert self.train_loader
        assert self.eval_loader
        assert self.device
        assert self.ckpt_dir
        assert self.display_freq
        assert self.scheduler

        for x in ['face', 'mouth']:
            if not osp.exists(self.ckpt_dir[x]):
                os.mkdir(self.ckpt_dir[x])


    def eval(self):
        self.model[self.x].eval()
        error = {x: 0
                 for x in ['face', 'mouth']}
        loss_list = []
        for batch in self.eval_loader[self.x]['val']:
            image, labels = batch['image'].to(self.device), batch['labels'].to(self.device)
            pred = self.model[self.x](image)
            error = self.criterion[self.x](pred, labels)
            loss_list.append(error.item())
        error[self.x] = np.mean(loss_list)

        if error[self.x] < self.best_error[self.x]:
            self.best_error[self.x] = error[self.x]
            self.save_state(osp.join(self.ckpt_dir[self.x], 'best.pth.tar'), False)
        self.save_state(osp.join(self.ckpt_dir[self.x], '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_%s' % self.x, error, self.epoch)
        print('epoch {}\terror {:.3}'.format(self.epoch, error))

        return error[self.x]

    def save_state(self, fname, optim=True):
        state = {}
        if isinstance(self.model[self.x], torch.nn.DataParallel):
            state['model'] = self.model[self.x].module.state_dict()
        else:
            state['model'] = self.model[self.x].state_dict()

        if optim:
            state['optimizer'] = self.optimizer[self.x].state_dict()
        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error[self.x]
        torch.save(state, fname)
        print('save model at {}'.format(fname))


def start_train():
    train = TrainStage2(args)
    for x in ['face', 'mouth']:
        train.x = x
        for epoch in range(args.epochs):
            train.scheduler[x].step()
            train.train()
            if (epoch + 1) % args.eval_per_epoch == 0:
                train.eval()


    print('Done!!!')


start_train()
