from utils.template import TemplateModel, F1Accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from models.model_1 import FaceModel, Stage2FaceModel
# from es_model import ICNN
from torch.utils.data import DataLoader, ConcatDataset
from datasets.dataset import HelenDataset, SinglePart, SinglepartAugmentation, DoublePartAugmentation
from datasets.Helen_transform import Resize, ToPILImage, ToTensor, Normalize, HorizontalFlip, \
                                RandomResizedCrop, CenterCrop, LabelsToOneHot
from torchvision import transforms
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboardX as tb
import os.path as osp
import os
from tqdm import tqdm
import uuid as uid
from prefetch_generator import BackgroundGenerator

uuid = str(uid.uuid1())[0:8]
print(uuid)

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=9, type=int, help="Choose which GPU")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.0025, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset Read_in Part
root_dir = "/data1/yinzi/facial_parts"
# root_dir = '/home/yinzi/data/facial_parts'
txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

transforms_list = {
    'train':
        transforms.Compose([
            LabelsToOneHot(),
            ToTensor(),
            Normalize()
        ]),
    'val':
        transforms.Compose([
            LabelsToOneHot(),
            ToTensor(),
            Normalize()
        ])
}


"""     
    Data Augmentation
"""

nose_augmentation = SinglepartAugmentation(dataset=SinglePart,
                                           txt_file=txt_file_names,
                                           root_dir=root_dir,
                                           resize=(64, 64),
                                           set_part=['nose', range(6, 7), 1]
                                           )

mouth_augmentation = SinglepartAugmentation(dataset=SinglePart,
                                            txt_file=txt_file_names,
                                            root_dir=root_dir,
                                            resize=(80, 80),
                                            set_part=['mouth', range(7, 10), 3]
                                            )
eye1_augmentation = DoublePartAugmentation(dataset=SinglePart,
                                           txt_file=txt_file_names,
                                           root_dir=root_dir,
                                           resize=(64, 64),
                                           set_part=['eye1', range(4, 5), 1]
                                           )
eye2_augmentation = DoublePartAugmentation(dataset=SinglePart,
                                           txt_file=txt_file_names,
                                           root_dir=root_dir,
                                           resize=(64, 64),
                                           set_part=['eye2', range(5, 6), 1],
                                           with_flip=True
                                           )
eyebrow1_augmentation = DoublePartAugmentation(dataset=SinglePart,
                                               txt_file=txt_file_names,
                                               root_dir=root_dir,
                                               resize=(64, 64),
                                               set_part=['eyebrow1', range(2, 3), 1]
                                               )
eyebrow2_augmentation = DoublePartAugmentation(dataset=SinglePart,
                                               txt_file=txt_file_names,
                                               root_dir=root_dir,
                                               resize=(64, 64),
                                               set_part=['eyebrow2', range(3, 4), 1],
                                               with_flip=True
                                               )
nose_dataset = nose_augmentation.get_dataset()
mouth_dataset = mouth_augmentation.get_dataset()
eye1_dataset = eye1_augmentation.get_dataset()
eye2_dataset = eye2_augmentation.get_dataset()
eyebrow1_dataset = eyebrow1_augmentation.get_dataset()
eyebrow2_dataset = eyebrow2_augmentation.get_dataset()


"""     
_______________________________________________________________
"""
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
Writer = SummaryWriter('log')

class StageTwoTrain(TemplateModel):
    def train_loss(self, batch):
        x, y = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)
        pred = self.model(x)
        # loss = self.criterion(pred, y)
        loss = self.criterion(pred, y.argmax(dim=1, keepdim=False))

        return loss, None

    def eval_error(self):
        loss_list = []
        for batch in tqdm(self.eval_loader):
            x, y = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)
            pred = self.model(x)
            error = self.criterion(pred, y.argmax(dim=1, keepdim=False))

            loss_list.append(error)

        return np.mean(loss_list), None

    def eval_accu(self):
        hist_list = []
        loss_list = []
        for batch in self.eval_loader:
            image, y = batch['image'].to(self.device), batch['labels'].to(self.device)
            pred = self.model(image)
            gt = y.argmax(dim=1, keepdim=False)
            pred_arg = torch.softmax(pred, dim=1).argmax(dim=1, keepdim=False)
            loss_list.append(self.criterion(pred,  gt).item())
            # pred_arg Shape(N, 256, 256)
            hist = self.fast_histogram(pred_arg.cpu().numpy(), gt.cpu().numpy(),
                                       self.label_channels, self.label_channels)
            hist_list.append(hist)

        mean_error = np.mean(loss_list)
        hist_sum = np.sum(np.stack(hist_list, axis=0), axis=0)
        A = hist_sum[1, :].sum()
        B = hist_sum[:, 1].sum()
        inter_select = hist_sum[1, 1]
        F1 = 2 * inter_select / (A + B)
        return F1, mean_error

    def fast_histogram(self, a, b, na, nb):
        '''
        fast histogram calculation
        ---
        * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
        '''
        assert a.shape == b.shape
        assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
        # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
        hist = np.bincount(
            nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
            minlength=na * nb).reshape(na, nb)
        assert np.sum(hist) == a.size
        return hist

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()

            loss, others = self.train_loss(batch)

            loss.backward()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_%s' % self.mode, loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
                if self.train_logger:
                    self.train_logger(self.writer, others)

    def eval(self):
        self.model.eval()
        accu, mean_error = self.eval_accu()

        if accu > self.best_accu:
            self.best_accu = accu
            self.save_state(osp.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(osp.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar(f'accu_val_{self.mode}_{uuid}', accu, self.epoch)
        print('epoch {}\t mean_error {:.3}\t accu {:.3}\tbest_accu {:.3}'.format(self.epoch, mean_error,
                                                                                 accu, self.best_accu))
        if self.eval_logger:
            self.eval_logger(self.writer, None)

class EyebrowTrain(StageTwoTrain):
    def __init__(self, argus=args):
        super(EyebrowTrain, self).__init__()
        self.label_channels = 2
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus
        self.mode = 'eyebrow'
        # ============== neccessary ===============
        self.writer = Writer
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel()
        self.model.set_label_channels(self.label_channels)
        self.model = self.model.to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.metric = nn.CrossEntropyLoss()
        self.metric = F1Accuracy()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['eyebrows']['train']
        self.eval_loader = data_lodaers['eyebrows']['val']

        self.ckpt_dir = f"checkpoints_eyebrow_{uuid}"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


class EyeTrain(StageTwoTrain):
    def __init__(self, argus=args):
        super(EyeTrain, self).__init__()
        self.label_channels = 2
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus
        self.mode = 'eye'

        # ============== neccessary ===============
        self.writer = Writer
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel()
        self.model.set_label_channels(self.label_channels)
        self.model = self.model.to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.metric = nn.CrossEntropyLoss()
        self.metric = F1Accuracy()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['eyes']['train']
        self.eval_loader = data_lodaers['eyes']['val']

        self.ckpt_dir = f"checkpoints_eyes_{uuid}"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


class NoseTrain(StageTwoTrain):
    def __init__(self, argus=args):
        super(NoseTrain, self).__init__()
        self.label_channels = 2
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus
        self.mode = 'nose'
        # ============== neccessary ===============
        self.writer = Writer
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel()
        self.model.set_label_channels(self.label_channels)
        self.model = self.model.to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.metric = nn.CrossEntropyLoss()
        self.metric = F1Accuracy()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['nose']['train']
        self.eval_loader = data_lodaers['nose']['val']

        self.ckpt_dir = f"checkpoints_nose_{uuid}"
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


class MouthTrain(StageTwoTrain):
    def __init__(self, argus=args):
        super(MouthTrain, self).__init__()
        self.label_channels = 4
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus
        self.mode = 'mouth'
        # ============== neccessary ===============
        self.writer = Writer
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d"%self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2FaceModel()
        self.model.set_label_channels(self.label_channels)
        self.model = self.model.to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.metric = nn.CrossEntropyLoss()
        self.metric = F1Accuracy()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = data_lodaers['mouth']['train']
        self.eval_loader = data_lodaers['mouth']['val']

        self.ckpt_dir = f"checkpoints_mouth_{uuid}"
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
        print("Train %s patch Now" % x)
        for epoch in tqdm(range(args.epochs)):
            train[x].train()
            train[x].scheduler.step()
            if (epoch + 1) % args.eval_per_epoch == 0:
                train[x].eval()

    print('Done!!!')


start_train()
