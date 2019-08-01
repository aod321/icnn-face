import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
    RandomResizedCrop, HorizontalFlip, LabelsToOneHot, \
    GaussianNoise, RandomAffine
from torchvision import transforms
import numpy as np
from visualize import show_mask, save_mask_result
import cv2
import matplotlib.pyplot as plt
from model_1 import FaceModel, Stage2FaceModel
np.set_printoptions(threshold=np.inf)

class HelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]

        image = io.imread(img_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(11)]
        labels = np.array(labels)
        bg = labels[0] + labels[1] + labels[10]
        labels = np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0)
        sample = {'image': image, 'labels': labels, 'index': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SinglePart(Dataset):

    def __len__(self):
        return len(self.name_list)

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.transform = transform
        self.part_name = None
        self.range = None
        self.label_numbers = 0

    def set_part(self, name, range, label_numbers):
        self.part_name = name
        self.range = range
        self.label_numbers = label_numbers
        return self

    def get_part_name(self):
        return self.part_name

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        part_path = os.path.join(self.root_dir, '%s' % self.part_name, 'images',
                                 img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, '%s' % self.part_name,
                                    'labels', img_name,
                                    img_name + "_lbl%.2d.png" % i)
                       for i in self.range]
        bg_range = set(range(11)).difference(set(self.range))
        bg_path = [os.path.join(self.root_dir, '%s' % self.part_name,
                                'labels', img_name,
                                img_name + "_lbl%.2d.png" % i)
                   for i in bg_range]
        image = np.array(io.imread(part_path))
        labels = np.array([io.imread(labels_path[i]) for i in range(self.label_numbers)])  # [L, 64, 64]
        bg = 255 - np.sum(labels, axis=0, keepdims=True) # [1, 64, 64]
        labels = np.concatenate([bg, labels], axis=0)  # [L + 1, 64, 64]
        labels = np.uint8(labels)
        # labels = {'fg': labels,
        #           'bg': 255 - labels}

        sample = {'image': image, 'labels': labels, 'index': idx}

        if self.transform:
            sample = self.transform(sample)
            img, new_label, idx = sample['image'], sample['labels'], sample['index']
            new_label_fg = torch.sum(new_label[1:new_label.shape[0]], dim=0, keepdim=True)  # 1 x 64 x 64
            new_label[0] = 1 - new_label_fg
            sample = {'image': img, 'labels': new_label, 'index': idx}
        return sample


class SinglepartAugmentation(object):
    def __init__(self, dataset, txt_file, root_dir, resize, set_part=None, with_flip=False):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
        self.randomchoice = None
        self.transforms = None
        self.transforms_list = None
        self.dataset = dataset
        self.set_part = set_part
        self.with_flip = with_flip
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.resize = resize
        self.set_choice()
        self.set_transformers()
        self.set_transforms_list()

    def set_choice(self):
        choice = {
            # random_choice 1:  h_flip, rotaion, h_flip + rotation + scale_translate (random_order)
            self.augmentation_name[1]: [HorizontalFlip(),
                                        RandomRotation(15),
                                        transforms.RandomOrder([HorizontalFlip(),
                                                                RandomRotation(15),
                                                                RandomAffine(degrees=0, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # random_choice 2:  noise, crop, noise + crop + rotation_scale_translate (random_order)
            self.augmentation_name[2]: [GaussianNoise(),
                                        RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                        RandomAffine(degrees=15, translate=(0.01, 0.1), scale=(0.9, 1.1)),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # random_choice 3:  noise + h_flip , noise + rotation ,noise + h_flip + rotation_scale_translate
            self.augmentation_name[3]: [transforms.RandomOrder([GaussianNoise(),
                                                                HorizontalFlip()
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomRotation(15)
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                HorizontalFlip(),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # random_choice 4:  noise + crop , h_flip + crop ,noise + h_flip + crop + rotation_scale_translate
            self.augmentation_name[4]: [transforms.RandomOrder([GaussianNoise(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1))
                                                                ]
                                                               ),
                                        transforms.RandomOrder([HorizontalFlip(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1))
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                HorizontalFlip(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ]
        }
        self.randomchoice = choice

    def set_resize(self, resize):
        self.resize = resize

    def set_transformers(self):
        self.transforms = {
            self.augmentation_name[0]: transforms.Compose([
                ToPILImage(),
                Resize(self.resize),
                LabelsToOneHot(),
                ToTensor()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                ToPILImage(),
                # Choose from tranforms_list randomly
                transforms.RandomChoice(self.randomchoice['choice1']),
                Resize(self.resize),
                LabelsToOneHot(),
                ToTensor()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice2']),
                Resize(self.resize),
                LabelsToOneHot(),
                ToTensor()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice3']),
                Resize(self.resize),
                LabelsToOneHot(),
                ToTensor()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice4']),
                Resize(self.resize),
                LabelsToOneHot(),
                ToTensor()
            ])
        }

    def set_transforms_list(self):
        self.transforms_list = {
            'train':
                self.transforms,
            'val':
                self.transforms['origin']
        }

    def get_dataset(self):
        datasets = {'train': [self.dataset(txt_file=self.txt_file['train'],
                                           root_dir=self.root_dir,
                                           transform=self.transforms_list['train'][r]
                                           )
                              for r in self.augmentation_name],
                    'val': self.dataset(txt_file=self.txt_file['val'],
                                        root_dir=self.root_dir,
                                        transform=self.transforms_list['val']
                                        )
                    }
        if self.set_part:
            datasets['val'].set_part(name=self.set_part[0], range=self.set_part[1], label_numbers=self.set_part[2])
            for r in range(len(self.augmentation_name)):
                datasets['train'][r].set_part(name=self.set_part[0], range=self.set_part[1],
                                              label_numbers=self.set_part[2])

        enhaced_datasets = {'train': ConcatDataset(datasets['train']),
                            'val': datasets['val']
                            }

        return enhaced_datasets


class DoublePartAugmentation(SinglepartAugmentation):
    def set_choice(self):
        choice = {  # rotation , rotation + affine (random_order)
            self.augmentation_name[1]: [
                RandomRotation(15),
                transforms.RandomOrder([
                    RandomRotation(15),
                    RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))
                ]
                )
            ],

            # noise, crop_with_resize, affine, all of them(random_order)
            self.augmentation_name[2]: [GaussianNoise(),
                                        RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                        RandomAffine(degrees=15, translate=(0.01, 0.1), scale=(0.9, 1.1)),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # noise + affine, noise + rotation,  noise + rotation + scale + translate, noise + rotation + scale
            self.augmentation_name[3]: [transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=15, translate=None,
                                                                             scale=None)
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=None)
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomAffine(degrees=15, translate=None,
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # noise + crop_with_resize , affine + crop_with_resize, noise + crop_with_resize + affine
            self.augmentation_name[4]: [transforms.RandomOrder([GaussianNoise(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1))
                                                                ]
                                                               ),
                                        transforms.RandomOrder([RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1)),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1))
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ]
        }
        self.randomchoice = choice

    def set_transformers(self):
        if self.with_flip is True:
            self.transforms = {
                self.augmentation_name[0]: transforms.Compose([
                    ToPILImage(),
                    HorizontalFlip(),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[1]: transforms.Compose([
                    ToPILImage(),
                    HorizontalFlip(),
                    transforms.RandomChoice(self.randomchoice['choice1']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[2]: transforms.Compose([
                    ToPILImage(),
                    HorizontalFlip(),
                    transforms.RandomChoice(self.randomchoice['choice2']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[3]: transforms.Compose([
                    ToPILImage(),
                    HorizontalFlip(),
                    transforms.RandomChoice(self.randomchoice['choice3']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[4]: transforms.Compose([
                    ToPILImage(),
                    HorizontalFlip(),
                    transforms.RandomChoice(self.randomchoice['choice4']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ])
            }
        else:
            self.transforms = {
                self.augmentation_name[0]: transforms.Compose([
                    ToPILImage(),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[1]: transforms.Compose([
                    ToPILImage(),
                    transforms.RandomChoice(self.randomchoice['choice1']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[2]: transforms.Compose([
                    ToPILImage(),
                    transforms.RandomChoice(self.randomchoice['choice2']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[3]: transforms.Compose([
                    ToPILImage(),
                    transforms.RandomChoice(self.randomchoice['choice3']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ]),
                self.augmentation_name[4]: transforms.Compose([
                    ToPILImage(),
                    transforms.RandomChoice(self.randomchoice['choice4']),
                    Resize(self.resize),
                    LabelsToOneHot(),
                    ToTensor()
                ])
            }


class TestStage(object):
    def __init__(self, device, model_class, statefile, dataset_class,
                 txt_file, root_dir, batch_size, is_shuffle=True, num_workers=4):
        self.F1_name_list = ['eyebrows', 'eyes', 'nose', 'u_lip', 'i_mouth', 'l_lip', 'mouth_all']

        self.dataloader = None
        self.predict = None
        self.device = device
        self.model = None
        self.model_class = model_class
        self.statefile = statefile
        self.dataset_class = dataset_class
        self.dataset = None
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.batch_size =batch_size
        self.is_shuffle = is_shuffle
        self.num_workers = num_workers

    def load_model(self):
        model = self.model_class().to(self.device)
        state = torch.load(self.statefile, map_location=self.device)
        state = state['model']
        model.load_state_dict(state)
        self.model = model

    def load_dataset(self):
        pass

    def start_test(self):
        pass

    def get_predict(self, model, image):
        self.predict = torch.softmax(model(image.to(self.device)), 1)

class TestStage1(TestStage):
    def __init__(self, device, model_class, statefile,
                 dataset_class, txt_file, root_dir, batch_size):
        super(TestStage1, self).__init__(device, model_class, statefile,
                                        dataset_class, txt_file, root_dir, batch_size)
        self.F1_name_list = ['eyebrow1','eyebrow2',
                             'eye1', 'eye2',
                             'nose', 'u_lip', 'i_mouth', 'l_lip']
        self.model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']

        self.TP = {x: 0.0
                   for x in self.F1_name_list}
        self.FP = {x: 0.0
                   for x in self.F1_name_list}
        self.TN = {x: 0.0
                   for x in self.F1_name_list}
        self.FN = {x: 0.0
                   for x in self.F1_name_list}
        self.recall = {x: 0.0
                       for x in self.F1_name_list}
        self.precision = {x: 0.0
                          for x in self.F1_name_list}
        self.F1_list = {x: []
                   for x in self.F1_name_list}
        self.F1 = {x: 0.0
                   for x in self.F1_name_list}
        self.load_model()
        self.load_dataset()
    def load_dataset(self):
        self.dataset = self.dataset_class(txt_file='testing.txt',
                                          root_dir=self.root_dir,
                                          transform=transforms.Compose([
                                              ToPILImage(),
                                              Resize((64, 64)),
                                              ToTensor()
                                          ])
                                          )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=self.is_shuffle, num_workers=self.num_workers)

    def start_test(self):
        for i_batch, sample_batched in enumerate(self.dataloader):
            img = sample_batched['image'].to(self.device)
            labels = sample_batched['labels'].to(self.device)
            self.get_predict(self.model, img)
            show_mask(img, self.predict)
        #     self.calc_f1(predict=self.predict, labels=labels)
        # self.output_f1_score()

    def calc_f1(self, predict, labels):
        part_name_list = {1: 'eyebrow1', 2: 'eyebrow2', 3: 'eye1', 4:'eye2',
                          5: 'nose', 6:'u_lip', 7:'i_mouth', 8:'l_lip'}
        pred = predict.argmax(dim=1, keepdim=False).to(self.device)
        ground = labels.argmax(dim=1, keepdim=False).to(self.device)
        for i in range(1, labels.shape[1]):
            self.TP[part_name_list[i]] += ((pred == i) * (ground == i)).sum().tolist()
            self.TN[part_name_list[i]] += ((pred != i) * (ground != i)).sum().tolist()
            self.FP[part_name_list[i]] += ((pred == i) * (ground != i)).sum().tolist()
            self.FN[part_name_list[i]] += ((pred != i) * (ground == i)).sum().tolist()
        for r in self.F1_name_list:
            self.recall[r] = self.TP[r] / (
                    self.TP[r] + self.FP[r])
            self.precision[r] = self.TP[r] / (
                    self.TP[r] + self.FN[r])
            self.F1_list[r].append((2 * self.precision[r] * self.recall[r]) / \
                                   (self.precision[r] + self.recall[r]))

    def output_f1_score(self):
        print("Stage1 F1_scores:")
        for r in self.F1_name_list:
            self.F1[r] = np.array(self.F1_list[r]).mean()
            print("{}:{}\t".format(r,self.F1[r]))

    def save_result(self, x, image, predict):
        binary_list = []
        predic_argm = predict.argmax(dim=1, keepdim=False)
        for i in range(predict.shape[1]):
            binary = (predic_argm == i)
            binary_list.append(binary)
        pred = torch.stack(binary_list, dim=1)
        pred = pred.detach().cpu().numpy()
        sample = {'image': image,
                  'labels': pred}
        masked_image = get_masked_image(sample)
        out = torchvision.utils.make_grid(masked_image)
        imshow(out)




class TestStage2(TestStage):
    def __init__(self, device, model_class, statefile, dataset_class, txt_file,
                 root_dir, batch_size, is_shuffle=True, num_workers=4):
        super(TestStage2, self).__init__(device, model_class, statefile, dataset_class, txt_file,
                                         root_dir, batch_size, is_shuffle, num_workers)
        self.model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']
        self.F1_name_list = ['eyebrows', 'eyes', 'nose', 'u_lip', 'i_mouth', 'l_lip', 'mouth_all']
        self.TP = {x: 0.0
                   for x in self.F1_name_list}
        self.FP = {x: 0.0
                   for x in self.F1_name_list}
        self.TN = {x: 0.0
                   for x in self.F1_name_list}
        self.FN = {x: 0.0
                   for x in self.F1_name_list}
        self.recall = {x: 0.0
                       for x in self.F1_name_list}
        self.precision = {x: 0.0
                          for x in self.F1_name_list}
        self.recall_overall_list = {x: []
                          for x in self.F1_name_list}
        self.precision_overall_list = {x: []
                          for x in self.F1_name_list}
        self.recall_overall = 0.0
        self.precision_overall = 0.0
        self.F1_overall = 0.0
        self.F1_list = {x: []
                   for x in self.F1_name_list}
        self.F1 = {x: 0.0
                   for x in self.F1_name_list}
        self.load_model()
        self.load_dataset()

    def load_model(self):
        model = {x: self.model_class().to(self.device)
                 for x in self.model_name_list}
        for x in ['eyebrows', 'eyes', 'nose']:
            model[x].set_label_channels(2)
        model['mouth'].set_label_channels(4)
        state = {x: torch.load(self.statefile[x], map_location=self.device)
                 for x in self.model_name_list}
        state = {x: state[x]['model']
                 for x in self.model_name_list
                 }
        model['eyes'].load_state_dict(state['eyes'])
        model['eyebrows'].load_state_dict(state['eyebrows'])
        model['nose'].load_state_dict(state['nose'])
        model['mouth'].load_state_dict(state['mouth'])

        self.model = {x: model[x].to(self.device)
                      for x in self.model_name_list}

    def load_dataset(self):
        eye1_dataset = self.dataset_class(txt_file=self.txt_file,
                                          root_dir=self.root_dir,
                                          transform=transforms.Compose([
                                              ToTensor()])
                                          ).set_part(name='eye1',
                                                     range=range(4, 5),
                                                     label_numbers=1)
        eye2_dataset = self.dataset_class(txt_file=self.txt_file,
                                          root_dir=self.root_dir,
                                          transform=transforms.Compose([
                                              ToPILImage(),
                                              HorizontalFlip(),
                                              ToTensor()
                                          ])
                                          ).set_part(name='eye2',
                                                     range=range(5, 6),
                                                     label_numbers=1)
        eyebrow1_dataset = self.dataset_class(txt_file=self.txt_file,
                                              root_dir=self.root_dir,
                                              transform=transforms.Compose([
                                                  ToTensor()])
                                              ).set_part(name='eyebrow1',
                                                         range=range(2, 3),
                                                         label_numbers=1)
        eyebrow2_dataset = self.dataset_class(txt_file=self.txt_file,
                                              root_dir=self.root_dir,
                                              transform=transforms.Compose([
                                                  ToPILImage(),
                                                  HorizontalFlip(),
                                                  ToTensor()
                                              ])
                                              ).set_part(name='eyebrow2',
                                                         range=range(3, 4),
                                                         label_numbers=1)
        nose_dataset = self.dataset_class(txt_file=self.txt_file,
                                          root_dir=self.root_dir,
                                          transform=transforms.Compose([
                                              ToTensor()])
                                          ).set_part(name='nose',
                                                     range=range(6, 7),
                                                     label_numbers=1)
        mouth_datset = self.dataset_class(txt_file=self.txt_file,
                                          root_dir=self.root_dir,
                                          transform=transforms.Compose([
                                              ToTensor()])
                                          ).set_part(name='mouth',
                                                     range=range(7, 10),
                                                     label_numbers=3)
        eyes_dataset = ConcatDataset([eye1_dataset, eye2_dataset])
        eyebrows_dataset = ConcatDataset([eyebrow1_dataset, eyebrow2_dataset])

        self.dataset = {'eyebrows': eyebrows_dataset,
                        'eyes': eyes_dataset,
                        'nose': nose_dataset,
                        'mouth': mouth_datset
                        }
        self.dataloader = {x: DataLoader(self.dataset[x], batch_size=self.batch_size,
                                         shuffle=self.is_shuffle, num_workers=self.num_workers)
                           for x in self.model_name_list
                           }

    def calc_f1(self, x, predict, labels):
        name_list = {1: 'u_lip', 2: 'i_mouth', 3: 'l_lip', 4: 'mouth_all'}
        pred = predict.argmax(dim=1, keepdim=False).to(self.device)
        ground = labels.argmax(dim=1, keepdim=False).to(self.device)
        if x is 'mouth':
            f1_name_list = ['u_lip', 'i_mouth', 'l_lip', 'mouth_all']
            self.TP[name_list[4]] += ((pred > 0) * (ground > 0)).sum().tolist()
            self.TN[name_list[4]] += ((pred == 0) * (ground == 0)).sum().tolist()
            self.FP[name_list[4]] += ((pred > 0) * (ground == 0)).sum().tolist()
            self.FN[name_list[4]] += ((pred == 0) * (ground > 0)).sum().tolist()

            for i in range(3):
                self.TP[name_list[i + 1]] += ((pred == i + 1) * (ground == i + 1)).sum().tolist()
                self.TN[name_list[i + 1]] += ((pred != i + 1) * (ground == i + 1)).sum().tolist()
                self.FP[name_list[i + 1]] += ((pred == i + 1) * (ground != i + 1)).sum().tolist()
                self.FN[name_list[i + 1]] += ((pred != i + 1) * (ground == i + 1)).sum().tolist()

        else:
            f1_name_list = [x]
            self.TP[x] += ((pred == 1) * (ground == 1)).sum().tolist()
            self.TN[x] += ((pred == 0) * (ground == 0)).sum().tolist()
            self.FP[x] += ((pred == 1) * (ground == 0)).sum().tolist()
            self.FN[x] += ((pred == 0) * (ground == 1)).sum().tolist()

        for r in f1_name_list:
            # print("name:{}\tTP:{}\tTN:{}\tFP:{}\tFN:{}".format(r,self.TP[r], self.TN[r],self.FP[r],self.FN[r]))
            self.recall[r] = self.TP[r] / (
                    self.TP[r] + self.FP[r])
            self.precision[r] = self.TP[r] / (
                    self.TP[r] + self.FN[r])
            self.recall_overall_list[r].append(self.recall[r])
            self.precision_overall_list[r].append(self.precision[r])
            self.F1_list[r].append((2 * self.precision[r] * self.recall[r]) /
                                   (self.precision[r] + self.recall[r]))

    def output_f1_score(self):
        print("F1_socres:\n")
        for x in self.F1_name_list:
            self.recall_overall_list[x] = np.array(self.recall_overall_list[x]).mean()
            self.precision_overall_list[x] = np.array(self.precision_overall_list[x]).mean()
            self.F1[x] = np.array(self.F1_list[x]).mean()
            print("{}:{}\t".format(x, self.F1[x]))
        for x in self.F1_name_list:
            self.recall_overall += self.recall_overall_list[x]
            self.precision_overall += self.precision_overall_list[x]
        self.recall_overall /= len(self.F1_name_list)
        self.precision_overall /= len(self.F1_name_list)
        self.F1_overall = (2 * self.precision_overall * self.recall_overall) /\
                          (self.precision_overall + self.recall_overall)
        print("{}:{}\t".format("overall", self.F1_overall))

    def start_test(self):
        for x in self.model_name_list:
            for i_batch, sample_batched in enumerate(self.dataloader[x]):
                image, labels = sample_batched['image'], sample_batched['labels']

                self.get_predict(model=self.model[x],
                                 image=image)
                self.calc_f1(x, self.predict, labels)
                # show_mask(image, self.predict)
                save_mask_result(i_batch, x, image, self.predict)
        self.output_f1_score()
