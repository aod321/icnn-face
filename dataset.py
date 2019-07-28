import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import  functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, HorizontalFlip, LabelsToOneHot,\
                                GaussianNoise, RandomAffine
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

        img_name = self.name_list[idx,1].strip()
        img_path = os.path.join(self.root_dir,'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir,'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]

        image =  io.imread(img_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(11)]
        labels = np.array(labels)
        bg = labels[0] + labels[1] + labels[10]
        labels = np.concatenate((labels[2:10], [bg.clip(0, 255)]), axis=0)
        # # one hot
        # labels = labels > 0
        # labels = labels.astype(np.float32)

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
        labels_path =[os.path.join(self.root_dir, '%s' % self.part_name,
                                   'labels', img_name,
                                    img_name + "_lbl%.2d.png" % i)
                       for i in self.range]
        bg_range = set(range(11)).difference(set(self.range))
        bg_path = [os.path.join(self.root_dir, '%s' % self.part_name,
                                   'labels', img_name,
                                    img_name + "_lbl%.2d.png" % i)
                       for i in bg_range]
        image = np.array(io.imread(part_path))
        labels = np.array([io.imread(labels_path[i]) for i in range(self.label_numbers)])     # [L, 64, 64]
        bg = np.array([io.imread(bg_path[i]) for i in range(len(bg_range))])                  # [11 - L, 64, 64]
        bg = np.sum(bg, axis=0, keepdims=True)                                                # [1, 64, 64]
        labels = np.concatenate([bg, labels], axis=0)                                         # [L + 1, 64, 64]
        labels = np.uint8(labels)
        # labels = {'fg': labels,
        #           'bg': 255 - labels}

        sample = {'image': image, 'labels': labels, 'index': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample

class SinglepartAugmentation(object):
    def __init__(self, dataset, txt_file, root_dir, resize, set_part=None, with_flip=False):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
        self.randomchoice = None
        self.transforms = None
        self.transforms_list = None
        self.dataset = dataset
        self.set_part = set_part
        self.with_flip= with_flip
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
                datasets['train'][r].set_part(name=self.set_part[0], range=self.set_part[1], label_numbers=self.set_part[2])

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
                  self.augmentation_name[3]:  [transforms.RandomOrder([GaussianNoise(),
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





