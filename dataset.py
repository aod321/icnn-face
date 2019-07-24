import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import  functional as TF


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
        self.name_list = np.loadtxt(os.path.join(root_dir,txt_file),dtype="str",delimiter=',')
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

        sample = {'image': image, 'labels': labels}

        if self.transform:

            sample = self.transform(sample)

        return sample


class FacialPartDataset(HelenDataset):
    """""
     Output:
            Image 5xHxW
            Label 5xHxW
    """""
    def __getitem__(self, idx):
        name = self.name_list[idx, 1].strip()
        dir_list = {2: os.path.join(self.root_dir, 'eyebrow1'),
                    3: os.path.join(self.root_dir, 'eyebrow2'),
                    4: os.path.join(self.root_dir, 'eye1'),
                    5: os.path.join(self.root_dir, 'eye2'),
                    6: os.path.join(self.root_dir, 'nose')
                    }
        eye_path = [os.path.join(dir_list[i], 'images',
                                name + '.jpg') for i in [4, 5]]  # eyes1 eyes2

        eyebrow_path = [os.path.join(dir_list[i], 'images',
                                name + '.jpg') for i in [2, 3] ]

        nose_path = os.path.join(dir_list[6], 'images',
                                name + '.jpg')

        labels_path =[os.path.join(dir_list[i], 'labels',
                                    name,
                                    name + "_lbl%.2d.png") % i
                       for i in range(2, 7)]

        eyes_image = np.array([np.array(io.imread(eye_path[i])) for i in range(2)])
        eyebrows_image =np.array([np.array(io.imread(eyebrow_path[i])) for i in range(2)])
        nose_image = np.array([np.array(io.imread(nose_path))])
        image = np.concatenate((eyebrows_image, eyes_image, nose_image), axis=0)
        labels = np.array([np.array(io.imread(labels_path[i])) for i in range(5)])

        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MouthDataset(HelenDataset):

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        mouth_path = os.path.join(self.root_dir,'mouth', 'images',
                                img_name + '.jpg')
        labels_path =[os.path.join(self.root_dir,'mouth',
                                   'labels', img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(7, 10)]

        image = io.imread(mouth_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(4)]
        labels = np.array(labels)
        mouth = labels[0] + labels[1] + labels[2]
        labels = np.concatenate((labels[2:10], [mouth.clip(0, 255)]), axis=0)
        # # one hot
        # labels = labels > 0
        # labels = labels.astype(np.float32)

        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample