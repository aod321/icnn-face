import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from dataset import HelenDataset, SinglePart, SinglepartAugmentation, DoublePartAugmentation
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, HorizontalFlip, LabelsToOneHot,\
                                GaussianNoise, RandomAffine
import numpy as np
import matplotlib.pyplot as plt

root_dir = '/home/yinzi/Downloads/datas'

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}
transforms_all = [
    RandomAffine(degrees=15, translate=(0.01, 0.1), scale=(0.9, 1.1)),       # translate + scale + rotation
    HorizontalFlip(),                                                        # hflip
    RandomRotation(15),                                                      # rotation
    GaussianNoise(),                                                         # noise
    RandomResizedCrop((64, 64), scale=(0.9, 1.1)),                           # crop, scale and resize
]

# Stage 1 augmentation
stage1_augmentation = SinglepartAugmentation(dataset=HelenDataset,
                                                txt_file=txt_file_names,
                                                root_dir=root_dir,
                                                resize=(64, 64)
                                                )
enhaced_stage1_datasets = stage1_augmentation.get_dataset()
stage1_dataloaders = {x: DataLoader(enhaced_stage1_datasets[x], batch_size=16,
                             shuffle=True, num_workers=4)
               for x in ['train', 'val']}

stage1_dataset_sizes = {x: len(enhaced_stage1_datasets[x]) for x in ['train', 'val']}
nose_augmentation = SinglepartAugmentation(dataset=SinglePart,
                                           txt_file=txt_file_names,
                                           root_dir=root_dir,
                                           resize=(64, 64),
                                           set_part=['nose', range(6, 7), 1]
                                           )

# Stage 2 Augmentation

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

eyes_dataset = {x: ConcatDataset([eye1_dataset[x], eye2_dataset[x]])
                      for x in ['train', 'val']}
eyebrows_dataset = {x: ConcatDataset([eyebrow1_dataset[x], eyebrow2_dataset[x]])
                      for x in ['train', 'val']}

#   Test Stage1 augmentation

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
batch = next(iter(stage1_dataloaders['train']))
# for batch in dataloaders['train']:
inputs = batch['image']
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)


