from utils.template import TemplateModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.dataset import HelenDataset
from datasets.Helen_transform import Resize, ToPILImage, ToTensor
from torchvision import transforms
import argparse
import numpy as np
from skimage import io
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Dataset and Dataloader
# Dataset Read_in Part
root_dir = "./datas/helen/"
parts_root_dir = "./datas/facial_parts/"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt",
    'test': "testing.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor()
        ]),
    'test':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor()
        ]),
    'orig':
        transforms.Compose([
            ToTensor()
        ])
}

# DataLoader
Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir,
                           transform=transforms_list[x]
                           )
           for x in ['train', 'val', 'test']
           }

dataloader = {x: DataLoader(Dataset[x], batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']
              }

train_loader = dataloader['train']

valid_loader = dataloader['val']

test_loader = dataloader['test']

unresized_train = HelenDataset(txt_file='exemplars.txt',
                               root_dir=root_dir,
                               transform=ToTensor())
unresized_valid = HelenDataset(txt_file='tuning.txt',
                               root_dir=root_dir,
                               transform=ToTensor())
unresized_test = HelenDataset(txt_file='testing.txt',
                              root_dir=root_dir,
                              transform=ToTensor())


def calculate_centroids(tensor):
    tensor = tensor.float() + 1e-10
    n, l, h, w = tensor.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = tensor.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    center_x = tensor.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    return torch.cat([center_y, center_x], 2)


def save_patches(image, labels, dataset, indexs):
    n, p, c, w, h = image.shape
    _, _, l, _, _ = labels.shape
    name_list = dataset.name_list

    image = np.uint8(image.to('cpu').numpy().transpose([0, 1, 3, 4, 2]))
    labels = np.uint8(labels.to('cpu').numpy())

    for i in range(n):
        # Save each patch in an image
        if p == 1:
            # mouth
            image_name = os.path.join(root_dir, dic[5], 'images', name_list[indexs[i], 1].strip() + '.jpg')
            io.imsave(image_name, image[i][0], quality=100)

            os.makedirs(os.path.join(root_dir, dic[5], 'labels', name_list[indexs[i], 1].strip()), exist_ok=True)
            label_name = os.path.join(root_dir, dic[5], 'labels', name_list[indexs[i], 1].strip(),
                                      name_list[indexs[i], 1].strip() + '_lbl%.2d.png')
            for k in range(l):
                io.imsave(label_name % k, labels[i][0][k], check_contrast=False)

        else:
            # non-mouth
            for pp in range(p):
                image_name = os.path.join(root_dir, dic[pp], 'images', name_list[indexs[i], 1].strip() + '.jpg')
                io.imsave(image_name, image[i][pp], quality=100)

                os.makedirs(os.path.join(root_dir, dic[pp], 'labels', name_list[indexs[i], 1].strip()), exist_ok=True)
                label_name = os.path.join(root_dir, dic[pp], 'labels', name_list[indexs[i], 1].strip(),
                                          name_list[indexs[i], 1].strip() + '_lbl%.2d.png')
                for k in range(l):
                    io.imsave(label_name % k, labels[i][pp][k], check_contrast=False)


def extract_parts(loader, orig_dataset):
    box_size = 1024
    with torch.no_grad():
        for batch in loader:
            labels, indexs = batch['labels'].to(device), batch['index']
            labels = F.normalize(labels.to(device), 1)
            centroids = calculate_centroids(labels)

            orig_images = torch.Tensor([])
            orig_labels = torch.Tensor([])

            for i, idx in enumerate(indexs):
                l, h, w = orig_dataset[idx]['labels'].shape
                offset_y, offset_x = (box_size - h) // 2, (box_size - w) // 2

                image = torch.zeros(3, box_size, box_size)
                labels = torch.zeros(l, box_size, box_size)
                image[:, offset_y:offset_y + h, offset_x:offset_x + w] = orig_dataset[idx]['image']
                labels[:, offset_y:offset_y + h, offset_x:offset_x + w] = orig_dataset[idx]['labels']

                orig_images = torch.cat([orig_images, image])
                orig_labels = torch.cat([orig_labels, labels])

                # Scale and shift centroids
                centroids[i] = centroids[i] * torch.Tensor([h / 64., w / 64.]).view(1, 2).to(device) \
                               + torch.Tensor([offset_y, offset_x]).view(1, 2).to(device)

            orig_images = orig_images.to(device).view(len(indexs), 3, box_size, box_size)
            orig_labels = orig_labels.to(device).view(len(indexs), l, box_size, box_size)

            #################
            # Non-Mouth parts
            index = centroids.index_select(1, torch.tensor(range(5)).to(device)).long()
            n_parts = index.shape[-2]

            # Construct repeated image of n x p x c x h x w
            repeated_images = orig_images.unsqueeze(1).repeat_interleave(n_parts, dim=1)
            repeated_labels = orig_labels.unsqueeze(1).repeat_interleave(n_parts, dim=1)

            # Calculate index of patches of the form n x p x 64 x 64 corresponding to each facial part
            # After this index_x/y will be n x p x 64 x 64
            index_y = index[:, :, 0].unsqueeze(-1) + torch.from_numpy(np.arange(-32, 32)).view(1, 1, 64).to(device)
            index_y = index_y.unsqueeze(-1).repeat_interleave(box_size, dim=-1)

            index_x = index[:, :, 1].unsqueeze(-1) + torch.from_numpy(np.arange(-32, 32)).view(1, 1, 64).to(device)
            index_x = index_x.unsqueeze(-2).repeat_interleave(64, dim=-2)

            # Get patch images (n x p x c x h x w)
            patch_images = torch.gather(repeated_images, -2, index_y.unsqueeze(2).repeat_interleave(3, dim=2))
            patch_images = torch.gather(patch_images, -1, index_x.unsqueeze(2).repeat_interleave(3, dim=2))

            # Get patch labels (n x p x l x h x w)
            patch_labels = torch.gather(repeated_labels, -2, index_y.unsqueeze(2).repeat_interleave(l, dim=2))
            patch_labels = torch.gather(patch_labels, -1, index_x.unsqueeze(2).repeat_interleave(l, dim=2))

            # Save patches
            save_patches(patch_images, patch_labels, orig_dataset, indexs)

            ##################
            # Mouth part
            index = centroids.index_select(1, torch.tensor(range(5, 8)).to(device)).mean(dim=1, keepdim=True).long()

            # Construct repeated image of n x 1 x c x h x w
            repeated_images = orig_images.unsqueeze(1)
            repeated_labels = orig_labels.unsqueeze(1)

            # Calculate index of mouth patches of the form n x 1 x 80 x 80 corresponding mouth part
            # After this index_x/y will be n x 1 x 80 x 80
            index_y = index[:, :, 0].unsqueeze(-1) + torch.from_numpy(np.arange(-40, 40)).view(1, 1, 80).to(device)
            index_y = index_y.unsqueeze(-1).repeat_interleave(box_size, dim=-1)

            index_x = index[:, :, 1].unsqueeze(-1) + torch.from_numpy(np.arange(-40, 40)).view(1, 1, 80).to(device)
            index_x = index_x.unsqueeze(-2).repeat_interleave(80, dim=-2)

            # Get patch images (n x 1 x c x 80 x 80)
            patch_images = torch.gather(repeated_images, -2, index_y.unsqueeze(2).repeat_interleave(3, dim=2))
            patch_images = torch.gather(patch_images, -1, index_x.unsqueeze(2).repeat_interleave(3, dim=2))

            # Get patch labels (n x 1 x l x 80 x 80)
            patch_labels = torch.gather(repeated_labels, -2, index_y.unsqueeze(2).repeat_interleave(l, dim=2))
            patch_labels = torch.gather(patch_labels, -1, index_x.unsqueeze(2).repeat_interleave(l, dim=2))

            # Save patches
            save_patches(patch_images, patch_labels, orig_dataset, indexs)
            print("processed %d images" % len(indexs))


if __name__ == '__main__':
    save_dir = './datas/facial_parts'
    dic = {0: 'eyebrow1', 1: 'eyebrow2', 2: 'eye1', 3: 'eye2', 4: 'nose', 5: 'mouth'}

    os.makedirs(save_dir, exist_ok=True)

    list_training = unresized_train.name_list
    list_tuning = unresized_valid.name_list
    list_testing = unresized_test.name_list

    # Create directories fill in files in each
    for k in dic:
        # Copy examplars.txt, tuning.txt and testing.txt
        os.makedirs(os.path.join(save_dir, dic[k]), exist_ok=True)
        os.makedirs(os.path.join(save_dir, dic[k], 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, dic[k], 'labels'), exist_ok=True)

        np.savetxt(os.path.join(save_dir, dic[k], 'exemplars.txt'), list_training, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(save_dir, dic[k], 'tuning.txt'), list_tuning, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(save_dir, dic[k], 'testing.txt'), list_testing, fmt='%s', delimiter=',')

    # Extract and save facial parts in batchs
    extract_parts(train_loader, unresized_train)
    extract_parts(valid_loader, unresized_valid)
    extract_parts(test_loader, unresized_test)
