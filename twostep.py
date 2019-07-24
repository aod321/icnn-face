import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import HelenDataset
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, CenterCrop
from model_1 import FaceModel
from train_stage2 import Stage2FaceModel
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test images readin
# root_dir = "/data1/yinzi/datas"
root_dir = '/home/yinzi/Downloads/datas'
test_dataset = HelenDataset(txt_file='testing.txt',
                            root_dir=root_dir,
                            transform= transforms.Compose([
                                ToPILImage(),
                                Resize((64, 64)),
                                ToTensor()
                                # Normalize(mean=[0.369, 0.314, 0.282],
                                #           std=[0.282, 0.251, 0.238])
                            ]))
unresized_dataset = HelenDataset(txt_file='testing.txt',
                                 root_dir=root_dir,
                                 transform= transforms.Compose([
                                     # ToPILImage(),
                                     ToTensor()
                                     # Normalize(mean=[0.369, 0.314, 0.282],
                                     #           std=[0.282, 0.251, 0.238])
                                               ]))

dataloader = DataLoader(test_dataset, batch_size=4,
                        shuffle=False, num_workers=4)
# Load models

# Stage 1 model
model_1 = FaceModel().to(device)
state = torch.load('model_stage1.pth.tar', map_location=device)
state = state['model']
model_1.load_state_dict(state)

# Stage2 model
model_2 = {x: Stage2FaceModel().to(device)
           for x in ['face', 'mouth']}

model_2['face'].set_label_channels(5)
state = torch.load('model_stage2_face.pth.tar', map_location=device)
state = state['model']
model_2['face'].load_state_dict(state)

model_2['mouth'].set_label_channels(1)
state = torch.load('model_stage2_mouth.pth.tar', map_location=device)
state = state['model']
model_2['mouth'].load_state_dict(state)

"""
Some helper functions

"""


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


def extract_parts(indexs, centroids, orig_dataset):
    orig_images = torch.Tensor([])
    orig_labels = torch.Tensor([])
    box_size = 1024
    res = {}
    offsets = []
    shapes = []
    for i, idx in enumerate(indexs):
        l, h, w = orig_dataset[idx]['labels'].shape
        offset_y, offset_x = (box_size - h) // 2, (box_size - w) // 2
        offsets.append((offset_y, offset_x))
        shapes.append((h, w))

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

    orig = {'images': orig_images, 'labels': orig_labels}

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

    res['non-mouth'] = {'patch_images': patch_images, 'patch_labels': patch_labels}

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

    res['mouth'] = {'patch_images': patch_images, 'patch_labels': patch_labels}

    return res, centroids.long(), orig, np.array(offsets), np.array(shapes)


def prepare_batches(parts):
  batches = {}

  # Non-mouth parts
  patches, labels = parts['non-mouth']['patch_images'], parts['non-mouth']['patch_labels']

  batches['eyebrow'] = {'image': torch.cat( [patches[:,0,:,:,:], patches[:,1,:,:,:].flip(-1) ]),
                        'labels': torch.cat( [bg(labels[:,0,:,:,:], [2]), bg(labels[:,1,:,:,:], [3]).flip(-1) ] ) }

  batches['eye'] = {'image': torch.cat( [patches[:,2,:,:,:], patches[:,3,:,:,:].flip(-1) ]),
                    'labels': torch.cat( [bg(labels[:,2,:,:,:], [4]), bg(labels[:,3,:,:,:], [5]).flip(-1) ] ) }

  batches['nose'] = {'image': patches[:,4,:,:,:],
                     'labels': bg(labels[:,4,:,:,:], [6]) }


  # Mouth parts
  patches, labels = parts['mouth']['patch_images'], parts['mouth']['patch_labels']

  batches['mouth'] = {'image': patches[:,0,:,:,:],
                   'labels': bg(labels[:,0,:,:,:], [7,8,9]) }

  return batches


"""
Two step pipline

"""

#  Stage 1
with torch.no_grad():
    for batch in dataloader:
        images, labels, indexs = batch['image'].to(device), batch['labels'].to(device), batch['index']

        # Calculate locations of facial parts
        pred_labels = F.softmax(model_1(images), 1)
        centroids = calculate_centroids(pred_labels)

        # Extract patches from face from their location given in centroids
        # Get also shift-scaled centroids, offsets and shapes
        parts, centroids, orig, offsets, shapes = extract_parts(indexs, centroids, unresized_dataset)

        # Prepare batches for facial parts
        batches = prepare_batches(parts)

        # Get stage 2 prediction


        # Update F1-measure stat for this batch
        calculate_F1(batches, pred_labels)

        # Rearrange patch results onto original image
        ground_result, pred_result = combine_results(pred_labels, orig, centroids)

        # Save results
        save_results(ground_result, pred_result, indexs, offsets, shapes)
        print("Processed %d images" % args.batch_size)