import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np


class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """
    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']

        resized_image = TF.resize(image, self.size, self.interpolation)

        resized_labels = [TF.resize(labels[r], self.size, self.interpolation)
                          for r in range(len(labels))
                          ]

        sample = {'image': resized_image,
                  'labels': resized_labels
                  }

        return sample


class ToPILImage(object):
    """Convert a  ``numpy.ndarray`` to ``PIL Image``

    """
    def __call__(self, sample):
        """
                Args:
                    dict of sample (numpy.ndarray): Image and Labels to be converted.

                Returns:
                    dict of sample(PIL,List of PIL): Converted image and Labels.
        """
        image, labels = sample['image'], sample['labels']

        image = TF.to_pil_image(image)

        labels = [TF.to_pil_image(labels[i])
                  for i in range(labels.shape[0])]

        return {'image': image,
                'labels': labels
                }


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:
                    Tensor: Converted image.
        """
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        # Don't need to swap label because
        # label image: 9 X H X W

        labels = [TF.to_tensor(labels[r])
                  for r in range(len(labels))
                  ]

        labels = torch.cat(labels)

        return {'image': TF.to_tensor(image),
                'labels': labels
                }


class Normalize(transforms.Normalize):
    """Normalize Tensors.

        Override the __call__ of transforms.Normalize

    """

    def __call__(self, sample):
        """
        Args:
            sample (dict of Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensors of sample: Normalized Tensor sample. Only the images need to be normalized.
        """

        image_tensor, labels_tensor = sample['image'], sample['labels']

        sample = {'image': TF.normalize(image_tensor, self.mean, self.std, self.inplace),
                  'labels': labels_tensor
                  }

        return sample


class RandomRotation(transforms.RandomRotation):
    """Rotate the image by angle.

        Override the __call__ of transforms.RandomRotation

    """

    def __call__(self, sample):
        """
            sample (dict of PIL Image and label): Image to be rotated.

        Returns:
            Rotated sample: dict of Rotated image.
        """

        angle = self.get_params(self.degrees)

        img, labels = sample['image'], sample['labels']

        rotated_img = TF.rotate(img, angle, self.resample, self.expand, self.center)
        rotated_labels = [TF.rotate(labels[r], angle, self.resample, self.expand, self.center)
                          for r in range(len(labels))
                          ]

        sample ={'image': rotated_img,
                 'labels': rotated_labels
                }

        return sample


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop the given PIL Image to random size and aspect ratio.

        Override the __call__ of transforms.RandomResizedCrop
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        img, labels = sample['image'], sample['labels']

        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        croped_img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        croped_labels = [TF.resized_crop(labels[r], i, j, h, w, self.size, self.interpolation)
                         for r in range(len(labels))
                         ]

        sample = {'image': croped_img,
                  'labels': croped_labels
                  }

        return sample


class LabelsToOneHot(object):

    """Convert Labels to one hot

    """

    def __call__(self, sample):
        """
             Args:
                 sample:{'image':PIL Image,'labels':labels to be converted}

             Returns:
                 sample:{'image':Not change ,'labels': one hot Tensors for labels}

        """
        img, labels = sample['image'], sample['labels']

        # Convert label to one-hot
        one_hot = labels.numpy()

        for j in range(one_hot.shape[0]):
            one_hot[j, :, :] = np.where(one_hot[j, :, :] != 0,
                                          1,
                                          one_hot[j, :, :]
                                       )

        one_hot = torch.from_numpy(one_hot)
        sample = {'image': img,
                  'labels': one_hot
                  }

        return sample
