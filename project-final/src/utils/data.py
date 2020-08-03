import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image


def load_image(img_path, noise=False, img_size=(256, 256)):
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = np.array(img, dtype=np.float32)
    if not len(img.shape) == 3 and not noise:
        # Grayscale
        img = img[np.newaxis, :, :]
        img = np.tile(img, [3, 1, 1])
    if not noise:
        img -= np.array((104.00699, 116.66877, 122.67892))
        img = img.transpose((2, 0, 1))
    return img


def load_label(label_path, label_size=(256, 256)):
    label = Image.open(label_path)
    label = label.resize(label_size)
    label = np.array(label, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.0
    label = label[np.newaxis, ...]
    return label


class ImageDataTrain(data.Dataset):
    """Image Dataset"""
    def __init__(self, filenames, noise_root, root='./input/dataset/MSRA-B'):
        self.root = root
        self.filenames = filenames
        self.num_img = len(self.filenames)
        self.noise_root = noise_root

    def __getitem__(self, item):
        img_name = self.filenames[item % self.num_img]
        gt_name = self.filenames[item % self.num_img].replace('.jpg', '.png')

        img = load_image(self.root + '/' + img_name)
        label = load_label(self.root + '/' + gt_name)

        # unsup_labels = []
        # for unsup_label_root in self.noise_root:
        #     ngt_name = img_name.replace('.jpg', '_ngt.png').strip()
        #     unsup_labels.append(torch.Tensor(load_image(unsup_label_root + '/' + ngt_name, noise=True)))

        # TODO: Not fix this list of unsupervised labels
        ngt_name = img_name.replace('.jpg', '_ngt.png')
        unsup_labels = [torch.Tensor(load_image('./input/unsup_labels/ft/' + ngt_name, noise=True)),
         torch.Tensor(load_image('./input/unsup_labels/hc/' + ngt_name, noise=True)),
        #  torch.Tensor(load_image('./input/unsup_labels/mbd/' + ngt_name, noise=True)),
         torch.Tensor(load_image('./input/unsup_labels/rc/' + ngt_name, noise=True)),
         torch.Tensor(load_image('./input/unsup_labels/rdb/' + ngt_name, noise=True))
        ]

        img = torch.Tensor(img)
        label = torch.Tensor(label)
        unsup_labels = torch.stack(unsup_labels)

        sample = {'image': img, 'label': label, 'index': item, 'unsup_labels': unsup_labels}
        return sample

    def __len__(self):
        return self.num_img


class ImageDataTest(data.Dataset):
    """Image Dataset"""
    def __init__(self, filenames, root='./input/dataset/MSRA-B'):
        self.root = root
        self.filenames = filenames
        self.num_img = len(self.filenames)

    def __getitem__(self, item):
        img_name = self.filenames[item % self.num_img]
        gt_name = self.filenames[item % self.num_img].replace('.jpg', '.png')

        img = load_image(self.root + '/' + img_name)
        label = load_label(self.root + '/' + gt_name)
        img = torch.Tensor(img)
        label = torch.Tensor(label)
        
        sample = {'image': img, 'label': label}
        return sample

    def __len__(self):
        return self.num_img


def min2d(tensor, dim1=2, dim2=3):
    return torch.min(torch.min(tensor, dim1, keepdim=True)[0], dim2, keepdim=True)[0]


def max2d(tensor, dim1=2, dim2=3):
    return torch.max(torch.max(tensor, dim1, keepdim=True)[0], dim2, keepdim=True)[0]
