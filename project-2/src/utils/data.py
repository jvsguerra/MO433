from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
from utils.GaussianBlur import GaussianBlur
import matplotlib.pyplot as plt
import torch
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import gdown
import os

transform_dict = {
    # RandomResizedCrop
    'rsc' : transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor()
    ]),
    'rsc_rhf' : transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]),
    'rsc_cj' : transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ]),
    'rsc_gs': transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
        ]),
    'rsc_gb': transforms.Compose([
        transforms.RandomResizedCrop(32),
        GaussianBlur(kernel=int(3)),
        transforms.ToTensor()
        ]),
    # HorizontalFlip
    'rhf' : transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]),
    'rhf_rsc' : transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor()
    ]),
    'rhf_cj' : transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ]),
    'rhf_gs' : transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]),
    'rhf_gb' : transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        GaussianBlur(kernel=int(3)),
        transforms.ToTensor()
    ]),
    # RandomApply ColorJitter
    'cj': transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ]),
    'cj_rsc': transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor()
    ]),
    'cj_rhf': transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]),
    'cj_gs': transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]),
    'cj_gb': transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        GaussianBlur(kernel=int(3)),
        transforms.ToTensor()
    ]),
    # Random Gray Scale
    'gs' : transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]),
    'gs_rsc' : transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor()
    ]),
    'gs_rhf' : transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]),
    'gs_cj' : transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ]),
    'gs_gb' : transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel=int(3)),
        transforms.ToTensor()
    ]),
    # GaussianBlur
    'gb' : transforms.Compose([
        GaussianBlur(kernel=int(3)),
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]),
    'gb_rsc' : transforms.Compose([
        GaussianBlur(kernel=int(3)),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor()
    ]),
    'gb_rhf' : transforms.Compose([
        GaussianBlur(kernel=int(3)),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]),
    'gb_cj' : transforms.Compose([
        GaussianBlur(kernel=int(3)),
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ]),
    'gb_gs' : transforms.Compose([
        GaussianBlur(kernel=int(3)),
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]),
}

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel=int(3)),
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

def downloadModels():
    url = "https://drive.google.com/u/1/uc?id=1VIeb558G9aWMoER5SQ8ljGKA9_A7MRms"
    path = "input/"
    filename = path + "models.zip"

    if not os.path.isfile(filename):
        gdown.download(url, filename, quiet=False)
        with ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(path)

class CIFAR10Pair(CIFAR10):
    """CIFAR-10 Dataset"""
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target
