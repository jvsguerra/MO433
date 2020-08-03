import torch
import torch.nn as nn
import numpy as np


class NMM(nn.Module):
    """ Noise Modelling Module (NMM) """
    
    def __init__(self, device='cpu', num_imgs=2500, num_maps=4, img_size=(256, 256), alpha=0.01):
        """
        It maintains a variance for each pixel of each image.
        The variances (var) are stored for each of the prior distributions (sd = sqrt(var)).
        Class responsible for sampling (using var * N(0, 1)) from the noise distribution, loss calculation and updating the variance.
        args:
        device: cpu or cuda:n.
        num_imgs: Number of training images.
        num_maps: Number of color maps.
        img_size: Height x Width (pixels).
        alpha: Step size of the noise update.
        """
        super(NMM, self).__init__()
        # Device
        self.device = device
        # Image parameters
        self.num_imgs = num_imgs
        self.num_maps = num_maps
        self.h, self.w = img_size
        self.num_pixels = np.prod(img_size)
        self.alpha = alpha
        # prior variance of the noise distribution (initialized with zeros - topic 3.2 paper)
        self.noise_var = torch.zeros(self.num_imgs * self.num_pixels)
        # emperical variance, observed variance between prediction and unsup labels
        self.emp_var = torch.zeros(self.num_imgs * self.num_pixels)

    def get_index(self, arr=None, img_index=None):
        """      
        Get index of pixels for img_index
        args:
        img_index: Image index (int)
        arr: Array (list)
        """
        if arr is None:
            arr = self.noise_var
        index = img_index * self.num_pixels

        # Get values and starting and ending pixels of the image
        values = arr[index: index + self.num_pixels]
        indexes = np.arange(index, index + self.num_pixels)
        return values, indexes

    def get_index_multiple(self, arr=None, img_indexes=None):
        """
        Get indexes of pixels for img_index
        args:
        img_index: Image index (int)
        arr: Array (list)
        """
        if arr is None:
            arr = self.noise_var
        noise = np.zeros((len(img_indexes), self.num_pixels), dtype=np.float)
        indexes = np.zeros((len(img_indexes), self.num_pixels), dtype=np.float)

        # Get values and starting and ending pixels of the image
        for key, img_index in enumerate(img_indexes):
            index = img_index * self.num_pixels
            noise[key] = arr[index: index + self.num_pixels]
            indexes[key] = np.arange(index, index + float(self.num_pixels))
        return noise, indexes

    def loss(self, var1, var2):
        """
        Loss per image (Eq. 6; https://arxiv.org/pdf/1803.10910.pdf) # FIXME: paper uses sum of noise
        args:
        var1: q variance (prior)
        var2: p variance (predictive) 
        """
        noise_loss = 0
        for index in range(var1.shape[0]):
            covar1 = var1[index].to(self.device) + 1e-6
            covar2 = var2[index].to(self.device) + 1e-6
            ratio = 1. * (covar1 / covar2)
            loss = -0.5 * (torch.log(ratio) - ratio + 1).to(self.device)
            loss = abs(loss)
            noise_loss += torch.sum(loss) / var1.shape[1]
        noise_loss /= var1.shape[0]
        return noise_loss

    def sample(self, indexes):
        """
        Sample noise from Gaussian distribution with prior variance.
        args:
        indexes: list (int)

        samples: np.array, noise samples of shape (len(idxs), NUM_MAPS, 256, 256)
        """
        samples = torch.zeros(len(indexes), self.num_maps, self.h, self.w)
        for index, img_index in enumerate(indexes):
            var, _ = self.get_index(self.noise_var, img_index) # var, var_index
            var = var.reshape(self.h, self.w).to(self.device)
            for map_index in range(self.num_maps):
                # elementwise mutliplication with prior variance for each image of samples from N(0, 1)
                sample = var * torch.zeros(self.h, self.w).normal_().to(self.device)
                samples[index][map_index] = sample
        # Return noise samples of shape (len(idxs), num_maps, 256, 256)
        return samples

    def update(self):
        """
        Update prior variance per pixel of each image by empirical variance.
        Eq. 7 (https://arxiv.org/pdf/1803.10910.pdf)
        """
        # Empirical variance (emp_var) must be update before this function
        print(f'[==> Updating noise')
        self.noise_var = self.noise_var + self.alpha * (self.emp_var - self.noise_var)
        print(f'Max: {torch.max(self.noise_var):.4e}, Min: {torch.min(self.noise_var):.4e}')
