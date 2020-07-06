import cv2
import numpy as np

# Seed
np.random.seed(0)

class GaussianBlur(object):
    """Gaussian Blur as described in SimCLR"""
    def __init__(self, prob=0.5, kernel=3, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel = kernel # 10% of image 32x32
        self.prob = prob

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.prob:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel, self.kernel), sigma)
        
        return sample