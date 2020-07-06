import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys
sys.path.append('../')
from nets.Flow import Flow, Loss, LossMeter


class Trainer(object):
    def __init__(self, lr=1e-3, epochs=100, device='cpu', subset=False, label='dog'):
        self.device = device
        self.trainset, self.testset = self.load_cifar(subset=subset, label=label)
        self.epochs = epochs
        self.lr = lr

    def load_cifar(self, subset=False, label='dog'):
        # Load CIFAR data
        print('> Loading data ...')

        # Classes
        labels = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

        # No normalization applied (values from 0 to 1)
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        train = torchvision.datasets.CIFAR10(root='./input', train=True, download=False, transform=transform_train)
        if subset:
            indices = np.array(train.targets) == labels[label]
            train.data = train.data[indices]
            if self.device == 'cpu':
                train.data = train.data[0:100]
        # NCHW 5000 x 3 x 32 x 32

        test = torchvision.datasets.CIFAR10(root='./input', train=False, download=False, transform=transforms.ToTensor())
        if subset:
            indices = np.array(test.targets) == labels[label]
            test.data = test.data[indices]
            if self.device == 'cpu':
                test.data = test.data[0:100]
        # NCHW 1000 x 3 x 32 x 32

        # Prepare DataLoader
        trainset = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
        testset = torch.utils.data.DataLoader(test, batch_size=8, shuffle=False)

        return trainset, testset   

    def build(self):
        self.loss_fn = Loss()
        self.net = Flow(nscales=2, cin=3, cmid=64, nblocks=8, device=self.device).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)

    def fit(self):
        # Create a loss record
        loss_history = {}
        loss_history['train'] = []
        loss_history['test'] = []

        # Create a bits_per_dim record
        bits_per_dim = {}
        bits_per_dim['train'] = []
        bits_per_dim['test'] = []
        
        # Save samples from initialization
        self.save_samples(f'output/samples/initialization.png')

        for epoch in range(0, self.epochs):
            
            # Count from one 
            epoch += 1
            print(f'\nEpoch: {epoch}')

            # Training step
            loss, bpd = self.train()
            loss_history['train'].append(loss)
            bits_per_dim['train'].append(bpd)
            np.save('output/losses/train_losses.npy', np.array(loss_history['train']))
            np.save('output/bpd/train_bpd.npy', np.array(bits_per_dim['train']))

            # Testing step
            loss, bpd = self.test()
            loss_history['test'].append(loss)
            bits_per_dim['test'].append(bpd)
            np.save('output/losses/test_losses.npy', np.array(loss_history['test']))
            np.save('output/bpd/test_bpd.npy', np.array(bits_per_dim['test']))

            # Sampling step
            self.save_samples(f'output/samples/epoch_{epoch}.png')

        #     # Save model
        #     if epoch in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        #         self.save_model(f'output/models/net_{epoch}.model')
        
        # self.save_model(f'output/models/net_final.model')
          
    def train(self, max_norm=100.0):
        train_loss = LossMeter()

        self.net.train()

        with tqdm(total=len(self.trainset.dataset)) as buffer:
            for img, _ in self.trainset:
               
                # Send img to gpu or cpu
                img = img.to(self.device)

                # Zero grad
                self.optim.zero_grad()
                
                # Forward pass
                z, sum_log_det = self.net(img, reverse=False)

                # Calculate loss
                loss = self.loss_fn(z, sum_log_det)

                # Update loss
                train_loss.update(loss.item(), img.size(0))

                # Calculate bits per dim
                bits_per_dim = train_loss.avg / (np.prod(img.size()[1:]) * np.log(2))

                # Backpropagate loss
                loss.backward()

                for group in self.optim.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_norm, 2)
                
                self.optim.step()

                # Update buffer
                buffer.set_postfix(loss=train_loss.avg, bpd=bits_per_dim)
                buffer.update(img.size(0))

        return train_loss.avg, bits_per_dim

    def test(self):
        test_loss = LossMeter()
        
        self.net.eval()

        # Do not calculate gradient
        with torch.no_grad():
            with tqdm(total=len(self.testset.dataset)) as buffer:
                for img, _ in self.testset:
                    # Send img to gpu or cpu
                    img = img.to(self.device)
                    
                    # Forward pass
                    z, sum_log_det = self.net(img, reverse=False)

                    # Calculate loss
                    loss = self.loss_fn(z, sum_log_det)

                    # Update loss
                    test_loss.update(loss.item(), img.size(0))

                    # Calculate bits per dim
                    bits_per_dim = test_loss.avg / (np.prod(img.size()[1:]) * np.log(2))
                                    
                    # Update buffer
                    buffer.set_postfix(loss=test_loss.avg, bpd=bits_per_dim)
                    buffer.update(img.size(0))

        return test_loss.avg, bits_per_dim

    def sample(self, batch_size=25):
        with torch.no_grad():
            z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=self.device)
            x, _ = self.net(z, reverse=True)
            x = torch.sigmoid(x)
        return x.cpu()
    
    def save_samples(self, fname):
        imgs = self.sample(batch_size=25) # * 255.0 / 255
        img_grid = torchvision.utils.make_grid(imgs, nrow=5, padding=2, pad_value=255, scale_each=True) #, range=(0, 1))

        # Plot
        plt.figure()
        plt.title('Samples')
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fname)

    def save_model(self, fname):
        torch.save(self.net, fname)

    def load_model(self, fname):
        self.net = torch.load(fname, map_location="cpu")

    def visualize(self, fname=f'output/trainset.png', batch_size=25):
        # Get data
        x = self.trainset.dataset.data
        # Get random images
        idxs = np.random.choice(x.shape[0], replace=False, size=(batch_size, ))

        # Prepare
        x = (torch.FloatTensor(x[idxs]) / 255).permute(0, 3, 1, 2)
        img_grid = torchvision.utils.make_grid(x, nrow=5, padding=2, pad_value=255)

        # Plot
        plt.figure()
        plt.title('Train Set')
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(fname, dpi=300)