import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt

from Flow import Flow, Loss
from misc import load_cifar


class LossMeter(object):

    def __init__(self):
        # self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
    
    def update(self, val, n):
        # self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self, lr=1e-3, epochs=100, device='cpu'):
        self.trainset, self.testset = load_cifar()
        self.epochs = epochs
        self.lr = lr
        self.device = device
    
    def build(self):
        self.loss_fn = Loss()
        self.net = Flow(nscales=2, cin=3, cmid=64, nblocks=8, device=self.device).to(self.device)
        # param_groups = self.get_param_groups()
        # optimizer = optim.Adam(param_groups, lr=args.lr)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self, max_norm=100.0):
        train_losses = []
        test_losses = []
        self.save_samples('initilization')
        for epoch in range(0, self.epochs):
            epoch += 1
            epoch_loss = LossMeter()
            print(f"[==> Epoch: {epoch}")

            self.net.train()

            batch = 0
            for img, _ in self.trainset:
                batch += 1
                # print(f'> Batch: {batch}')
                img = img.to(self.device)
                
                self.optim.zero_grad()

                z, sum_log_det = self.net(img, reverse=False)
                
                loss = self.loss_fn(z, sum_log_det)
                epoch_loss.update(loss.item(), img.size(0))
                
                loss.backward()

                for group in self.optim.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_norm, 2)
                
                self.optim.step()
            
            # Bits per dim per epoch
            # TODO
            print(f'Loss: {epoch_loss.avg:.2f}')

            # Save model
            # if epoch % 5 == 0:
            #     self.save_model(f'net_{epoch}.model')
            os.makedirs('input/models', exist_ok=True)
            self.save_model(f'input/models/net_{epoch}.model')
            # Save some samples
            self.save_samples(epoch)
            
            # Save train_losses
            os.makedirs('output/losses', exist_ok=True)
            train_losses.append(epoch_loss.avg)
            np.save('output/losses/train_losses.npy', np.array(train_losses))

            # Evaluate in testset
            test_loss, _ = self.eval() # _ = bpd
            test_losses.append(test_loss)
            np.save('output/losses/test_losses.npy', np.array(test_losses))
        
        self.save_model('net_{epoch}.model')

        return train_losses, test_losses

    
    def eval(self):
        self.net.eval()
        test_loss = LossMeter()
        with torch.no_grad():
            for img, _ in self.trainset:
                img = img.to(self.device)
                z, sum_log_det = self.net(img, reverse=False)
                loss = self.loss_fn(z, sum_log_det)
                test_loss.update(loss.item(), img.size(0))
        # Bits per dim
        bpd = 0
        # TODO 
        return test_loss.avg, bpd

    def sample(self, batch_size=25):
        with torch.no_grad():
            z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=self.device)
            x, _ = self.net(z, reverse=True)
            x = torch.sigmoid(x)
        return x.cpu()
    
    def save_samples(self, epoch):
        imgs = self.sample(batch_size=25)
        print(imgs.shape)
        os.makedirs('output/samples', exist_ok=True)
        img_grid = torchvision.utils.make_grid(imgs, nrow=5, padding=2, pad_value=255)
        
        # Plot
        plt.figure()
        plt.title('Samples')
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'output/samples/epoch_{epoch}.png')


    def save_model(self, fname):
        torch.save(self.net, fname)

    def load_model(self, fname):
        self.net = torch.load(fname, map_location="cpu")

    def visualize(self, fname, batch_size=25):
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
        plt.savefig(f'output/trainset.png', dpi=300)
        