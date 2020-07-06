import torch
import numpy as np

from utils.train import Trainer
from utils.plots import plot_loss, plot_bpd

if __name__ == "__main__":
    # Seed
    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Gitlab
    print('Running our network for one epoch ...')
    model = Trainer(lr=1e-3, epochs=1, device='cpu', subset=True, label='dog')
    print('[==> Visualize training images ...')
    model.visualize(fname='output/dogs_cifar_trainset.png')
    print('[==> Building model ...')
    model.build()
    print('[==> Fitting model ...')
    model.fit()

    # Pre-trained model
    # 1) Full CIFAR-10
    model = Trainer(lr=1e-3, epochs=30, device='cpu', subset=False)
    print('[==> Visualize training images ...')
    model.visualize(fname='output/cifar_trainset.png')
    model.build()
    # Load pre-trained model
    print('[==> Loading pre-trained model')
    model.load_model('input/pre_trained/cifar/net_final.model')
    print('> Sampling')
    model.save_samples('output/cifar/pre_trained_cifar.png')
    print('> Plotting loss and bits/dim')
    # Load data
    train_losses = np.load('input/pre_trained/cifar/train_losses.npy')
    test_losses = np.load('input/pre_trained/cifar/test_losses.npy')
    train_bpd = np.load('input/pre_trained/cifar/train_bpd.npy')
    test_bpd = np.load('input/pre_trained/cifar/test_bpd.npy')

    # Remove outliers cifar
    correct_test_losses = []
    for value in test_losses:
        if value < 60000:
            correct_test_losses.append(value)
        else: 
            correct_test_losses.append(correct_test_losses[-1])
    test_losses = np.array(correct_test_losses)

    correct_test_bpd = []
    for value in test_bpd:
        if value < 60000/(3072*np.log(2)):
            correct_test_bpd.append(value)
        else: 
            correct_test_bpd.append(correct_test_bpd[-1])
    test_bpd = np.array(correct_test_bpd)
    print(test_bpd)
    # Plot
    plot_loss(train_losses, 'Train Loss', 'output/cifar/train_loss.png')
    plot_loss(test_losses, 'Test Loss', 'output/cifar/test_loss.png')
    plot_bpd(train_bpd, 'Train bits/dim', 'output/cifar/train_bpd.png')
    plot_bpd(test_bpd, 'Test bits/dim', 'output/cifar/test_bpd.png')

    # 2) Dog CIFAR-10
    model = Trainer(lr=1e-5, epochs=50, device='cpu', subset=True, label='dog')
    print('[==> Visualize training images ...')
    model.visualize(fname='output/dogs_cifar_trainset.png') 
    model.build()
    # Load pre-trained model
    print('[==> Loading pre-trained model')
    model.load_model('input/pre_trained/dogs_cifar/net_final.model')
    print('> Sampling')
    model.save_samples('output/dogs_cifar/pre_trained_dogs_cifar.png')
    print('> Plotting loss and bits/dim')
    # Load data
    train_losses = np.load('input/pre_trained/dogs_cifar/train_losses.npy')
    test_losses = np.load('input/pre_trained/dogs_cifar/test_losses.npy')
    train_bpd = np.load('input/pre_trained/dogs_cifar/train_bpd.npy')
    test_bpd = np.load('input/pre_trained/dogs_cifar/test_bpd.npy')
    # Plot
    plot_loss(train_losses, 'Train Loss', 'output/dogs_cifar/train_loss.png')
    plot_loss(test_losses, 'Test Loss', 'output/dogs_cifar/test_loss.png')
    plot_bpd(train_bpd, 'Train bits/dim', 'output/dogs_cifar/train_bpd.png')
    plot_bpd(test_bpd, 'Test bits/dim', 'output/dogs_cifar/test_bpd.png')
    print(test_bpd)

