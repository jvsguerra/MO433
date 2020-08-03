import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os

def plot_image(image, filename):
    image = torch.Tensor(image / 255)
    image = image.permute(1,2,0)
    image += np.array((104.00699/255, 116.66877/255, 122.67892/255))
    plt.cla()
    plt.clf()
    plt.axis('off')
    plt.imshow(image)
    plt.tight_layout()
    plt.savefig(filename)
    plt.cla()
    plt.clf()

def plot_label(image, filename, bin=True):
    image = torch.Tensor(image[0])
    plt.cla()
    plt.clf()
    plt.axis('off')
    if bin:
        plt.imshow(image, cmap=plt.cm.binary)
    else:
        plt.imshow(image)
    plt.tight_layout()
    plt.savefig(filename)
    plt.cla()
    plt.clf()

def plot_transformations(images, path):
    unsup_methods = ['ft', 'hc', 'rc', 'rdb']
    for index in range(len(images)):
        plt.cla()
        plt.clf()
        plt.axis('off')
        plt.imshow(images[index], cmap=plt.cm.binary)
        plt.tight_layout()
        plt.savefig(path + unsup_methods[index] + '.png')
        plt.cla()
        plt.clf()

def plot_stats(epoch=20):
    name_types = ['Real', 'Noise', 'Average', 'Complete']
    optimizers = ['Adam', 'SGD']
    steps = ['ReduceLROnPlateau', 'StepLR']
    datasets = ['train', 'val']

    for name_type in name_types:
        for optimizer in optimizers:
            for step in steps:
                for dataset in datasets:
                    data_path = 'input/results/' + name_type + '/' + optimizer + '/' + step + '/' + optimizer + '_' + step + '_' + str(epoch) + '_' + dataset + '_stats.csv'
                    if os.path.isfile(data_path):
                        train_stats = pd.read_csv(data_path)
                        for stat in train_stats.columns[1:]:
                            values = train_stats[stat]
                            stat = stat.capitalize()
                            if stat == "Mae":
                                stat = "Mean Absolute Error"
                            elif stat == "Pred_loss":
                                stat = "Prediction Loss"
                            if name_type == 'Complete':
                                plot_graph(train_stats.epoch.iloc[1:], values.iloc[1:], stat, 'output/' + name_type + '/' + optimizer + '/' + step + '/images/' + dataset + '/')
                            else: 
                                plot_graph(train_stats.epoch, values, stat, 'output/' + name_type + '/' + optimizer + '/' + step + '/images/' + dataset + '/')

def plot_final(epochs=100):
    # Read Data
    name_types = ['Final-Noise', 'Final-Average']
    optimizer = 'SGD'
    step = 'ReduceLROnPlateau'
    dataset = 'test'

    for name_type in name_types:
        data = pd.DataFrame()
        for epoch in [20, 40, 60, 80, 100]:
            data_path = 'input/results/' + name_type + '/' + optimizer + '/' + step + '/' + optimizer + '_' + step + '_' + str(epochs) + '_' + dataset + '_' + str(epoch) + '_stats.csv'
            stats = pd.read_csv(data_path)
            data = data.append(stats)
        data['Rounds'] = [1, 2, 3, 4, 5]

        # Plot MAE
        fig, ax = plt.subplots()
        data.plot.bar(x='Rounds', y='mae', rot=0, ax=ax, legend=False, color='black')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

        for bar in ax.patches:
            width, height = bar.get_width(), bar.get_height()
            x, y = bar.get_xy() 
            ax.annotate('{:.1%}'.format(height), (x + 0.025, y + height + 0.005), weight='bold', size=12)

        plt.rcParams.update({'font.size': 14})
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_ylim(0, 0.2)
        plt.tight_layout()
        plt.savefig('output/' + name_type + '/' + optimizer + '/' + step + '/images/' + dataset + '/MAE.png')

        # Plot F1
        fig, ax = plt.subplots()
        data.plot.bar(x='Rounds', y='f1', rot=0, ax=ax, legend=False, color='black')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.annotate('{:.1%}'.format(height), (x + 0.025, y + height + 0.01), weight='bold', size=12)

        plt.rcParams.update({'font.size': 14})
        ax.set_xlabel('Rounds')
        ax.set_ylabel(r'$F_\beta$')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig('output/' + name_type + '/' + optimizer + '/' + step + '/images/' + dataset + '/F1.png')

def plot_graph(x, y, y_label, path):
    plt.cla()
    plt.clf()
    plt.axis('on')
    plt.plot(x, y)
    plt.rcParams.update({'font.size': 18})
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(path + y_label + '.png')
    plt.cla()
    plt.clf()
