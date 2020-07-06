import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from utils.GaussianBlur import GaussianBlur


def optim(files = ['input/optim/lars/128_0.5_32_10_stats.csv', 'input/optim/adam/128_0.5_32_10_stats.csv']):
    # Loss
    plt.clf()
    names = {'lars': 'LARS', 'adam': 'ADAM'}
    for f in files: 
        base_name = f.rsplit('/')[-2]
        df = pd.read_csv(f)
        x = df.index.values
        y = df['train_loss'].values
        plt.plot(x, y, label=names[base_name])
    plt.legend(title='Optimizer')
    plt.ylabel('Train Loss')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/loss/optim_loss.png')

    # Test Accuracy (%)
    plt.clf()
    for f in files: 
        base_name = f.rsplit('/')[-2]
        df = pd.read_csv(f)
        x = df.index.values
        y = df['test_acc@1'].values
        plt.plot(x, y, label=base_name)
    plt.legend(title='Optimizer')
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/acc/optim_acc.png')


def batch(files = ['input/batch/128_0.5_8_10_stats.csv', 'input/batch/128_0.5_16_10_stats.csv', 'input/batch/128_0.5_32_10_stats.csv']):
    # Loss 
    plt.clf()
    for f in files: 
        base_name = f.rsplit('_')[-3]
        df = pd.read_csv(f)
        x = df.index.values
        y = df['train_loss'].values
        plt.plot(x, y, label=base_name)
    plt.legend(title='Batch Size')
    plt.ylabel('Train Loss')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/loss/batch_loss.png')

    # Test Accuracy (%)
    plt.clf()
    for f in files: 
        base_name = f.rsplit('_')[-3]
        df = pd.read_csv(f)
        x = df.index.values
        y = df['test_acc@1'].values
        plt.plot(x, y, label=base_name)
    plt.legend(title='Batch Size')
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/acc/batch_acc.png')


def simfunc(files = ['input/simfunc/cos/128_0.5_32_10_stats.csv', 'input/simfunc/dot/128_0.5_32_10_stats.csv']):
    # Loss
    plt.clf()
    names = {'cos': 'cosine', 'dot': 'dot'}
    for f in files: 
        base_name = f.rsplit('/')[-2]
        df = pd.read_csv(f)
        x = df.index.values
        y = df['train_loss'].values
        plt.plot(x, y, label=names[base_name])
    plt.legend(title='Similarity')
    plt.ylabel('Train Loss')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/loss/simfunc_loss.png')

    # Test Accuracy (%)
    plt.clf()
    for f in files: 
        base_name = f.rsplit('/')[-2]
        df = pd.read_csv(f)
        x = df.index.values
        y = df['test_acc@1'].values
        plt.plot(x, y, label=base_name)
    plt.legend(title='Similarity')
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/acc/simfunc_acc.png')


def final(file = 'input/final/128_0.5_32_100_stats.csv'):
    # Loss
    plt.clf()
    df = pd.read_csv(file)
    x = df.index.values
    y = df['train_loss'].values
    plt.plot(x, y)
    plt.ylabel('Train Loss')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/loss/final_loss.png')

    # Test Accuracy (%)
    plt.clf()
    df = pd.read_csv(file)
    x = df.index.values
    y = df['test_acc@1'].values
    plt.plot(x, y)
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('output/acc/final_acc.png')


def prepare_transformations_data(labels):
    # Extract data
    names = {'rsc': 'Crop and resize', 'rhf': 'Horizontal flip', 'cj': 'Color jitter', 'gb': 'Gaussian blur', 'gs': 'Color drop'}
    base_dir = 'output/transf'
    raw_data = dict(transf_1=[], transf_2=[], acc=[])
    for label in labels:
        f = pd.read_csv(f'{base_dir}/{label}/128_0.5_32_10_stats.csv')
        transfs = label.split('_')
        if len(transfs) == 1:
            transf_1 = transf_2 = transfs[0]
        else:
            transf_1 = transfs[0]
            transf_2 = transfs[1]
        raw_data['transf_1'].append(names[transf_1])
        raw_data['transf_2'].append(names[transf_2])
        raw_data['acc'].append(f['test_acc@1'].iloc[-1])
    # Prepare table
    data = pd.DataFrame(raw_data, columns=['transf_1', 'transf_2', 'acc'])
    heatmap_data = pd.pivot_table(data, values='acc', index='transf_1', columns='transf_2')
    heatmap_data['Average'] = heatmap_data.mean(axis=1)
    heatmap_data.to_csv('output/transf/transf_acc.csv')


def transf(file = 'input/transf/transf_acc.csv'):
    plt.clf()
    f = pd.read_csv(file, index_col=0)
    p = sns.heatmap(f, square=True, annot=True, fmt='.1f', linewidths=.2, cmap='coolwarm')
    p.set_xticklabels(p.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light',
    fontsize=10)
    p.set_yticklabels(p.get_yticklabels(), fontweight='light',
    fontsize=10)
    plt.ylabel('1st Transformation')
    plt.xlabel('2nd Transformation')
    plt.tight_layout()
    plt.savefig('output/acc/transf_acc.png')


def plot_transformed(data):
    path = 'output/images/'
    plot_image(data, path + 'noTransformations.png')

    data = transforms.ToPILImage()(data)

    example_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=1),
        GaussianBlur(prob=1, kernel=int(3))
    ])

    for transformation in example_transforms.transforms:
        transformed = transformation(data)
        transformed = transforms.ToTensor()(transformed)
        transformed = transformed.permute(1, 2, 0)
        plot_image(transformed, path + transformation.__class__.__name__ + '.png')

def plot_image(image, filename):
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(image)
    plt.savefig(filename)
    plt.cla()
    plt.clf()