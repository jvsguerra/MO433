import torch
from torch.utils.data import DataLoader
import numpy as np

from trainer import Trainer
from nets.SimCLR import SimCLR
from utils.plots import plot_transformed, prepare_transformations_data, final, batch, optim, simfunc, transf
from utils.data import downloadModels, transform_dict, CIFAR10Pair

if "__main__" == __name__:
    # Set seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Download pre-trained models
    downloadModels()

    # Load best model
    bestCRLDict = torch.load("input/models/final/128_0.5_32_100_model.pth", map_location='cpu') 
    bestCRL = SimCLR()
    bestCRL.load_state_dict(bestCRLDict)
    bestTrainer = Trainer(device='cpu')
    bestTrainer.net = bestCRL
    bestTrainer.linearEvaluation()

    # Change this variable to execute training
    shouldTrain = False 

    # Creating sample of image in trainset and a transformed one
    exampleTrainer = Trainer()
    plot_transformed(exampleTrainer.trainset.dataset.data[20007])

    ###### Experiments ######
    if shouldTrain:
        # 1) Batch size: [8, 16, 32]
        batch_size = [8, 16, 32]
        for n in batch_size:
            model = Trainer(batch_size=n, epochs=10)
            model.build(use_lars=False)
            model.fit('batch')

        # 2) LARS x Adam
        use_lars = [True, False]
        for flag in use_lars:
            model = Trainer(batch_size=32, epochs=10)
            model.build(use_lars=flag)
            if flag:
                model.fit('optim/lars')
            else:
                model.fit('optim/adam')

        # 3) Cosine Similarity x Dot Similarity
        use_cos_sim = [True, False]
        for flag in use_cos_sim:
            model = Trainer(batch_size=32, epochs=10, use_cos_sim=flag)
            model.build(use_lars=False)
            if flag:
                model.fit('simfunc/cos')
            else:
                model.fit('simfunc/dot')

        # 4) Effects of transformations (pairwise)
        for label, transform in transform_dict.items():
            model = Trainer(batch_size=32, epochs=10, use_cos_sim=True)
            # Change trainset
            train_data = CIFAR10Pair(root='./input', train=True, transform=transform, download=True)
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
            model.trainset = train_loader
            # Build
            model.build(use_lars=False)
            # Fit
            model.fit(f'transf/{label}')
        # Save accuracies
        prepare_transformations_data([*transform_dict.keys()])

        # 5) Final model: LARS, batch_size=32, Cos Similarity, epochs=100
        model = Trainer(batch_size=32, epochs=100, use_cos_sim=True)
        model.build(use_lars=False)
        model.fit('final')

    ###### Evaluation ######
    if shouldTrain:
        # 1) Batch Size
        batch(['output/batch/128_0.5_8_10_stats.csv', 'output/batch/128_0.5_16_10_stats.csv', 'output/batch/128_0.5_32_10_stats.csv'])
        # 2) Optimizer
        optim(['output/optim/lars/128_0.5_32_10_stats.csv', 'output/optim/adam/128_0.5_32_10_stats.csv'])
        # 3) Similarity Function
        simfunc(['output/simfunc/cos/128_0.5_32_10_stats.csv', 'output/simfunc/dot/128_0.5_32_10_stats.csv'])
        # 4) Effects of transformations (pairwise)
        transf('output/transf/transf_acc.csv')
        # 5) Final
        final('output/final/128_0.5_32_100_stats.csv')
    else: 
        # 1) Batch Size
        batch()
        # 2) Optimizer
        optim()
        # 3) Similarity Function
        simfunc()
        # 4) Effects of transformations (pairwise)
        transf()
        # 5) Final
        final()