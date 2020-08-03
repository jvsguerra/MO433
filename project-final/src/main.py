import torch
import numpy as np
import random

from Trainer import Trainer
from utils.plots import plot_image, plot_label, plot_transformations, plot_stats, plot_final
from utils.download import unzip_results, download_models

if "__main__" == __name__:
    # Set seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # NOTE: Illustration 
    model = Trainer(device='cpu', batch_size=4, experiment='Real')
    trainset, valset, testset = model.trainset, model.valset, model.testset
    model.build()
    model.visualize_results(epoch='pre', pre_training=True)
    # model.fit()
    
    shouldTrain = False
    downloadModels = False   
    if shouldTrain:
        ## Experiments:
        # Real
        for exp in ['Real']:
            model = Trainer(device='cuda:0', batch_size=4, experiment=exp)
            model.trainset, model.valset, model.testset = trainset, valset, testset
            for optim in ['Adam', 'SGD']:
                for scheduler in ['StepLR', 'ReduceLROnPlateau']:
                    print(f'Running: {exp}, {optim}, {scheduler}')
                    model.build(optim=optim, scheduler=scheduler)
                    model.fit()

        # Noise
        for exp in ['Noise']:
            model = Trainer(device='cuda:0', batch_size=4, experiment=exp)
            model.trainset, model.valset, model.testset = trainset, valset, testset
            for optim in ['Adam', 'SGD']:
                for scheduler in ['StepLR', 'ReduceLROnPlateau']:
                    print(f'Running: {exp}, {optim}, {scheduler}')
                    model.build(optim=optim, scheduler=scheduler)
                    model.fit()

        # Average
        for exp in ['Average']:
            model = Trainer(device='cuda:0', batch_size=4, experiment=exp)
            model.trainset, model.valset, model.testset = trainset, valset, testset
            for optim in ['Adam', 'SGD']:
                for scheduler in ['StepLR', 'ReduceLROnPlateau']:
                    print(f'Running: {exp}, {optim}, {scheduler}')
                    model.build(optim=optim, scheduler=scheduler)
                    model.fit()

        # Complete
        for exp in ['Complete']:
            model = Trainer(device='cuda:0', batch_size=4, experiment=exp)
            model.trainset, model.valset, model.testset = trainset, valset, testset
            for optim in ['Adam', 'SGD']:
                for scheduler in ['StepLR', 'ReduceLROnPlateau']:
                    print(f'Running: {exp}, {optim}, {scheduler}')
                    model.build(optim=optim, scheduler=scheduler)
                    model.fit()

        # Final-Noise
        for exp in ['Final-Noise']:
            model = Trainer(device='cuda:0', batch_size=4, experiment=exp, epochs=100)
            model.trainset, model.valset, model.testset = trainset, valset, testset
            for optim in ['SGD']:
                for scheduler in ['ReduceLROnPlateau']:
                    print(f'Running: {exp}, {optim}, {scheduler}')
                    model.build(optim=optim, scheduler=scheduler)
                    model.fit_final()

        # Final-Average
        for exp in ['Final-Average']:
            model = Trainer(device='cuda:0', batch_size=4, experiment=exp, epochs=100)
            model.trainset, model.valset, model.testset = trainset, valset, testset
            for optim in ['SGD']:
                for scheduler in ['ReduceLROnPlateau']:
                    print(f'Running: {exp}, {optim}, {scheduler}')
                    model.build(optim=optim, scheduler=scheduler)
                    model.fit_final()
    else:
        unzip_results()
        if downloadModels:
            download_models() 

    # Save images, ground truth and transformations
    plot_image(model.trainset.dataset[0]['image'], 'output/samples/dataset.png')
    plot_label(model.trainset.dataset[0]['label'], 'output/samples/label.png')
    plot_transformations(model.trainset.dataset[0]['unsup_labels'], 'output/samples/')
    # Plot results
    plot_stats(epoch=20)
    plot_final()
