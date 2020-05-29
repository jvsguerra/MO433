import torch
import numpy as np

from train import Trainer

if __name__ == "__main__":
    # Seed
    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Creating model
    model = Trainer(lr=1e-3, epochs=20, device='cuda:0')
    model.visualize(model.trainset)
    # exit()

    # Building model
    print('[==> Building model ...')
    model.build()

    # Training model
    train_losses, test_losses = model.train() 

    # Generate samples
    # model.load_model("input/models/net_1.model")
    # model.save_samples(1)