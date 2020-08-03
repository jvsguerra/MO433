import os
import glob
import zipfile
import numpy as np
import pandas as pd
import math
import torch
from torch.optim import SGD, Adam, lr_scheduler # Lars tb possivel (projeto 2) 
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms.functional import to_tensor
from PIL import Image
from random import randint, sample

# Custom modules
from models.NoiseModule import NMM
from models.SaliencyModule import SPM
from utils.data import ImageDataTrain, ImageDataTest, min2d, max2d
from utils.metrics import mae, f1, recall, precision, to_numpy
from utils.plots import plot_image, plot_label

# NOTE: Bom para fazer barra de progresso, compativel com o DataLoader
from tqdm import tqdm

class Trainer(object):
    def __init__(self, device='cpu', batch_size=32, n_train=2500, n_val=500, n_test=2000, epochs=20, experiment='Real'):
        """
        experiment = Training methodology (Real, Noise, Average, Complete) [default: Real]
        * Real: SPM trained with ground truth label (loss = mean((pred_y_i, y_gt_i))))
        * Noise: SPM trained using loss of unsupervised labels (loss = mean((pred_y_i, unup_y_i_m)))
        * Average: SPM trained with average of unsupervised labels as ground truth (loss = mean((pred_y_i, mean(unup_y_i_m))))
        * Complete: SPM and NMM trained together with the following procedure:
            Round 1:
                - Initialize variance of prior noise: 0.0
                - Train SPM on unsupervised label (loss = bcewithlogitloss(pred_y, unsup_y_i))
                - Trained until convergence
                - Update NMM using update rule (Eq. 7)
            Round i:
                - Sample noise from NMM
                - Train SPM on unsupervised label (loss = bcewithlogitloss(pred_y, unsup_y_i) + noise_loss (Eq. 6))
                - Train until convergence
                - Update NMM using update rule (Eq. 7) 
        """
        # Device to run torch
        self.device = device
        # Epochs
        self.epochs = epochs
        # Batch size
        self.batch_size = batch_size
        # Size of sets
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        # Prepare data
        self.trainset, self.valset, self.testset = self.prepare_data()
        # Experiment
        self.experiment = experiment

    @staticmethod
    def unzip(path, dataset_name):
        # Unzip dataset if is not unzip yet
        if not os.path.isdir(path + dataset_name):
            print("[==> Extracting database ...")
            with zipfile.ZipFile(path + dataset_name + ".zip", 'r') as zip_ref:
                zip_ref.extractall(path)
            print("Database extracted!")

    def prepare_data(self, unsup_methods=['ft', 'hc', 'rc', 'rdb']):
        # Base information: root and dataset name
        root = "./input/dataset/"
        noise_root = "./input/unsup_labels/"
        dataset_name = "MSRA-B"

        # Unzip dataset
        self.unzip(root, dataset_name)

        # Get images names and unsup labels
        filenames = [f.split('/')[-1] for f in sorted(glob.glob(root + dataset_name + '/*.jpg'))]
        filenames_with_ngt = [f.split('/')[-1].replace('_ngt.png', '.jpg') for f in sorted(glob.glob(noise_root + unsup_methods[0] + '/*.png'))]

        # Get random ngt_names to prepare trainset
        train = sample(filenames_with_ngt, self.n_train)
        
        # Diff between train and filenames
        filenames = list(set(filenames) - set(train))

        # Prepare validation set
        val = sample(filenames, self.n_val)
        filenames = list(set(filenames) - set(val))

        # Prepare test set
        test = sample(filenames, self.n_test)

        # Prepare data.Dataset
        noise_root = ('./input/unsup_labels/' + d for d in unsup_methods)
        trainset = ImageDataTrain(filenames=train, noise_root=noise_root)
        valset = ImageDataTest(filenames=val)
        testset = ImageDataTest(filenames=test)

        # Prepare DataLoader
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        return train_loader, val_loader, test_loader
    
    def build(self, optim='Adam', lr=1e-3, momentum=0.9, weight_decay=0.0, betas=(0.9, 0.99), scheduler='StepLR', step_size=10, factor=0.9, min_lr=1e-12, patience=10, cooldown=1, threshold=1e-6, eps=1e-24):
        """
        optim = ['Adam', 'SGD', 'Lars'] [default: Adam]
        optim_options = {lr = learning rate [default: 1e-3],
                         momentum = momentum [default: 0.9],
                         weight_decay = weight_decay [default: 0.0],
                         betas = coefficients used for computing running averages of gradient and its square [default: (0.9, 0.99)]
                         }
        scheduler = ['StepLR', 'ReduceLROnPlateau'] [default: scheduler.StepLR]
        schduler_options = {step_size = step size [default: 10],
                            factor = decay_factor [default: 0.9],
                            min_lr = minimum learning rate [default: 1e-12],
                            patience = patience [default: 10],
                            cooldown = cooldown [default: 1],
                            threshold = threshold [default: 1e-6],
                            eps = minimal decay [default: ]
                            }
        """
        # Define networks
        self.SPM = SPM().build().to(self.device)
        if self.experiment == 'Final-Average':
            self.NMM = NMM(device=self.device, num_imgs=self.n_train, num_maps=1).to(self.device)
        else:
            self.NMM = NMM(device=self.device, num_imgs=self.n_train).to(self.device)
        # Define optimizer
        if optim == 'Adam':
            self.optim = Adam(self.SPM.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif optim == 'SGD':
            self.optim = SGD(self.SPM.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.optim.__dict__['__name__'] = optim
        # Define scheduler
        if scheduler == 'StepLR':
            self.scheduler = lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=factor)
        elif scheduler == 'ReduceLROnPlateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=factor, min_lr=min_lr, patience=patience, cooldown=cooldown, threshold=threshold, eps=eps)
        self.scheduler.__dict__['__name__'] = scheduler
        # Define loss function
        self.loss_fn = BCEWithLogitsLoss()
        return True

    def fit(self):
        # Best loss
        best_f1 = 0.0

        # Flag for training noise
        self.train_noise = False

        # Statistics
        results = {
            'train': {'loss': [], 'noise_loss': [], 'pred_loss': [], 'f1': [], 'mae': [], 'precision': [], 'recall': []}, 
            'val': {'pred_loss': [], 'f1': [], 'mae': [], 'precision': [], 'recall': []}
        }
        
        for epoch in range(1, self.epochs + 1):
            print(f'\nEpoch: {epoch}')

            # Train procedure
            tl, pl, nl, m, p, r, f = self.train(epoch)
            results['train']['loss'].append(tl)
            results['train']['pred_loss'].append(pl)
            results['train']['noise_loss'].append(nl)
            results['train']['mae'].append(m)
            results['train']['precision'].append(p)
            results['train']['recall'].append(r)
            results['train']['f1'].append(f)

            # Save statistics (Training step)
            df = pd.DataFrame(data=results['train'], index=range(1, epoch + 1))
            df.to_csv(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_train_stats.csv', index_label='epoch')

            # Visualize some results
            if epoch in [1, 2, 5, 10, 20]:
                self.visualize_results(epoch=epoch, n_samples=5)
        if self.experiment == 'Final':
            self.NMM = NMM(device=self.device, num_imgs=self.n_train, num_maps=1).to(self.device)
        else:
            # Validation procedure
            vl, m, p, r, f = self.eval(epoch)
            results['val']['pred_loss'].append(to_numpy(vl))
            results['val']['mae'].append(m)
            results['val']['precision'].append(p)
            results['val']['recall'].append(r)
            results['val']['f1'].append(f)

            # Save statistics (Validation step)
            df = pd.DataFrame(data=results['val'], index=range(1, epoch + 1))
            df.to_csv(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_val_stats.csv', index_label='epoch')

            # Update noise
            if self.experiment in ['Real', 'Noise', 'Average']:
                self.train_noise = False
            else:
                if self.train_noise == False:
                    self.update_noise()
                    self.train_noise = True

            # Update scheduler
            print('[==> Updating learning rate')
            if self.scheduler.__name__ == 'StepLR':
                self.scheduler.step()
            else:
                self.scheduler.step(vl)
            print(f"lr: {self.optim.param_groups[0]['lr']}")

            # Save models [1, 20]
            if epoch in [1, 20]:
                torch.save(self.SPM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_SPM_{epoch}.pth')
                torch.save(self.NMM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_NMM_{epoch}.pth')
            
            # Save best model
            if best_f1 < f:
                best_f1 = f
                torch.save(self.SPM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_SPM_best.pth')
                torch.save(self.NMM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_NMM_best.pth')
                with open(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/best', 'w') as f:
                    f.write(f'Best epoch: {epoch}\n')

        # Test procedure
        self.test()

        return True

    def train(self, epoch):
        # Train SPM
        self.SPM.train()

        # Statistics
        train_pred_loss, train_noise_loss, train_count = 0.0, 0.0, 0
        metrics = {'mae': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        with tqdm(self.trainset) as train_bar:
            for batch_idx, data in enumerate(train_bar):
                # x: Input data, (batch_size, 3, 256, 256)
                x = data['image'].to(self.device)
                # y: GT label (batch_size, 1, 256, 256)
                y = data['label'].to(self.device)
                # Keep original x to compute metrics upon true data
                original_x = x

                # Prepare unsupervised labels to experiments
                if self.experiment == 'Real':
                    # y_noise: GT label, (batch_size, 1, 256, 256)
                    y_noise = y
                elif self.experiment == 'Noise':
                    y_noise = data['unsup_labels'].to(self.device)
                    # x: repeat input for each map (batch_size*NUM_MAPS, 3, 256, 256)
                    # x = torch.repeat_interleave(x, repeats=4, dim=0)
                    # y_noise: Unsupervised labels (batch_size, 1, 256, 256)
                    # y_noise = y_noise.view(-1).view(-1, 1, 256, 256)
                elif self.experiment == 'Average':
                    y_noise = data['unsup_labels'].to(self.device)
                    # y_noise: Taking mean unsupervised labels (batch_size, 1, 256, 256)
                    y_noise = torch.mean(y_noise, dim=1, keepdim=True)
                else:
                    y_noise = data['unsup_labels'].to(self.device)

                # Normalizing y_noise between (0, 1)
                y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
                y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)
                # pred: (batch_size, 1, 256, 256)
                pred = self.SPM(x)['out']
                y_pred = pred

                # Noise training in Complete experiment
                if self.experiment == 'Complete':
                    # Round 1
                    if not self.train_noise:
                        # pred: In round 1 repeat along dim=1, (batch_size, NUM_MAPS(4), 256, 256)
                        pred = torch.repeat_interleave(pred, repeats=4, dim=1)
                        y_pred = pred
                    # Round > 1
                    else:
                        # noise_prior: noise from NMM (batch_size, NUM_MAPS, 256, 256)
                        noise_prior = self.NMM.sample(data['index']).to(self.device)
                        # noisy_pred: Noisy predictions after adding noise to predictions, (batch_size, NUM_MAPS, 256, 256)
                        noisy_pred = pred + noise_prior
                        # Range inside [0, 1] (see 3.2 after Eq 4)
                        noisy_min, noisy_max = min2d(noisy_pred), max2d(noisy_pred)
                        noisy_pred = (noisy_pred - noisy_min) / (noisy_max - noisy_min)
                        y_pred = noisy_pred
                
                # Compute BCE loss (Eq. 4)
                if self.experiment == 'Noise':
                    pred_loss = 0
                    for label in range(y_noise.shape[1]):
                        pred_loss += self.loss_fn(y_pred, y_noise[:, label, :, :][:, np.newaxis, ...])
                else:
                    pred_loss = self.loss_fn(y_pred, y_noise)

                # Noise loss
                noise_loss = 0
                if self.train_noise:
                    # compute batch noise loss (Eq 6)
                    emp_var = torch.var(y_noise - pred, 1).reshape(self.batch_size, -1) + 1e-16
                    prior_var, var_idx = self.NMM.get_index_multiple(img_indexes=data['index'])
                    prior_var = torch.from_numpy(prior_var).float()
                    # Important Order for loss var needs to get close to emp_var
                    noise_loss = self.NMM.loss(prior_var, emp_var)

                # Backprogation
                self.optim.zero_grad()
                # Total loss computed (Eq. 2)
                total_loss = pred_loss + 0.01 * noise_loss # lambda: 0.01
                total_loss.backward()
                self.optim.step()

                # Save batch losses
                train_count += self.batch_size
                train_pred_loss += to_numpy(pred_loss) * self.batch_size
                if self.experiment == 'Complete':
                    if not self.train_noise:
                        train_noise_loss += noise_loss * self.batch_size
                    else:
                        train_noise_loss += to_numpy(noise_loss) * self.batch_size
                else:
                    train_noise_loss += noise_loss * self.batch_size

                # Compute metrics
                y_pred_t = self.SPM(original_x)['out']
                y_pred_t = to_numpy(y_pred_t)
                y = to_numpy(y)
                metrics['mae'] += mae(y_pred_t, y) * self.batch_size
                metrics['precision'] += precision(y_pred_t, y) * self.batch_size
                metrics['recall'] += recall(y_pred_t, y) * self.batch_size

                # Update tdqm
                train_bar.set_description(f'Train Epoch [{epoch}/{self.epochs}] Loss: { (train_noise_loss + train_pred_loss) / train_count:.4f} Noise: {train_noise_loss/train_count:.4f} Pred: {train_pred_loss / train_count:.4f}')

                # Remove data from GPU memory
                del x, y, y_noise, y_pred, y_pred_t, pred, data
                torch.cuda.empty_cache()
        
        # Prepare statistics
        total_train_loss = (train_noise_loss + train_pred_loss) / train_count
        train_noise_loss /= train_count
        train_pred_loss /= train_count
        metrics['mae'] /= train_count
        metrics['precision'] /= train_count
        metrics['recall'] /= train_count
        metrics['f1'] = f1(metrics['precision'], metrics['recall'], beta2=0.3)

        return total_train_loss, train_pred_loss, train_noise_loss, metrics['mae'], metrics['precision'], metrics['recall'], metrics['f1']

    def eval(self, epoch):
        # Eval SPM
        self.SPM.eval()
        
        # Validation statistics
        val_pred_loss, val_count = 0.0, 0
        metrics = {'mae': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Run Batch
        with torch.no_grad():
            with tqdm(self.valset) as val_bar:
                for batch_idx, data in enumerate(val_bar):
                    # Prepare data
                    x = data['image'].to(self.device)
                    y = data['label'].to(self.device)
                    y_pred = self.SPM(x)['out']
                    
                    # Compute loss
                    pred_loss = self.loss_fn(y_pred, y)

                    # Save batch losses
                    val_count += self.batch_size
                    val_pred_loss += pred_loss * self.batch_size

                    # Compute metrics
                    y_pred = to_numpy(y_pred)
                    y = to_numpy(y)
                    metrics['mae'] += mae(y_pred, y) * self.batch_size
                    metrics['precision'] += precision(y_pred, y) * self.batch_size
                    metrics['recall'] += recall(y_pred, y) * self.batch_size

                    # Update tdqm
                    val_bar.set_description(f'Validation [{epoch}/{self.epochs}] Loss: {val_pred_loss / val_count:.4f}')

                    # Remove data from GPU memory
                    del x, y, y_pred, data, batch_idx, pred_loss
                    torch.cuda.empty_cache()
           
        # Save epoch losses and metrics
        val_pred_loss = val_pred_loss / val_count 
        metrics['mae'] = metrics['mae'] / val_count 
        metrics['precision'] = metrics['precision'] / val_count
        metrics['recall'] = metrics['recall'] / val_count
        metrics['f1'] = f1(metrics['precision'] / val_count, metrics['recall'] / val_count, beta2=0.3) 

        return val_pred_loss, metrics['mae'], metrics['precision'], metrics['recall'], metrics['f1']

    def update_noise(self):
        """
        Update noise (Eq. 7)
        """
        with tqdm(self.trainset) as noise_bar:
            for batch_idx, data in enumerate(noise_bar):
                # x : input data, (None, 3, 256, 256)
                x = data['image'].to(self.device)
                # y_noise: Unsup labels (None, NUM_MAPS(4), 256, 256)
                y_noise = data['unsup_labels'].to(self.device)
                # normalize noise value
                y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
                y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)
                # pred: (None, 1, 256, 256)
                pred = self.SPM(x)['out']
                # emp_var: Emperical Variance for each pixel for each image (None, 1, 256, 256)
                emp_var = torch.var(y_noise - pred, 1).reshape(self.batch_size, -1).detach().cpu()
                _, var_index = self.NMM.get_index_multiple(img_indexes=data['index'])
                self.NMM.emp_var[var_index.reshape(-1)] = emp_var.reshape(-1).to(self.NMM.emp_var.device)
                noise_bar.set_description(f'Noise update')
        # Compute emperical variance for each image and update prior variance
        self.NMM.update()

    def visualize_results(self, epoch, n_samples=5, pre_training=False):
        if pre_training:
            print('[==> Visualizing pre-training images')
            filename = f'output/samples'
            n_samples = 2
        else:
            print('[==> Visualizing results')
            filename = f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/images' 
        
        images = []
        images.append(self.trainset.dataset[0]['image'])
        if epoch == 1:
            plot_image(self.trainset.dataset[0]['image'], filename=f'{filename}/0_real.png')
            plot_label(self.trainset.dataset[0]['label'], filename=f'{filename}/0_gt.png')
        elif epoch == 'pre':
            plot_image(self.trainset.dataset[0]['image'], filename=f'{filename}/dataset_real.png')
            # plot_label(self.trainset.dataset[0]['label'], filename=f'{filename}/dataset_gt.png')
        for num in range(1, n_samples):
            if epoch == 1:
                plot_image(self.valset.dataset[num]['image'], filename=f'{filename}/{num}_real.png')
                plot_label(self.valset.dataset[num]['label'], filename=f'{filename}/{num}_gt.png')
            images.append(self.valset.dataset[num]['image'])
        images = torch.stack(images)

        x = images.to(self.device)
        seg_maps = self.SPM(x)['out']

        if not pre_training:
            num = 0    
            for seg_map in seg_maps:
                seg_map = seg_map.squeeze().cpu().data.numpy()
                im = to_tensor(Image.fromarray(seg_map * 255).convert('RGB'))
                plot_label(im, filename=f'{filename}/{num}_pred_{epoch}.png', bin=False)
                num += 1
        else:
            seg_map = seg_maps[0].squeeze().cpu().data.numpy()
            im = to_tensor(Image.fromarray(seg_map * 255).convert('RGB'))
            plot_label(im, filename=f'{filename}/dataset_pred_{epoch}.png', bin=False)

        seg_maps[seg_maps >= 0.5] = 1.0
        seg_maps[seg_maps < 0.5] = 0.0
        if not pre_training:
            num = 0
            for seg_map in seg_maps:
                seg_map = seg_map.squeeze().cpu().data.numpy()
                im = to_tensor(Image.fromarray(seg_map * 255).convert('RGB'))
                plot_label(im, filename=f'{filename}/{num}_pred_binary_{epoch}.png')
                num += 1
        else:
            seg_map = seg_maps[0].squeeze().cpu().data.numpy()
            im = to_tensor(Image.fromarray(seg_map * 255).convert('RGB'))
            plot_label(im, filename=f'{filename}/dataset_pred_binary_{epoch}.png')

        # Remove data from GPU memory      
        del x, seg_maps, images
        torch.cuda.empty_cache()

        return True

    def test(self):
        # Test SPM
        self.SPM.eval()

        # Test statistics
        test_count = 0
        metrics = {'mae': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Run Batch
        with torch.no_grad():
            with tqdm(self.testset) as test_bar:
                for batch_idx, data in enumerate(test_bar):
                    # Prepare data
                    x = data['image'].to(self.device)
                    y = data['label'].to(self.device)
                    y_pred = self.SPM(x)['out']

                    # Save batch losses
                    test_count += self.batch_size

                    # Compute metrics
                    y_pred = to_numpy(y_pred)
                    y = to_numpy(y)
                    metrics['mae'] += mae(y_pred, y) * self.batch_size
                    metrics['precision'] += precision(y_pred, y) * self.batch_size
                    metrics['recall'] += recall(y_pred, y) * self.batch_size
                    metrics['f1'] = f1(metrics['precision'] / test_count, metrics['recall'] / test_count, beta2=0.3)

                    # Update tdqm
                    F1 = metrics['f1']
                    MAE = metrics['mae']
                    test_bar.set_description(f'Test: F1-score: {F1:.4f} MAE: {MAE / test_count:.4f}')
        
        # Save statistcs
        metrics['mae'] /= test_count
        metrics['precision'] /= test_count
        metrics['recall'] /= test_count
        df = pd.DataFrame(data=metrics, index=[0])
        df.to_csv(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_test_stats.csv', index_label='epoch')
        
        return True

    def fit_final(self):
        # Best loss
        best_f1 = 0.0

        # Flag for training noise
        self.train_noise = False

        # Statistics
        results = {
            'train': {'loss': [], 'noise_loss': [], 'pred_loss': [], 'f1': [], 'mae': [], 'precision': [], 'recall': []}, 
            'val': {'pred_loss': [], 'f1': [], 'mae': [], 'precision': [], 'recall': []}
        }
        
        for epoch in range(1, self.epochs + 1):
            print(f'\nEpoch: {epoch}')

            # Train procedure
            tl, pl, nl, m, p, r, f = self.train_final(epoch)
            results['train']['loss'].append(tl)
            results['train']['pred_loss'].append(pl)
            results['train']['noise_loss'].append(nl)
            results['train']['mae'].append(m)
            results['train']['precision'].append(p)
            results['train']['recall'].append(r)
            results['train']['f1'].append(f)

            # Save statistics (Training step)
            df = pd.DataFrame(data=results['train'], index=range(1, epoch + 1))
            df.to_csv(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_train_stats.csv', index_label='epoch')

            # Validation procedure
            vl, m, p, r, f = self.eval(epoch)
            results['val']['pred_loss'].append(to_numpy(vl))
            results['val']['mae'].append(m)
            results['val']['precision'].append(p)
            results['val']['recall'].append(r)
            results['val']['f1'].append(f)

            # Save statistics (Validation step)
            df = pd.DataFrame(data=results['val'], index=range(1, epoch + 1))
            df.to_csv(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_val_stats.csv', index_label='epoch')

            # Update noise
            if epoch in [20, 40, 60, 80, 100]:
                self.update_noise()
                self.train_noise = True

            # Visualize some results
            if epoch in [1, 20, 40, 60, 80, 100]:
                self.visualize_results(epoch=epoch, n_samples=3)

            # Test procedure
            if epoch in [20, 40, 60, 80, 100]:
                self.test_final(epoch)

            # Update scheduler
            print('[==> Updating learning rate')
            if self.scheduler.__name__ == 'StepLR':
                self.scheduler.step()
            else:
                self.scheduler.step(vl)
            print(f"lr: {self.optim.param_groups[0]['lr']}")

            # Save models [20, 40, 60, 80, 100]
            if epoch in [20, 40, 60, 80, 100]:
                torch.save(self.SPM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_SPM_{epoch}.pth')
                torch.save(self.NMM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_NMM_{epoch}.pth')
            
            # Save best model
            if best_f1 < f:
                best_f1 = f
                torch.save(self.SPM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_SPM_best.pth')
                torch.save(self.NMM.state_dict(), f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_NMM_best.pth')
                with open(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/best', 'w') as f:
                    f.write(f'Best epoch: {epoch}\n')

    def train_final(self, epoch):
        # Train SPM
        self.SPM.train()

        # Statistics
        train_pred_loss, train_noise_loss, train_count = 0.0, 0.0, 0
        metrics = {'mae': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        with tqdm(self.trainset) as train_bar:
            for batch_idx, data in enumerate(train_bar):
                # x: Input data, (batch_size, 3, 256, 256)
                x = data['image'].to(self.device)
                # y: GT label (batch_size, 1, 256, 256)
                y = data['label'].to(self.device)
                # Keep original x to compute metrics upon true data
                original_x = x

                # Prepare unsupervised labels to experiments
                if self.experiment == 'Final-Average':                    
                    y_noise = data['unsup_labels'].to(self.device)
                    # y_noise: Taking mean unsupervised labels (batch_size, 1, 256, 256)
                    y_noise = torch.mean(y_noise, dim=1, keepdim=True)
                elif self.experiment == 'Final-Noise':                
                    y_noise = data['unsup_labels'].to(self.device)

                # Normalizing y_noise between (0, 1)
                y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
                y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)
                # pred: (batch_size, 1, 256, 256)
                pred = self.SPM(x)['out']
                y_pred = pred

                # Noise training in Complete experiment
                # Round 1
                if epoch <= 20:
                    if self.experiment == 'Final-Noise':
                        repeat = 4
                    elif self.experiment == 'Final-Average':
                        repeat = 1
                    # pred: In round 1 repeat along dim=1, (batch_size, NUM_MAPS(4), 256, 256)
                    pred = torch.repeat_interleave(pred, repeats=repeat, dim=1)
                    y_pred = pred
                # Round > 1
                else:
                    # noise_prior: noise from NMM (batch_size, NUM_MAPS, 256, 256)
                    noise_prior = self.NMM.sample(data['index']).to(self.device)
                    # noisy_pred: Noisy predictions after adding noise to predictions, (batch_size, NUM_MAPS, 256, 256)
                    noisy_pred = pred + noise_prior
                    # Range inside [0, 1] (see 3.2 after Eq 4)
                    noisy_min, noisy_max = min2d(noisy_pred), max2d(noisy_pred)
                    noisy_pred = (noisy_pred - noisy_min) / (noisy_max - noisy_min)
                    y_pred = noisy_pred
                
                # Compute BCE loss (Eq. 4)
                pred_loss = self.loss_fn(y_pred, y_noise)

                # Noise loss
                noise_loss = 0
                if self.experiment == 'Final-Average': 
                    y_noise = data['unsup_labels'].to(self.device)
                if epoch > 20:
                    # compute batch noise loss (Eq 6)
                    emp_var = torch.var(y_noise - pred, 1).reshape(self.batch_size, -1) + 1e-16
                    prior_var, var_idx = self.NMM.get_index_multiple(img_indexes=data['index'])
                    prior_var = torch.from_numpy(prior_var).float()
                    # Important Order for loss var needs to get close to emp_var
                    noise_loss = self.NMM.loss(prior_var, emp_var)

                # Backprogation
                self.optim.zero_grad()
                # Total loss computed (Eq. 2)
                total_loss = pred_loss + 0.01 * noise_loss # lambda: 0.01
                total_loss.backward()
                self.optim.step()

                # Save batch losses
                train_count += self.batch_size
                train_pred_loss += to_numpy(pred_loss) * self.batch_size
                if epoch <= 20:
                    train_noise_loss += noise_loss * self.batch_size
                else:
                    train_noise_loss += to_numpy(noise_loss) * self.batch_size

                # Compute metrics
                y_pred_t = self.SPM(original_x)['out']
                y_pred_t = to_numpy(y_pred_t)
                y = to_numpy(y)
                metrics['mae'] += mae(y_pred_t, y) * self.batch_size
                metrics['precision'] += precision(y_pred_t, y) * self.batch_size
                metrics['recall'] += recall(y_pred_t, y) * self.batch_size

                # Update tdqm
                train_bar.set_description(f'Train Epoch [{epoch}/{self.epochs}] Loss: { (train_noise_loss + train_pred_loss) / train_count:.4f} Noise: {train_noise_loss/train_count:.4f} Pred: {train_pred_loss / train_count:.4f}')

                # Remove data from GPU memory
                del x, y, y_noise, y_pred, y_pred_t, pred, data
                torch.cuda.empty_cache()
        
        # Prepare statistics
        total_train_loss = (train_noise_loss + train_pred_loss) / train_count
        train_noise_loss /= train_count
        train_pred_loss /= train_count
        metrics['mae'] /= train_count
        metrics['precision'] /= train_count
        metrics['recall'] /= train_count
        metrics['f1'] = f1(metrics['precision'], metrics['recall'], beta2=0.3)

        return total_train_loss, train_pred_loss, train_noise_loss, metrics['mae'], metrics['precision'], metrics['recall'], metrics['f1']

    def test_final(self, epoch):
        # Test SPM
        self.SPM.eval()

        # Test statistics
        test_count = 0
        metrics = {'mae': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Run Batch
        with torch.no_grad():
            with tqdm(self.testset) as test_bar:
                for batch_idx, data in enumerate(test_bar):
                    # Prepare data
                    x = data['image'].to(self.device)
                    y = data['label'].to(self.device)
                    y_pred = self.SPM(x)['out']

                    # Save batch losses
                    test_count += self.batch_size

                    # Compute metrics
                    y_pred = to_numpy(y_pred)
                    y = to_numpy(y)
                    metrics['mae'] += mae(y_pred, y) * self.batch_size
                    metrics['precision'] += precision(y_pred, y) * self.batch_size
                    metrics['recall'] += recall(y_pred, y) * self.batch_size
                    metrics['f1'] = f1(metrics['precision'] / test_count, metrics['recall'] / test_count, beta2=0.3)

                    # Update tdqm
                    F1 = metrics['f1']
                    MAE = metrics['mae']
                    test_bar.set_description(f'Test: F1-score: {F1:.4f} MAE: {MAE / test_count:.4f}')
        
        # Save statistcs
        metrics['mae'] /= test_count
        metrics['precision'] /= test_count
        metrics['recall'] /= test_count
        df = pd.DataFrame(data=metrics, index=[0])
        df.to_csv(f'output/{self.experiment}/{self.optim.__name__}/{self.scheduler.__name__}/{self.optim.__name__}_{self.scheduler.__name__}_{self.epochs}_test_{epoch}_stats.csv', index_label='epoch')
        
        return True
