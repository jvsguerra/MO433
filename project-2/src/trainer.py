from tqdm import tqdm 
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.SimCLR import SimCLR
from nets.NTXentLoss import NTXentLoss
import utils.data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Debug:
from utils.lars import LARS

class Trainer(object):
    def __init__(self, feature_dim=128, temperature=0.5, batch_size=32, epochs=100, device='cuda:0', use_cos_sim=True):
        self.device = device
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_cos_sim = use_cos_sim

        # Prepare data
        self.trainset, self.bank, self.testset = self.prepare_data()


    def prepare_data(self):
        # Train Data
        train_data = utils.data.CIFAR10Pair(root='./input', train=True, transform=utils.data.train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Feature Bank
        feature_bank = utils.data.CIFAR10Pair(root='./input', train=True, transform=utils.data.test_transform, download=True)
        feature_bank = DataLoader(feature_bank, batch_size=self.batch_size, shuffle=False)

        # Test Data
        test_data = utils.data.CIFAR10Pair(root='./input', train=False, transform=utils.data.test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, feature_bank, test_loader

    def build(self, use_lars=True):
        # self.net = SimCLR(self.feature_dim).cuda()
        self.net = SimCLR(self.feature_dim).to(self.device)
        # Optimizer: LARS or Adam
        if use_lars:
            self.optim = LARS(self.net.parameters(), lr=1e-3, weight_decay=1e-6, eta=1e-5)
        else:
            self.optim = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-6)
        # Loss function
        self.loss_fn = NTXentLoss(device=self.device, batch_size=self.batch_size, temperature=self.temperature, use_cosine_similarity=self.use_cos_sim)

    def fit(self, base_dir='test'):
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@3': [], 'test_acc@5': []}
        base_name = f'{self.feature_dim}_{self.temperature}_{self.batch_size}_{self.epochs}'
        best = 0.0

        for epoch in range(1, self.epochs+1):
            print(f'\nEpoch: {epoch}')

            # Train step
            total_loss, total = self.train(epoch)
            train_loss = total_loss / total
            results['train_loss'].append(train_loss)

            # Feature Extraction and Test Step
            acc1, acc3, acc5 = self.test(epoch)
            results['test_acc@1'].append(acc1)
            results['test_acc@3'].append(acc3)
            results['test_acc@5'].append(acc5)

            # Statistics
            df = pd.DataFrame(data=results, index=range(1, epoch + 1))
            df.to_csv(f'output/{base_dir}/{base_name}_stats.csv', index_label='epoch')
            if acc1 > best:
                best = acc1
                torch.save(self.net.state_dict(), f'output/{base_dir}/{base_name}_model.pth')
    
    def train(self, epoch):
        self.net.train()
        total_loss, total = 0.0, 0

        with tqdm(self.trainset) as train_bar:
            for pos_1, pos_2, target in train_bar:
                # Send imgs to GPU
                # pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
                pos_1, pos_2 = pos_1.to(self.device, non_blocking=True), pos_2.to(self.device, non_blocking=True)
                
                # Get hi and zi
                h1, z1 = self.net(pos_1)
                h2, z2 = self.net(pos_2)

                # [2 * B, 2 * B]
                loss = self.loss_fn(z1, z2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total += self.batch_size
                total_loss += loss.item() * self.batch_size
                train_bar.set_description(f'Train Epoch: [{epoch}/{self.epochs}] Loss: {total_loss/total:.4f}')

        return total_loss, total

    def saveTargetsAndH(self):
        allH = np.load('input/models/h/h.npy')
        allTargets = np.load('input/models/h/targets.npy')

        with torch.no_grad():
            with tqdm(self.bank, desc='Feature extracting') as feature_bar:
                start = int(len(allH)/32)
                index = 0
                print(start)
                for data, _, target in feature_bar:

                    if index < start:
                        index+=1
                        continue
                    
                    h, z = self.net(data.to(self.device, non_blocking=True))
                    for indexData in range(len(h)):
                        allH = np.vstack([allH, torch.flatten(h[indexData]).numpy()])
                        allTargets = np.append(allTargets, target[indexData].numpy())

                    if index % 10 == 0:
                        print('salvando com index:', index)
                        currentH = np.array(allH)
                        currentTargets = np.array(allTargets)

                        np.save('input/models/h/h.npy', currentH)
                        np.save('input/models/h/targets.npy', currentTargets)
                    index+=1
                
                allH = np.array(allH)
                allTargets = np.array(allTargets)

                np.save('input/models/h/h.npy', allH)
                np.save('input/models/h/targets.npy', allTargets)

    def linearEvaluation(self):
        print("Loading h and targets:")
        h = np.load('input/models/h/h.npy')
        targets = np.load('input/models/h/targets.npy')

        print("Spliting in train and test:")
        X_train, X_test, y_train, y_test = train_test_split(h, targets, test_size=0.25, random_state=0)

        print("Training Logistic Regression:")
        # Linear Evaluation using Logistic Regression (Training)
        lrModel = LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=1.0)
        lrModel.fit(X_train, y_train)

        # Linear Evaluation using Logistic Regression (Testing)
        print("Linear Evaluation Accuracy Train set:", 100*lrModel.score(X_train, y_train), "%")
        print("Linear Evaluation Accuracy Test set:", 100*lrModel.score(X_test, y_test), "%")

    def test(self, epoch):
        self.net.eval()
        top1, top3, top5, total, feature_bank = 0.0, 0.0, 0.0, 0, []

        with torch.no_grad():

            with tqdm(self.bank, desc='Feature extracting') as feature_bar:
                for data, _, target in feature_bar:
                    # h, z = self.net(data.cuda(non_blocking=True))
                    h, z = self.net(data.to(self.device, non_blocking=True))
                    feature_bank.append(h)
                # [D, N]
                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
                # [N]
                feature_labels = torch.tensor(self.bank.dataset.targets, device=feature_bank.device)

            with tqdm(self.testset) as test_bar:
                for data, _, target in test_bar:
                    # data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    h, z = self.net(data)
                    total += data.size(0)

                    # compute cosine similarity between each feature vector and feature bank ---> [B, N]
                    k = 200
                    c = 10
                    sim_matrix = torch.mm(h, feature_bank)
                    # [B, K]
                    sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                    # [B, K]
                    sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                    sim_weight = (sim_weight / self.temperature).exp()

                    # counts for each class
                    one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
                    # [B*K, C]
                    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.reshape(-1, 1), value=1.0)
                    # weighted score ---> [B, C]
                    pred_scores = torch.sum(one_hot_label.reshape(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

                    pred_labels = pred_scores.argsort(dim=-1, descending=True)
                    top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top3 += torch.sum((pred_labels[:, :3] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                    test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@3:{:.2f}% Acc@5:{:.2f}%'.format(epoch, self.epochs, top1 / total * 100, top3 / total * 100, top5 / total * 100))

            return top1 / total * 100, top3 / total * 100, top5 / total * 100
