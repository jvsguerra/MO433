import torch
import torch.nn as nn
import numpy as np

class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        self.similarity_function = self._get_sim_func(use_cosine_similarity)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
    
    def _get_sim_func(self, use_cosine_similarity):
        '''similarity=(x1​⋅x2)/max(∥x1​∥2​⋅∥x2​∥2​,t)​​'''
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(zi, zj):
        """
        https://pytorch.org/docs/master/generated/torch.tensordot.html#torch.tensordot
        zi shape: (N, 1, C)
        zj shape: (1, C, 2N)
        lij shape: (N, 2N)
        """
        lij = torch.tensordot(zi.unsqueeze(1), zj.T.unsqueeze(0), dims=2)
        return lij
    
    def _cosine_simililarity(self, zi, zj):
        """
        https://pytorch.org/docs/master/generated/torch.nn.CosineSimilarity.html
        zi shape: (N, 1, C)
        zj shape: (1, C, 2N)
        lij shape: (N, 2N)
        """
        lij = self._cosine_similarity(zi.unsqueeze(1), zj.unsqueeze(0))
        return lij

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].reshape(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)