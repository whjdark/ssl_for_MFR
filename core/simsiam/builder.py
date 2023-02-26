'''
Author: whj
Date: 2022-02-27 17:03:34
LastEditors: whj
LastEditTime: 2022-03-04 20:10:19
Description: file content
'''

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, encoder_pretrain=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output classifier dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim)

        if encoder_pretrain:
            self.load_encoder_pretrain(encoder_pretrain)

        # build a 3-layer projector
        prev_dim = self.encoder.classifier.weight.shape[1]
        self.encoder.classifier = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.classifier,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.classifier[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def load_encoder_pretrain(self, pretrain_path):
        checkpoint = torch.load(pretrain_path)
        msg = self.encoder.load_state_dict(checkpoint, strict=False)
        print(msg)


