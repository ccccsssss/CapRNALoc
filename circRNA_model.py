import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,matthews_corrcoef,f1_score
import pdb
USE_CUDA = True


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=8, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
                          for _ in range(num_capsules)])

    def forward(self, x):

        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)

        u = u.view(x.size(0), 8*24*24 , -1)


        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=2, num_routes=8*24*24, in_channels=8, out_channels=32):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.num_routes = num_routes

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):

        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)


        W = self.W.repeat(batch_size, 1, 1, 1, 1).to(x.device)

        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1, device=x.device)


        num_iterations = 3
        for iteration in range(num_iterations):

            c_ij = F.softmax(b_ij, dim=1)

            c_ij = c_ij.unsqueeze(4)


            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)


            if iteration < num_iterations - 1:

                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))

                b_ij = b_ij + a_ij.squeeze(4)


        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.reconstraction_layers = nn.Sequential(
            nn.Linear(32 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2*64*64),
            nn.Sigmoid()

        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.reconstraction_layers[4]:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes,dim=1)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(2))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

        reconstractions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstractions = reconstractions.view(-1, 2, 64, 64)

        return reconstractions, masked


class CapsNet(nn.Module):
    def __init__(self,Primary_capsule_num=8,in_channels=1):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer(in_channels=in_channels)
        self.primary_capsules = PrimaryCaps(num_capsules=Primary_capsule_num)
        self.digit_capsules = DigitCaps(in_channels=Primary_capsule_num)
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        conv_out = self.conv_layer(data)
        primary_out = self.primary_capsules(conv_out)
        digit_out = self.digit_capsules(primary_out)
        output = digit_out

        reconstractions, masked = self.decoder(output, data)
        return output, reconstractions, masked, primary_out, conv_out

    def loss(self, data, x, target, reconstractions):
        return self.margin_loss(x, target) + self.reconstraction_loss(data, reconstractions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstraction_loss(self, data, reconstractions):
        loss = self.mse_loss(reconstractions.view(reconstractions.size(0), -1), data.view(reconstractions.size(0), -1))
        return loss * 0.0005
