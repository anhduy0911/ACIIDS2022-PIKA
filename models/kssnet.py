import torchvision.models as models
from torch.nn import Parameter
#from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from utils.utils import weights_init
from data.graph_data import build_adj_matrix

class GraphConvolution(nn.Module):
    """
    	Source: "https://github.com/Megvii-Nanjing/ML-GCN/blob/master/models.py"
    """
    def __init__(self, in_features, out_features, num_classes=76, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.adj = build_adj_matrix()
        self.adj = nn.Parameter(torch.from_numpy(self.adj).float())

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        print(self.adj.size(), support.size())
        output = torch.matmul(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LC(nn.Module):
    def __init__(self, hid_dim, num_classes=76):
        super(LC, self).__init__()
        self.conv1x1 = nn.Conv2d(num_classes, hid_dim, kernel_size=1, stride=1, bias=False)
        self.conv1x1.apply(weights_init)

    def forward(self, x, E):
        N, C, H, W = x.size()
        original_x = x
        # [N, C, H, W] -> [N, HW, C]
        x = x.transpose(1,2).transpose(2,3).view(N, H*W, -1)
        # E = [N, C] -> [C, N]
        E = E.transpose(1, 0)
        E = torch.tanh(E)
        x = torch.matmul(x, E)
        x = x.view(N, H, W, -1)
        x = x.transpose(3, 1).transpose(2, 3)
        x = self.conv1x1(x)
        x = x + original_x
        return x


class KSSNet(nn.Module):
    def __init__(self, ):
        super(KSSNet, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.res_block1 = self.backbone.layer1
        self.res_block2 = self.backbone.layer2
        self.res_block3 = self.backbone.layer3
        self.res_block4 = self.backbone.layer4

        self.gcn1 = GraphConvolution(64, 256)
        self.gcn2 = GraphConvolution(256, 512)
        self.gcn3 = GraphConvolution(512, 1024)
        self.gcn4 = GraphConvolution(1024, 2048)

        self.lc1 = LC(256)
        self.lc2 = LC(512)
        self.lc3 = LC(1024)
        self.lc4 = LC(2048)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(2048, 80)

    def forward(self, x, word_embedding):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_block1(x)
        e = self.gcn1(word_embedding)
        x = self.lc1(x, e)
        e = F.leaky_relu(e, 0.2)

        x = self.res_block2(x)
        e = self.gcn2(e)
        x = self.lc2(x, e)
        e = F.leaky_relu(e, 0.2)

        x = self.res_block3(x)
        e = self.gcn3(e)
        x = self.lc3(x, e)
        e = F.leaky_relu(e, 0.2)

        x = self.res_block4(x)
        e = self.gcn4(e)
        x = self.lc4(x, e)
        #e = torch.sigmoid(e)

        feat = self.gap(x)
        x = feat.view(feat.size(0), -1)
        e = e.transpose(0,1)
        y = self.fc(x)
        y = torch.sigmoid(y)
        return y

if __name__ == "__main__":
    kssnet = KSSNet()
    x = torch.rand((32, 3, 224, 224))
    word_embedding = torch.rand((76, 64))
    out = kssnet(x,word_embedding)
