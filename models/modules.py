import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torchvision import models
import torch.nn.functional as F
import config as CFG

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name=CFG.image_model_name, pretrained=CFG.image_pretrained, trainable=CFG.image_trainable, image_num_classes=CFG.n_class):
        super(ImageEncoder, self).__init__()
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            for param in self.model.parameters():
                param.requires_grad = trainable
            self.img_num_classes = image_num_classes
            
            self.visual_features = self.model.fc.in_features

            self.model.fc = nn.Identity()
            self.classifier = nn.Linear(self.visual_features, image_num_classes)
    
    def forward(self, x):
        visual_embedding = self.model(x)
        y = self.classifier(visual_embedding)
        return visual_embedding, y


class GCN(nn.Module):
    def __init__(self, n_class=CFG.n_class) -> None:
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(1, 16)
        self.relu = nn.ReLU()
        self.conv2 = pyg_nn.GCNConv(16, n_class)
    
    def forward(self, data):
        x, edge_idx, edge_w = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_idx, edge_w)
        x = self.relu(x)
        x = self.conv2(x, edge_idx, edge_w)
    
        return x
