import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torchvision import models
import torch.nn.functional as F
import config as CFG
import torch
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name=CFG.image_model_name, pretrained=CFG.image_pretrained, trainable=CFG.image_trainable, image_num_classes=CFG.n_class):
        super(ImageEncoder, self).__init__()
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.visual_features = self.model.classifier[-1].in_features
            self.model.classifier = self.model.classifier[:-1]
            
            for param in self.model.parameters():
                param.requires_grad = trainable
                
            self.img_num_classes = image_num_classes
            self.classifier = nn.Linear(self.visual_features, image_num_classes)
        else:
            if model_name == "resnet50":
                self.model = models.resnet50(pretrained=pretrained)
            elif model_name == 'resnet101':
                self.model = models.resnet101(pretrained=pretrained)
            elif model_name == 'resnet152':
                self.model = models.resnet152(pretrained=pretrained)

            for param in self.model.parameters():
                param.requires_grad = trainable
            self.img_num_classes = image_num_classes
            
            self.visual_features = self.model.fc.in_features
            # self.half_visual_features = self.visual_features // 2
            self.model.fc = nn.Identity()
            self.classifier = nn.Linear(self.visual_features, image_num_classes)
            # self.classifier = nn.Sequential( nn.Linear(self.visual_features, image_num_classes) )


    def forward(self, x):
        visual_embedding = self.model(x)
        y = self.classifier(visual_embedding)
        return visual_embedding, y

class ImageEncoderHancraft(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name=CFG.image_model_name, pretrained=CFG.image_pretrained, trainable=CFG.image_trainable, image_num_classes=CFG.n_class):
        super(ImageEncoderHancraft, self).__init__()
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            self.model_c = models.resnet50(pretrained=pretrained)
            self.model_t = models.resnet50(pretrained=pretrained)
            
            self.visual_features = self.model.fc.in_features
            self.visual_mid_features = int(self.visual_features / 2)

            self.self_att = nn.MultiheadAttention(embed_dim=self.visual_features, num_heads=4)

            for param in self.model.parameters():
                param.requires_grad = trainable
            self.img_num_classes = image_num_classes
            

            self.model.fc = nn.Identity()
            self.model_c.fc = nn.Identity()
            self.model_t.fc = nn.Identity()

            self.classifier = nn.Sequential(
                nn.Linear(self.visual_features * 3, self.visual_mid_features),
                nn.ReLU(),
                nn.Linear(self.visual_mid_features, self.img_num_classes)
            ) 
    
    def forward(self, x, x_c, x_t):
        visual_embedding = self.model(x)
        visual_embedding_c = self.model(x_c)
        visual_embedding_t = self.model(x_t)

        # keys = torch.stack((visual_embedding, visual_embedding_c, visual_embedding_t), dim=0)
        # att_ve, _ = self.self_att(query=torch.unsqueeze(visual_embedding, 0), key=keys, value=keys)
        # att_ve = att_ve.squeeze()
        ve = torch.cat((visual_embedding, visual_embedding_c, visual_embedding_t), dim=-1)
        y = self.classifier(ve)

        return ve, y

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
