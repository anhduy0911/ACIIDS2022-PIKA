import torch.nn as nn
from torchvision import models
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

