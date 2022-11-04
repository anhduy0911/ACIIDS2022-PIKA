import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torchvision import models
import torch.nn.functional as F
import config as CFG
import torch
from torch.nn import Parameter

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
            if model_name == "resnet18":
                self.model = models.resnet18(pretrained=pretrained)
            elif model_name == "resnet34":
                self.model = models.resnet34(pretrained=pretrained)
            elif model_name == "resnet50":
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

class GCN(nn.Module):
    def __init__(self, n_class=CFG.n_class) -> None:
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(CFG.n_class, 32)
        self.relu = nn.ReLU()
        self.conv2 = pyg_nn.GCNConv(32, n_class)
    
    def forward(self, data):
        x, edge_idx, edge_w = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_idx, edge_w)
        x = self.relu(x)
        x = self.conv2(x, edge_idx, edge_w)
        
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Critic(nn.Module):
    def __init__(self, latent_dim=64):
        super(Critic, self).__init__()
        self.latent_dim = latent_dim
        self.double_latent_dim = 2 * latent_dim
        self.half_latent_dim = self.latent_dim // 2
        
        self.layer1 = SpectralNorm(nn.Linear(self.double_latent_dim, self.latent_dim))
        self.layer2 = SpectralNorm(nn.Linear(self.latent_dim, self.half_latent_dim))
        self.layer3 = SpectralNorm(nn.Linear(self.half_latent_dim, 1))
        
    def forward(self, x, g):
        inp = torch.cat((x, g), dim=1)
        inp = nn.LeakyReLU()(self.layer1(inp))
        inp = nn.LeakyReLU()(self.layer2(inp))
        
        return nn.Sigmoid()(self.layer3(inp))
    
class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        '''
        Module return the alignment scores
        '''
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Linear(hidden_size, 1, bias=False)
  
    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)
        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1)
        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = decoder_hidden + encoder_outputs.unsqueeze(1)
            out = torch.tanh(self.fc(out))
            
            return self.weight(out).squeeze(-1)