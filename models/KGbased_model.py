import torch
import torch.nn as nn
from models.modules import ImageEncoder
import config as CFG

class KGBasedModel(nn.Module):
    def __init__(self, hidden_size=CFG.g_embedding_features, num_class=CFG.n_class):
        super().__init__()

        self.backbone = self.setup_backborn_model()
        
        hidden_visual_features = self.backbone.visual_features

        self.pseudo_classifier = nn.Linear(hidden_visual_features, num_class)
        self.projection = nn.Linear(hidden_visual_features, hidden_size)

        self.attention_dense = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_class)

    def setup_backborn_model(self):
        model = ImageEncoder()
        model.load_state_dict(torch.load(CFG.backbone_path))
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def forward(self, x, g_embedding):
        visual_embedding, pseudo_classifier_output = self.backbone(x)
        
        mapped_visual_embedding = self.projection(visual_embedding)
        # print(mapped_visual_embedding.shape)
        condensed_graph_embedding = torch.mm(pseudo_classifier_output, g_embedding)
        # print(condensed_graph_embedding.shape)
        # context attention module
        scores = torch.mm(mapped_visual_embedding, condensed_graph_embedding.t())
        # print(scores.shape)
        distribution = nn.Softmax(dim=-1)(scores)
        # print(distribution.shape)
        context_val = torch.mm(distribution, mapped_visual_embedding)
        # print(context_val.shape)
        context_and_visual_vec = torch.cat([context_val, mapped_visual_embedding], dim=-1)
        # print(context_and_visual_vec.shape)
        attention_vec = nn.Tanh()(self.attention_dense(context_and_visual_vec))
        # print(attention_vec.shape)

        output = self.classifier(attention_vec)
        return pseudo_classifier_output, mapped_visual_embedding, output


if __name__ == '__main__':
    kg_based_model = KGBasedModel()
    x = torch.randn(32, 3, 224, 224)
    graph_embedding = torch.randn(89, CFG.g_embedding_features)
    output = kg_based_model(x, graph_embedding)
    print(output.shape)