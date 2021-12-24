import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import wandb
import config as CFG
import torch.nn.functional as F
from torch import nn

class MetricLogger:
    def __init__(self, project_name='KGbasedPR', args=None, tags=None):
        self.run = wandb.init(project_name, entity='anhduy0911', config=args, tags=tags)
        self.metrics = {}
        self.preds = []
        self.targets = []

    def calculate_metrics(self):
        preds = torch.cat(self.preds).cpu().detach().numpy()
        targets = torch.cat(self.targets).cpu().detach().numpy()

        acc = accuracy_score(targets, preds)
        self.metrics['accuracy'] = acc
        dict_avg = classification_report(targets, preds, output_dict=True, zero_division=0)['weighted avg']
        for k in dict_avg.keys():
            self.metrics[k] = dict_avg[k]
        
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)

    def reset(self):
        self.metrics.clear()
        self.preds.clear()
        self.targets.clear()


    def log_metrics(self, loss, step):
        self.metrics['loss'] = loss
        self.calculate_metrics()
        self.run.log(data=self.metrics, step=step)
        self.reset()

 
class MetricTracker:
    def __init__(self, labels=None):
        super().__init__()
        self.target_names = labels
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds).cpu().numpy()
        targets = torch.cat(self.targets).cpu().numpy()
        return classification_report(targets, preds,
                                     target_names=self.target_names, zero_division=0)

    def reset(self):
        self.preds = []
        self.targets = []

def test_metric():
    a = torch.tensor([1,5,2], dtype=torch.long, device='cuda')
    b = torch.tensor([1,5,3], dtype=torch.long, device='cuda')
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()

    print(classification_report(a, b, zero_division=0, output_dict=True))
    print(accuracy_score(a, b))


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            
    def calc_cosinsimilarity(self, x1, x2):
        return self.cos(x1, x2)

    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_cosinsimilarity(anchor, positive)
        distance_negative = self.calc_cosinsimilarity(anchor, negative)

        losses = torch.relu(-distance_positive + distance_negative + self.margin)

        return losses.mean()

if __name__ == '__main__':
    test_metric()