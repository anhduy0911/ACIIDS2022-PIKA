import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import wandb
import config as CFG
import torch.nn.functional as F
from torch import nn
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

class MetricLogger:
    def __init__(self, project_name='KGbased_PillRecognition', args=None, tags=None):
        self.run = wandb.init(project_name, entity='aiotlab', config=args, tags=tags, group='DuyNA')
        self.type = '_'.join(tags)
        self.train = tags[1] == 'train'
        self.metrics = {}
        self.preds = []
        self.targets = []
        self.target_conf_scores = []

    def calculate_metrics(self):
        preds = torch.cat(self.preds).cpu().detach().numpy()
        targets = torch.cat(self.targets).cpu().detach().numpy()
        
        def walk_cooccurence():
            pred_ls = preds.tolist()
            targ_ls = targets.tolist()
            n_per_class = {}
            classify_matr = np.zeros((CFG.n_class, CFG.n_class))
            
            for idx, lab in enumerate(targ_ls):
                n_per_class[lab] = n_per_class.get(lab, 0) + 1
                p = pred_ls[idx]
                classify_matr[lab, p] += 1
            
            sns.heatmap(classify_matr, linewidths=0.5)
            plt.savefig(CFG.log_dir_run + self.type + '.png')
            
            np.save(CFG.log_dir_run + self.type + '.npy', classify_matr)
            for k, v in n_per_class.items():
                if v != 0:
                    print('\n' + f'class {k}: {v}')

        if not self.train:
            walk_cooccurence()
        
        acc = accuracy_score(targets, preds)
        self.metrics['accuracy'] = acc
        # ap = average_precision_score(target_conf_scores, conf_scores)
        # ap = self.calculate_average_precision()
        # self.metrics['AP'] = ap

        report = classification_report(targets, preds, output_dict=True, zero_division=0)

        dict_avg = report['weighted avg']
        for k in dict_avg.keys():
            self.metrics[k] = dict_avg[k]
        
        json_report = json.dumps(report)
        with open(CFG.log_dir_run + self.type, 'w') as f:
            f.write(json_report)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, conf_scores: torch.Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)
        self.target_conf_scores.append(conf_scores)

    def reset(self):
        self.metrics.clear()
        self.preds.clear()
        self.targets.clear()
        self.target_conf_scores.clear()


    def log_metrics(self, loss, step, val_acc=None):
        self.metrics['loss'] = loss
        if val_acc:
            self.metrics['val_acc'] = val_acc
        self.calculate_metrics()
        self.run.log(data=self.metrics, step=step)
        self.reset()

    def calculate_average_precision(self):
        conf_scores = torch.cat(self.target_conf_scores).cpu().detach().numpy()
        targets = torch.cat(self.targets).cpu().detach().numpy()
        targets = label_binarize(targets, classes=range(CFG.n_class))
        
        ap = average_precision_score(targets, conf_scores)
        return ap
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

    c = np.zeros((100,89))
    c[np.arange(100), np.random.randint(0, 89, size=100)] = 1
    d = np.random.rand(100,89)
    print(average_precision_score(a, b))
    # print(classification_report(a, b, zero_division=0, output_dict=True))
    # print(accuracy_score(a, b))


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