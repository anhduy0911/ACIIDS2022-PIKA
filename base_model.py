import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data.pill_dataset import PillFolder
import config as CFG
from models.modules import ImageEncoder
from utils.metrics import MetricLogger
from utils.utils import EarlyStopping
class BaseModel:
    def __init__(self, args):
        """
        Model wrapper for all models
        """
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print("CUDA status: ", args.cuda)

        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.train_dataset, self.test_dataset = PillFolder(CFG.train_folder_new), PillFolder(CFG.test_folder_new, mode='test')
        self.train_loader, self.test_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True), DataLoader(self.test_dataset, batch_size=args.v_batch_size, shuffle=False)

        self.es = EarlyStopping(patience=50)

        self.model = ImageEncoder()
        self.model.to(self.device)
        print(self.model)

    def train(self):
        """
        Train the model
        """
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.model.train()
        logger = MetricLogger(args=self.args, tags=['baseline', 'train'])
        prev_val_acc = 0
        for epoch in tqdm(range(self.args.epochs)):
            running_loss = 0.0
            running_corrects = 0
            sample_len = 0

            for x, y, _  in self.train_loader:
                x = x.to(self.device)
                sample_len += y.shape[0]
                y = y.to(self.device)
                
                optimizer.zero_grad()
                _, outputs = self.model(x)
                _, y_pred = torch.max(outputs, 1)
                
                logger.update(preds=y_pred, targets=y, conf_scores=outputs)
                # print(y_pred)
                # print(y)
                loss = loss_func(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(y_pred == y)
            
            sample_eval_len = 0
            runn_acc_val = 0
            for x, y, _ in self.test_loader:
                sample_eval_len += x.shape[0]

                x = x.to(self.device)
                y = y.to(self.device)
                _, outputs = self.model(x)
                _, y_pred = torch.max(outputs, 1)
                
                runn_acc_val += torch.sum(y_pred == y)
            
            epoch_acc_val = runn_acc_val.double() / sample_eval_len

            epoch_loss = running_loss / sample_len
            epoch_acc = running_corrects.double() / sample_len
            logger.log_metrics(epoch_loss, val_acc=epoch_acc_val, step=epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_acc))

            self.es(-epoch_acc_val)
            if self.es.early_stop:
                print("Early stopping...")
                self.save()
                break

            if prev_val_acc < epoch_acc_val:
                print("Save Best...")
                self.save(best=True)
                prev_val_acc = epoch_acc_val
            else:
                print("Save Normal...")
                self.save()

    def evaluate(self):
        """
        Evaluate the model
        """
        self.load(best=True, device='cuda' if self.args.cuda else 'cpu')
        loss_func = torch.nn.CrossEntropyLoss()
        
        self.model.eval()
        logger = MetricLogger(args=self.args, tags=['baseline', 'test'])

        running_loss = 0.0
        running_corrects = 0
        sample_len = 0

        # cnt = 0
        print('Start Evaluating...')
        print(f'Batch_size: {self.args.v_batch_size}')
        for x, y, _ in self.test_loader:
            sample_len += y.shape[0]

            x = x.to(self.device)
            y = y.to(self.device)
            # print(cnt)
            _, outputs = self.model(x)
            _, y_pred = torch.max(outputs, 1)
            
            logger.update(preds=y_pred, targets=y, conf_scores=outputs)
            # print(y_pred)
            # print(y)
            loss = loss_func(outputs, y)
            
            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(y_pred == y)
            
        epoch_loss = running_loss / sample_len
        epoch_acc = running_corrects.double() / sample_len
        logger.log_metrics(epoch_loss, step=1)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def test(self):
        for x, y, _ in self.test_dataset:
            print(x.shape)
            print(y.shape)

    def save_cpu(self, best=True):
        """
        Save the model on CPU
        """
        self.load(best=best)
        self.model.cpu()
        self.save(device='cpu', best=best)

    def save(self, best=False, device='cuda'):
        """
        Save the model to a file
        """
        if not best:
            if device == 'cuda':
                torch.save(self.model.state_dict(), 'logs/checkpoints/baseline.pt')
            else:
                torch.save(self.model.state_dict(), 'logs/checkpoints/baseline_cpu.pt')
        else:
            if device == 'cuda':
                torch.save(self.model.state_dict(), 'logs/checkpoints/baseline_best.pt')
            else:
                torch.save(self.model.state_dict(), 'logs/checkpoints/baseline_cpu_best.pt')

    def load(self, best=False, device='cuda'):
        """
        Load the model
        """
        if not best:
            if device == 'cuda':
                self.model.load_state_dict(torch.load('logs/checkpoints/baseline.pt'))
            else:
                self.model.load_state_dict(torch.load('logs/checkpoints/baseline_cpu.pt'))
        else:
            if device == 'cuda':
                self.model.load_state_dict(torch.load('logs/checkpoints/baseline_best.pt'))
            else:
                self.model.load_state_dict(torch.load('logs/checkpoints/baseline_cpu_best.pt'))

    def __str__(self):
        """
        Return a string representation of the model
        """
        return "[{}] ({})".format(self.__class__.__name__, self.model)
