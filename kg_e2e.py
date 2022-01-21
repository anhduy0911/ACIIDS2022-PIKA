import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data.pill_dataset import PillFolder
import config as CFG
from models.KGBased_e2emodel import KGBasedModel
from utils.metrics import MetricLogger
from utils.utils import EarlyStopping
from data.graph_data import build_data

class KGPillRecognitionModel:
    def __init__(self, args):
        """
        Model wrapper for all models
        """
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print("CUDA status: ", args.cuda)

        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.es = EarlyStopping(CFG.early_stop)

        self.train_dataset, self.test_dataset = PillFolder(CFG.train_folder_new), PillFolder(CFG.test_folder_new, mode='test')
        self.train_loader, self.test_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True), DataLoader(self.test_dataset, batch_size=args.v_batch_size, shuffle=False)

        self.g_embedding = build_data()
        self.model = KGBasedModel()

        self.model.to(self.device)
        self.g_embedding = self.g_embedding.to(self.device)
        # print(self.model)

    def train(self):
        """
        Train the model
        """
        # import time
        categorical_func_1 = torch.nn.CrossEntropyLoss()
        # categorical_func_2 = torch.nn.CrossEntropyLoss()
        domain_linkage_func = torch.nn.CosineEmbeddingLoss()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        # optimizer_projection = torch.optim.AdamW(self.model.projection.parameters(), lr=0.001)
        # start_time = time.time()
        self.model.train()
        logger = MetricLogger(args=self.args, tags=['train'])
        prev_val_acc = 0
        for epoch in tqdm(range(self.args.epochs)):
            running_loss = 0.0
            running_corrects = 0
            sample_len = 0

            for x, y, _ in self.train_loader:
                # x, y, g = self.train_dataset[i]
                bs = y.shape[0]
                sample_len += bs
                # print(f'1: {time.time() - start_time}')
                # start_time = time.time()
                x = x.to(self.device)
                y = y.to(self.device)
                # print(f'2: {time.time() - start_time}')
                # start_time = time.time()
                optimizer.zero_grad()
                # optimizer_projection.zero_grad()
                mapped_ebd, g_ebd, outputs = self.model(x, self.g_embedding)

                g_ebd_target = g_ebd[y]
                # print(f'3: {time.time() - start_time}')
                # start_time = time.time()
                _, y_pred = torch.max(outputs, 1)
                # print(f'4: {time.time() - start_time}')
                # start_time = time.time()
                logger.update(preds=y_pred, targets=y, conf_scores=outputs)

                closs_1 = categorical_func_1(outputs, y)
                # closs_1.backward()
                # optimizer.step()
                # closs_2 = categorical_func_2(pseudo_outputs, y)
                dloss = domain_linkage_func(mapped_ebd, g_ebd_target, torch.ones(bs).to(self.device))
                # dloss.backward()
                # optimizer_projection.step()
                # print(f'5: {time.time() - start_time}')
                # start_time = time.time()
                total_loss = closs_1 + 0.1 * dloss
                total_loss.backward()
                optimizer.step()
                # print(f'6: {time.time() - start_time}')
                # start_time = time.time()
                running_loss += total_loss.item() * x.size(0)
                running_corrects += torch.sum(y_pred == y)

            sample_eval_len = 0
            runn_acc_val = 0
            for x, y, _ in self.test_loader:
                sample_eval_len += x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                _, _, outputs = self.model(x, self.g_embedding)
                _, y_pred = torch.max(outputs, 1)
                
                runn_acc_val += torch.sum(y_pred == y)
            
            epoch_acc_val = runn_acc_val.double() / sample_eval_len

            epoch_loss = running_loss / sample_len
            epoch_acc = running_corrects.double() / sample_len
            logger.log_metrics(epoch_loss, step=epoch, val_acc=epoch_acc_val)
            print('{} Loss: {:.4f} Acc: {:.4f} Val_Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_acc, epoch_acc_val))
        
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
        logger = MetricLogger(args=self.args, tags=['test'])

        running_loss = 0.0
        running_corrects = 0
        sample_len = 0
        # cnt = 0
        for x, y, _ in self.test_loader:
            sample_len += y.shape[0]

            x = x.to(self.device)
            y = y.to(self.device)
            # print(cnt)
            _, _, outputs = self.model(x, self.g_embedding)
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

    def save_cpu(self):
        """
        Save the model on CPU
        """
        self.load(best=True)
        self.model.cpu()
        self.save(device='cpu')

    def save(self, best=False, device='cuda'):
        """
        Save the model to a file
        """
        if not best:
            if device == 'cuda':
                torch.save(self.model.state_dict(), 'logs/checkpoints/' + self.args.name + '.pt')
            else:
                torch.save(self.model.state_dict(), 'logs/checkpoints/' + self.args.name + '_cpu.pt')
        else:
            if device == 'cuda':
                torch.save(self.model.state_dict(), 'logs/checkpoints/' + self.args.name + '_best.pt')
            else:
                torch.save(self.model.state_dict(), 'logs/checkpoints/' + self.args.name + '_cpu_best.pt')

    def load(self, best=False, device='cuda'):
        """
        Load the model
        """
        if not best:
            if device == 'cuda':
                self.model.load_state_dict(torch.load('logs/checkpoints/' + self.args.name + '.pt'))
            else:
                self.model.load_state_dict(torch.load('logs/checkpoints/' + self.args.name + '_cpu.pt'))
        else:
            if device == 'cuda':
                self.model.load_state_dict(torch.load('logs/checkpoints/' + self.args.name + '_best.pt'))
            else:
                self.model.load_state_dict(torch.load('logs/checkpoints/' + self.args.name + '_cpu_best.pt'))
                
    def __str__(self):
        """
        Return a string representation of the model
        """
        return "[{}] ({})".format(self.__class__.__name__, self.model)
