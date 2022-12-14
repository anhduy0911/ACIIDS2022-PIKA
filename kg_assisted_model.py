import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data.pill_dataset_v2 import PillDataset

import config as CFG
from models.KGbased_model import KGBasedModel
from models.modules import Critic
from utils.metrics import MetricLogger
from utils.utils import *

class KGPillRecognitionModel:
    def __init__(self, args):
        """
        Model wrapper for all models
        """
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print("CUDA status: ", args.cuda)

        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.es = EarlyStopping(args.patience)

        # self.train_dataset, self.test_dataset = PillFolder(CFG.train_folder_fewshot), PillFolder(CFG.test_folder_new, mode='test')
        # self.train_loader, self.test_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True), DataLoader(self.test_dataset, batch_size=args.v_batch_size, shuffle=False)
        self.train_loader, self.test_loader = PillDataset(CFG.train_folder_v2, args.batch_size, args.g_emd_path, 'train', exclude_path=args.exclude_path), PillDataset(CFG.test_folder, args.v_batch_size, CFG.g_embedding_condensed, 'test', exclude_path=args.exclude_path)

        self.g_embedding = self.train_loader.g_embedding_np.to(self.device)
        # self.g_embedding = self.train_dataset.g_embedding_np.to(self.device)
        self.model = KGBasedModel(backbone_name=args.backbone)
        
        if self.args.loss == 'wd':
            self.critic = Critic(CFG.g_embedding_features)
            self.critic.to(self.device)
            self.buffer = MemoryBuffer(CFG.n_class, CFG.buffer_size)
            
        self.model.to(self.device)
        # print(self.model)

    def train(self):
        """
        Train the model
        """
        # import time
        categorical_func_1 = torch.nn.CrossEntropyLoss()
        # categorical_func_2 = torch.nn.CrossEntropyLoss()
        # domain_linkage_func = torch.nn.CosineEmbeddingLoss()
        if self.args.loss == 'js':
            domain_linkage_func = JS_loss_fast_compute
        elif self.args.loss == 'kl':
            domain_linkage_func = KL_divergence
        else:
            critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        # optimizer_projection = torch.optim.AdamW(self.model.projection.parameters(), lr=0.001)
        # start_time = time.time()
        self.model.train()
        logger = MetricLogger(args=self.args, tags=['KG_deepwalk_w', 'train'])
        prev_val_acc = 0
        for epoch in tqdm(range(self.args.epochs)):
            running_loss = 0.0
            running_corrects = 0
            sample_len = 0

            for x, y, g in self.train_loader:
                # x, y, g = self.train_dataset[i]
                bs = y.shape[0]
                sample_len += bs
                # print(f'1: {time.time() - start_time}')
                # start_time = time.time()
                x = x.to(self.device)
                y = y.to(self.device)
                g = g.to(self.device)
                # print(f'2: {time.time() - start_time}')
                # start_time = time.time()
                optimizer.zero_grad()
                # optimizer_projection.zero_grad()
                _, mapped_ebd, outputs = self.model(x, self.g_embedding)
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
                # dloss = domain_linkage_func(mapped_ebd, g, torch.ones(bs).to(self.device))
                if self.args.loss == 'wd':
                    self.buffer.add(mapped_ebd, g, y)
                    loss_real = self.critic(mapped_ebd, g)
                    fake_samples = self.buffer.generate_fake_samples(y)
                    loss_fake = self.critic(fake_samples, g)
                    
                    dloss = - torch.abs(torch.mean(loss_real - loss_fake))
                else:
                    dloss = domain_linkage_func(mapped_ebd, g)
                
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

            if self.args.loss == 'wd':
                for i in range(5):
                    critic_optimizer.zero_grad()
                    real_samples, g, y = self.buffer.generate_real_samples(self.args.batch_size)
                    loss_real = self.critic(real_samples, g)
                    fake_samples = self.buffer.generate_fake_samples(y)
                    loss_fake = self.critic(fake_samples, g)
                    
                    loss_critic = - torch.abs(torch.mean(loss_real - loss_fake))
                    loss_critic.backward()
                    critic_optimizer.step()
                # print('CHECKPOINT')
            
            self.model.eval()
            sample_eval_len = 0
            runn_acc_val = 0
            for x, y, _ in self.test_loader:
                sample_eval_len += x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                _, _, outputs = self.model(x, self.g_embedding)
                _, y_pred = torch.max(outputs, 1)

                logger.update_val(preds=y_pred, targets=y)
                runn_acc_val += torch.sum(y_pred == y)
            
            epoch_acc_val = runn_acc_val.double() / sample_eval_len

            epoch_loss = running_loss / sample_len
            epoch_acc = running_corrects.double() / sample_len
            logger.log_metrics(epoch_loss, val_acc=epoch_acc_val, step=epoch)
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
        logger = MetricLogger(args=self.args, tags=['KG_deepwalk_w', 'test'])

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
            logger.update_val(preds=y_pred, targets=y)
            
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
