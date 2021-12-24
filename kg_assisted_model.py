import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data.pill_dataset import PillDataset
import config as CFG
from models.KGbased_model import KGBasedModel
from utils.metrics import MetricLogger


class KGPillRecognitionModel:
    def __init__(self, args):
        """
        Model wrapper for all models
        """
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print("CUDA status: ", args.cuda)

        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.train_dataset, self.test_dataset = PillDataset(CFG.train_folder, batch_size=args.batch_size), PillDataset(CFG.test_folder, batch_size=args.v_batch_size, mode='test')

        self.g_embedding = self.train_dataset.g_embedding.to(self.device)
        self.model = KGBasedModel()
        self.model.to(self.device)
        print(self.model)

    def train(self):
        """
        Train the model
        """
        # import time
        categorical_func_1 = torch.nn.CrossEntropyLoss()
        categorical_func_2 = torch.nn.CrossEntropyLoss()
        domain_linkage_func = torch.nn.CosineEmbeddingLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # start_time = time.time()
        # self.model.train()
        logger = MetricLogger(args=self.args, tags=['KG_v1', 'train'])
        for epoch in tqdm(range(self.args.epochs)):
            running_loss = 0.0
            running_corrects = 0
            sample_len = 0

            for i in range(len(self.train_dataset)):
                x, y, g = self.train_dataset[i]

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
                pseudo_outputs, mapped_ebd, outputs = self.model(x, self.g_embedding)
                # print(f'3: {time.time() - start_time}')
                # start_time = time.time()
                _, y_pred = torch.max(outputs, 1)
                # print(f'4: {time.time() - start_time}')
                # start_time = time.time()
                logger.update(preds=y_pred, targets=y)

                closs_1 = categorical_func_1(outputs, y)
                closs_2 = categorical_func_2(pseudo_outputs, y)
                dloss = domain_linkage_func(mapped_ebd, g, torch.ones(bs).to(self.device))
                # print(f'5: {time.time() - start_time}')
                # start_time = time.time()
                total_loss = closs_1 + 0.5 * closs_2 + dloss
                total_loss.backward()
                optimizer.step()
                # print(f'6: {time.time() - start_time}')
                # start_time = time.time()
                running_loss += total_loss.item() * x.size(0)
                running_corrects += torch.sum(y_pred == y)

            epoch_loss = running_loss / sample_len
            epoch_acc = running_corrects.double() / sample_len
            logger.log_metrics(epoch_loss, step=epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_acc))
        
        self.save()

    def evaluate(self):
        """
        Evaluate the model
        """
        self.load()
        loss_func = torch.nn.CrossEntropyLoss()
        
        self.model.eval()
        logger = MetricLogger(args=self.args, tags=['KG_v1', 'test'])

        running_loss = 0.0
        running_corrects = 0
        sample_len = 0

        # cnt = 0
        for x, y, _ in self.test_dataset:
            x = x.to(self.device)
            sample_len += y.shape[0]
            y = y.to(self.device)
            
            # print(cnt)
            outputs = self.model(x)
            _, y_pred = torch.max(outputs, 1)
            
            logger.update(preds=y_pred, targets=y)
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

    def save(self):
        """
        Save the model to a file
        """

        torch.save(self.model.state_dict(), 'logs/checkpoints/kg_v1.pt')
    
    def load(self):
        """
        Load the model
        """
        self.model.load_state_dict(torch.load('logs/checkpoints/kg_v1.pt'))

    def __str__(self):
        """
        Return a string representation of the model
        """
        return "[{}] ({})".format(self.__class__.__name__, self.model)
