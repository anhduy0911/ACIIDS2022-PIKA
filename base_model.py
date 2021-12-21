import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data.pill_dataset import PillDataset
import config as CFG
from models.modules import ImageEncoder
from utils.metrics import MetricLogger
class BaseModel:
    def __init__(self, args):
        """
        Model wrapper for all models
        """
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print("CUDA status: ", args.cuda)

        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.train_dataset, self.test_dataset = PillDataset(CFG.train_folder), PillDataset(CFG.test_folder, mode='test')

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

        self.model = ImageEncoder(image_num_classes=CFG.n_class)
        self.model.to(self.device)
        print(self.model)

    def train(self):
        """
        Train the model
        """
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        logger = MetricLogger(args=self.args, tags=['baseline', 'train'])

        for epoch in tqdm(range(10)):
            running_loss = 0.0
            running_corrects = 0
            sample_len = 0
            for x, y in self.train_dataset:
                x = x.to(self.device)
                sample_len += y.shape[0]
                y = y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                _, y_pred = torch.max(outputs, 1)
                
                logger.update(y_pred=y_pred, targets=y)
                # print(y_pred)
                # print(y)
                loss = loss_func(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(y_pred == y)
            
            epoch_loss = running_loss / sample_len
            epoch_acc = running_corrects.double() / sample_len
            logger.log_metrics(epoch_loss, step=epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_acc))


    def evaluate(self):
        """
        Evaluate the model
        """
        
        loss_func = torch.nn.CrossEntropyLoss()
        
        self.model.evaluate()
        logger = MetricLogger(args=self.args, tags=['baseline', 'test'])

        running_loss = 0.0
        running_corrects = 0
        sample_len = 0

        for x, y in self.train_dataset:
            x = x.to(self.device)
            sample_len += y.shape[0]
            y = y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(x)
            _, y_pred = torch.max(outputs, 1)
            
            logger.update(y_pred=y_pred, targets=y)
            # print(y_pred)
            # print(y)
            loss = loss_func(outputs, y)
            
            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(y_pred == y)
            
        epoch_loss = running_loss / sample_len
        epoch_acc = running_corrects.double() / sample_len
        logger.log_metrics(epoch_loss, step=1)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def save(self):
        """
        Save the model to a file
        """

        torch.save(self.model.state_dict(), 'logs/checkpoints/baseline.pt')
    
    def load(self):
        """
        Load the model
        """
        self.model.load_state_dict('logs/checkpoints/baseline.pt')

    def __str__(self):
        """
        Return a string representation of the model
        """
        return "[{}] ({})".format(self.__class__.__name__, self.model)