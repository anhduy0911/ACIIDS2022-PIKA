import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import argparse
from data.pill_dataset_v2 import PillDataset
from models import *
import config as CFG
from models.kssnet import KSSNet

def train(args, model, train_loader, test_loader):
    criterion = nn.BCELoss().to(device)
    #criterion = nn.MultiLabelSoftMarginLoss()
    #optimizer = torch.optim.SGD(model.parameters(),
    #                            lr=0.1,
    #                            momentum=0.9,
    #                            weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    for epoch in range(20):
        for idx, ((imgs, img_names, word_embedding), targets) in tqdm(enumerate(train_loader)):
            imgs, targets = imgs.to(device), targets.to(device)
            #import pdb
            #pdb.set_trace()
            #predicts = model(imgs, word_embedding)
            predicts = model(imgs)
            #print(predicts)
            loss = criterion(predicts, targets)
            #print('Epoch: {} | loss: {:.2f} |'.format(epoch+1, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if isinstance(model, nn.DataParallel):
            if not os.path.exists(os.path.dirname(args.ckpt)):
                os.makedirs(os.path.dirname(args.ckpt))
            torch.save(model.module, args.ckpt)
        else:
            torch.save(model, args.ckpt)
        if epoch % args.eval_freq == 0:
            mAP = test(model, test_loader)
            print("Epoch: {} | mAP: {}".format(epoch, mAP))

def mean_ap(scores, targets):
    if scores.numel() == 0:
        return 0
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0)).float()
    for k in range(scores.size(1)):
        score = scores[:, k]
        target = targets[:, k]
        #print(score.shape)
        #print(target.shape)
        ap[k] = average_precision(score, target)
    #print(ap)
    return ap

def average_precision(output, target, difficult_examples=True):
    #import pdb
    #pdb.set_trace()
    sorted, indices = torch.sort(output, dim=0, descending=True)
    #import pdb
    #pdb.set_trace()
    pos_count = 0.
    total_count = 0.
    precision_at_i = 0.
    for i in indices:
        label = target[i]
        #print(label)
        #if label == 0:
        #    continue
        if label == 1:
            pos_count += 1
        total_count += 1
        if label == 1:
            precision_at_i += pos_count / total_count
    #print(precision_at_i)
    #print(pos_count, total_count)
    if pos_count == 0:
        return 0
    precision_at_i /= pos_count
    #print(pos_count)
    return precision_at_i


def test(model, test_loader):
    predicts = []
    targets = []
    for idx, ((imgs, img_names, word_embedding),labels) in tqdm(enumerate(test_loader)):
        imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            #predict = model(imgs, word_embedding)
            predict = model(imgs)
        #print(predict)
        #print(labels)
        targets.append(labels)
        predicts.append(predict)

    predicts = torch.cat(predicts, dim=0)
    targets = torch.cat(targets, dim=0)
    #import pdb
    #pdb.set_trace()
    ap = mean_ap(predicts, targets)
    return torch.mean(ap).item()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multi-Label-Classification Task')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', dest='batch_size')
    parser.add_argument('--val-batch-size', type=int, default=32, metavar='N', dest='v_batch_size')
    parser.add_argument('--model', type=str, default='kssnet')

    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_loader, test_loader = PillDataset(CFG.train_folder_v2, args.batch_size, CFG.g_embedding_condensed, 'train'), PillDataset(CFG.test_folder, args.v_batch_size, CFG.g_embedding_condensed, 'test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Mode: {}".format(device))
    if args.model == 'kssnet':
        model = KSSNet().to(device)
    
    if args.test:
        model.load_state_dict(torch.load(args.ckpt).state_dict())
        model = nn.DataParallel(model)
        test(model, test_loader)
    else:
        model = nn.DataParallel(model)
        train(args, model, train_loader, test_loader)
