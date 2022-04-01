import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List, Dict
import json
import config as CFG
import numpy as np
import cv2 as cv
import os
import pickle

class PillFolder(ImageFolder):
    def __init__(self, root, g_ebedding, class_to_idx, transform=None):
        super(PillFolder, self).__init__(root, transform=transform)
        self.class_to_idx = {key: class_to_idx[key] for key in self.classes}
        img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        self.imgs = self.make_dataset(self.root, self.class_to_idx, extensions=img_extensions)
        self.g_ebd = g_ebedding
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return sample, target, self.g_ebd[target], path
        return sample, target, self.g_ebd[target]

class PillDataset(Dataset):
    def __init__(self, root_folder, batch_size=32, g_ebedding_path=CFG.g_embedding_condensed, mode='train', exclude_path='', collate_func=None) -> None:
        self.root_folder = root_folder
        self.mode = mode
        self.batch_size = batch_size
        self.class_to_idx = json.load(open(CFG.map_class_to_idx, 'r'))
        self.transform = self.__get_transforms()
        self.prescriptions_folder = [d.name for d in os.scandir(self.root_folder) if d.is_dir()]
        self.g_embedding, self.g_embedding_np = self.__get_g_embedding(g_ebedding_path)
        # self.collate_fn = ImageCollator()
        self.collate_fn = None
        # self.img_dicts = {}
        
        if exclude_path != '':
            with open(exclude_path, 'rb') as f:
                self.exclude_list = pickle.load(f)
            print(self.exclude_list)
            self.collate_fn = CustomCollator(self.exclude_list, self.mode)
            # print(self.collate_fn)

    def __len__(self):
        return len(self.prescriptions_folder)

    def __getitem__(self, index):
        import time
        # start = time.time()
        # print('Get item...')
        pill_folder = PillFolder(self.root_folder + self.prescriptions_folder[index], self.g_embedding_np, self.class_to_idx, self.transform)
        if self.collate_fn is not None:
            pill_dts = DataLoader(pill_folder, batch_size=self.batch_size, shuffle=True, num_workers=CFG.num_workers, collate_fn=self.collate_fn)
        else:
            pill_dts = DataLoader(pill_folder, batch_size=self.batch_size, shuffle=True, num_workers=CFG.num_workers)
            
        # imgs, labels, g_ebd, path_dict = next(iter(pill_dts))
        imgs, labels, g_ebd = next(iter(pill_dts))
        
        # for k,v in path_dict.items():
        #     if k not in self.img_dicts:
        #         self.img_dicts[k] = v

        if imgs is not None:
            return imgs, labels, g_ebd
        else:
            return self.__getitem__(index - 1)

    def __get_g_embedding(self, g_embedding_path):
        g_embedding = json.load(open(g_embedding_path, 'r'))
        g_embedding_np = np.zeros((CFG.n_class, CFG.g_embedding_features), dtype=np.float32)
        
        for k, v in self.class_to_idx.items():
            if k in g_embedding.keys():
                g_embedding[v] = g_embedding[k]
                g_embedding.pop(k)

                g_embedding_np[v] = np.array(g_embedding[v])
        
        return g_embedding, torch.from_numpy(g_embedding_np)

    def __get_transforms(self):
        if self.mode == "train":
            transform = transforms.Compose([transforms.Resize((CFG.image_size, CFG.image_size)),
                                        transforms.RandomRotation(10),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=CFG.chanel_mean, std=CFG.chanel_std)])
        else:
            transform = transforms.Compose([transforms.Resize((CFG.image_size, CFG.image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=CFG.chanel_mean, std=CFG.chanel_std)])
        return transform

class CustomCollator(object):
    def __init__(self, exclude_list, mode):
        self.discard = mode == 'train'
        if self.discard:
            self.exclude_list = exclude_list
        else:
            self.exclude_list = pickle.load(open(CFG.list_test, 'rb'))
            
    def __call__(self, batch):
        xs, ys, gs = [], [], []
        
        for _x, _y, _g in batch:
            if _y in self.exclude_list:
                if self.discard:
                    continue
                else:
                    xs.append(_x)
                    ys.append(_y)
                    gs.append(_g)
            else:
                if self.discard:
                    xs.append(_x)
                    ys.append(_y)
                    gs.append(_g)
                else:
                    continue
                
        if len(xs) > 0:
            xs = torch.stack(xs)
            ys = torch.tensor(ys)
            gs = torch.stack(gs)
        
            return xs, ys, gs
        else:
            return None, None, None

class ImageCollator(object):
    def __init__(self):
        self.path_dict = {}
        
    def __call__(self, batch):
        xs, ys, gs = [], [], []
        
        for _x, _y, _g, _p in batch:
            xs.append(_x)
            ys.append(_y)
            gs.append(_g)
            if _y not in self.path_dict:
                self.path_dict[_y] = _p

        if len(xs) > 0:
            xs = torch.stack(xs)
            ys = torch.tensor(ys)
            gs = torch.stack(gs)
        
            return xs, ys, gs, self.path_dict
        else:
            return None, None, None

def visualize_imgs(imgs_dict):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    fig = plt.figure(figsize=(10,10))
    cols = 10
    rows = 8
    axs = []
    for i in range(76):
        if i in imgs_dict:
            path = imgs_dict[i]
            img = mpimg.imread(path)
            
            axs.append(fig.add_subplot(rows, cols, i+1))
            axs[-1].xaxis.set_visible(False)
            axs[-1].yaxis.set_visible(False)
            axs[-1].set_title(i)
            plt.imshow(img)
    
    plt.savefig('./logs/imgs/train_v2.png')

if __name__ == '__main__':
    # print(CFG)
    pill_dts = PillDataset(CFG.train_folder_v2, 32, './data/converted_graph/graph_exp2/name_pill_weighted_e25.json', mode='train')
    
    cnt = 0

    for img, label, g in pill_dts:
        print('Looppp')
        print(img.shape)
        print(label.shape)
        print(g.shape)
    
    # print(pill_dts.img_dicts)
    # visualize_imgs(pill_dts.img_dicts)
        # print(time.time() - start)
        # cnt += 1
        # print(cnt)