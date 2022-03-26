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

        return sample, target, self.g_ebd[target]

class PillDataset(Dataset):
    def __init__(self, root_folder, batch_size=32, g_ebedding_path=CFG.g_embedding_condensed, mode='train') -> None:
        self.root_folder = root_folder
        self.mode = mode
        self.batch_size = batch_size
        self.class_to_idx = json.load(open(CFG.map_class_to_idx, 'r'))
        self.transform = self.__get_transforms()
        self.prescriptions_folder = [d.name for d in os.scandir(self.root_folder) if d.is_dir()]
        self.g_embedding, self.g_embedding_np = self.__get_g_embedding(g_ebedding_path)


    def __len__(self):
        return len(self.prescriptions_folder)

    def __getitem__(self, index):
        import time
        # start = time.time()
        # print('Get item...')
        pill_folder = PillFolder(self.root_folder + self.prescriptions_folder[index], self.g_embedding_np, self.class_to_idx, self.transform)
        pill_dts = DataLoader(pill_folder, batch_size=self.batch_size, shuffle=True, num_workers=CFG.num_workers)
        
        # print(time.time() - start)
        # start = time.time()
        # indexes = random.sample(range(len(pill_folder)), self.batch_size)

        # # rand_idx = random.randint(0, len(pill_folder) - 1)
        # imgs = []
        # labels = []
        # for i in indexes:
        #     img, label = pill_folder[i]
        #     imgs.append(img.unsqueeze(0))
        #     labels.append(label)
        
        # print(time.time() - start)
        # start = time.time()
        
        # imgs = torch.cat(imgs, dim=0)
        # labels = torch.cat(labels, dim=0)

        imgs, labels, g_ebd = next(iter(pill_dts))
        
        # print(time.time() - start)
        # start = time.time()
        # print(imgs.shape)
        # print(labels)

        return imgs, labels, g_ebd

    def __get_g_embedding(self, g_embedding_path):
        g_embedding = json.load(open(g_embedding_path, 'r'))
        g_embedding_np = np.zeros((CFG.n_class, CFG.g_embedding_features), dtype=np.float32)

        for k, v in self.class_to_idx.items():
            g_embedding[v] = g_embedding[k]
            g_embedding.pop(k)

            g_embedding_np[v] = np.array(g_embedding[v])
        
        # print(g_embedding_np.shape)

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


if __name__ == '__main__':
    # print(CFG)
    pill_dts = PillDataset(CFG.train_folder_v2, 32, CFG.g_embedding_condensed, mode='train')
    pill_dts = PillDataset(CFG.train_folder_v2, 32, CFG.g_embedding_condensed, mode='train')

    cnt = 0
    print(len(pill_dts))

    for img, label, g in pill_dts:
        print('Looppp')
        print(img.shape)
        print(label.shape)
        print(g.shape)
        # print(time.time() - start)
        # cnt += 1
        # print(cnt)