import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import config as CFG
import numpy as np
import cv2 as cv
class PillFolder(ImageFolder):
    def __init__(self, root, mode='train', g_embedding_path=CFG.g_embedding_condensed):
        self.mode = mode
        self.transform = self.__get_transforms()

        super(PillFolder, self).__init__(root, transform=self.transform)
        self.g_embedding, self.g_embedding_np = self.__get_g_embedding(g_embedding_path)
    
    def __len__(self):
        return len(self.imgs)

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

    def __getitem__(self, index: int):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return sample, torch.tensor([target], dtype=torch.long)
        return sample, target, torch.tensor(self.g_embedding[target], dtype=torch.float)
    
if __name__ == '__main__':
    # print(CFG)
    pill_dts = PillFolder(CFG.train_folder_new)
    print(pill_dts.g_embedding_np.shape)
    # pill_dts = PillFolder(CFG.train_folder, CFG.label_dict, pill_dts.transform)
    # dt_loader = DataLoader(pill_dts, batch_size=32, shuffle=True)
    # cnt = 0
    # print(len(pill_dts))

    # import time
    # start = time.time()
    # for img, label, g in dt_loader:
    #     print('Looppp')
    #     print(img.shape)
    #     print(label)
    #     print(g.shape)