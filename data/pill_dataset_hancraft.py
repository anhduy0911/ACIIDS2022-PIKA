import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import config as CFG
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

    def __transform_infos(self, sample):
        # print('hehehe')
        img = cv.imread(sample)
        grays = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gauss = cv.GaussianBlur(grays, (3, 3), 0)
        edges_contour = cv.Canny(gauss, 10, 50)

        texture = gauss - grays
        # plt.subplot(321), plt.imshow(img)
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(322), plt.imshow(grays, cmap='gray')
        # plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(323),plt.imshow(gauss,cmap = 'gray')
        # plt.title('Gauss Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(324),plt.imshow(edges_contour,cmap = 'gray')
        # plt.title('Contour Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(325),plt.imshow(texture,cmap = 'gray')
        # plt.title('Texture Image'), plt.xticks([]), plt.yticks([])

        # plt.savefig('test_2.png')

        return cv.resize(edges_contour, (CFG.image_size, CFG.image_size)), cv.resize(texture, (CFG.image_size, CFG.image_size))

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

    def __handcraft_transform(self, img):
        img = img[np.newaxis, :, :]
        transform_img = np.repeat(img, 3, axis=0)
        ts = torch.tensor(transform_img, dtype=torch.float)
        ts = transforms.Normalize(mean=CFG.chanel_mean, std=CFG.chanel_std)(ts)
        
        return ts

    def __getitem__(self, index: int):
        path, target = self.imgs[index]
        
        contour, texture = self.__transform_infos(path)
        contour = self.__handcraft_transform(contour)
        texture = self.__handcraft_transform(texture)
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(sample.shape)
        # print(contour.shape)
        # print(texture.shape)

        return sample, contour, texture, target, torch.tensor(self.g_embedding[target], dtype=torch.float)
    
if __name__ == '__main__':
    pill_dts = PillFolder(CFG.train_folder_new)
    randn = np.random.randint(0, 3000)
    print(pill_dts[randn][3])