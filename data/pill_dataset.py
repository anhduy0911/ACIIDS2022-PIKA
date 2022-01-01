import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from typing import Tuple, List, Dict
import config as CFG
import random

class PillFolder(ImageFolder):
    def __init__(self, root, total_label=[], transform=None):
        self.total_label = total_label
        super(PillFolder, self).__init__(root, transform=transform)
        self.imgs_dict = self.__separate_data_to_label()

    def __separate_data_to_label(self):
        img_data = {}
        for i, l in self.imgs:
            if l not in img_data.keys():
                img_data[l] = []
            img_data[l].append(i)
        
        return img_data
    
    def __len__(self):
        return len(self.imgs)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: self.total_label.index(cls_name) for cls_name in classes}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return sample, torch.tensor([target], dtype=torch.long)
        return sample, target
    
        
class PillDataset(Dataset):
    def __init__(self, train_folder, batch_size=1, g_ebedding_path=CFG.g_embedding_path, mode='train') -> None:
        self.train_folder = train_folder
        self.g_embedding_path = g_ebedding_path
        self.mode = mode
        self.batch_size = batch_size

        self.label_dict, self.g_embedding = self.__get_label_gebedding()
        self.transform = self.__get_transforms()
        self.prescriptions_folder = [d.name for d in os.scandir(self.train_folder) if d.is_dir()]

    def __len__(self):
        return len(self.prescriptions_folder)

    def __getitem__(self, index):
        # import time
        # start = time.time()
        # print('Get item...')
        pill_folder = PillFolder(self.train_folder + self.prescriptions_folder[index], self.label_dict, self.transform)
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

        imgs, labels = next(iter(pill_dts))
        
        # print(time.time() - start)
        # start = time.time()
        # print(imgs.shape)
        # print(labels)

        return imgs, labels, self.g_embedding[labels]

    def __get_label_gebedding(self):
        labels = []
        g_embedding = []
        with open(self.g_embedding_path) as f:
            lines = f.readlines()
            for line in lines:
                _, mapped_pill, ebd = line.strip().split('\\')
                labels.append(mapped_pill)
                g_embedding.append([float(x) for x in ebd.split(' ')])

        return labels, torch.Tensor(g_embedding)

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
    pill_dts = PillDataset(CFG.train_folder, 32, CFG.g_embedding_path, mode='train')
    # pill_dts = PillFolder(CFG.train_folder, CFG.label_dict, pill_dts.transform)
    # dt_loader = DataLoader(pill_dts, batch_size=1, shuffle=True)
    cnt = 0
    print(len(pill_dts))

    import time
    start = time.time()
    for img, label, g in pill_dts:
        print('Looppp')
        print(img.shape)
        print(label)
        print(g.shape)
        # print(time.time() - start)
        # cnt += 1
        # print(cnt)