import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import json

class PillMultilabel(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = os.path.abspath(root)
        self.phase = phase
        self.img_list = []
        self.transform = transform
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        print('[dataset] Pill classification phase={} number of classes={}  number of images={}'.format(phase, self.num_classes, len(self.json_list)))
    
    def get_list_file(self,folder_path):
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        return onlyfiles

    def get_anno(self):
        self.json_list = self.get_list_file(self.root + "/data_{}/labels".format(self.phase))
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.cat2idx = json.load(open(os.path.join(self.root, 'cat2idx.json'), 'r'))
        self.instance_list =  json.load(open(os.path.join(self.root + f"/data_{self.phase}", f'instances_{self.phase}.json'), 'r'))
    
    def __len__(self):
        # return 100
        return len(self.json_list)

    def __getitem__(self, index):
        item = json.load(open(os.path.join(self.root + f"/data_{self.phase}/labels",self.json_list[index]), 'r'))
        filename = item["path"]
        img = Image.open(os.path.join(self.root + f"/data_{self.phase}/images", filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        labels =  []
        for box in item["boxes"]:
            labels.append(self.cat2idx[box['label']])
        target = torch.zeros(self.num_classes,  dtype=torch.float32) - 1
        target[labels] = 1
        data = {'image':img, 'name': filename, 'target': target}
        return data 