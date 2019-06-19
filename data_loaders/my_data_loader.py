# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bases.data_loader_base import DataLoaderBase
from root_dir import DATA_DIR


class MyDataLoader(DataLoaderBase):
    def __init__(self, config=None):
        super(MyDataLoader, self).__init__(config)

        print("[Info] Initializing Datasets and Dataloaders...")

        input_size = 224
        batch_size = 20

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = DATA_DIR
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                          ['train', 'val']}
        self.dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                                 for x in ['train', 'val']}

        print('[Info] 加载数据完成!')

    def get_data_dict(self):
        return self.dataloaders_dict
