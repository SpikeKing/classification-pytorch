# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bases.data_loader_base import DataLoaderBase
from root_dir import DATASET


class MyDataLoader(DataLoaderBase):
    def __init__(self, input_size, config=None):
        super(MyDataLoader, self).__init__(config)

        print("[Info] Initializing Datasets and Dataloaders...")

        dataset_dir = DATASET
        batch_size = config.batch_size

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

        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) for x in
                          ['train', 'val']}
        self.dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                                 for x in ['train', 'val']}

        print('[Info] 加载数据完成!')

    def get_data_dict(self):
        return self.dataloaders_dict
