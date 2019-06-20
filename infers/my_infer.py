# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

import torch
from torchvision import transforms

from bases.infer_base import InferBase
from root_dir import MODELS_DIR


class MyInfer(InferBase):
    def __init__(self, model_path, config=None):
        super(MyInfer, self).__init__(config)
        self.model = self.load_model(model_path)

        self.trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model

    def predict(self, data):
        img_torch = self.trans(data)  # 标准变换
        print("[Info] 变换之后的图像: {}".format(img_torch.shape))

        img_torch = torch.unsqueeze(img_torch, 0).to(torch.device("cpu"))
        print("[Info] 增加1维: {}".format(img_torch.shape))

        return self.model(img_torch)[0]
