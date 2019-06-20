#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/6/20
"""

import os

from PIL import Image

from infers.my_infer import MyInfer
from root_dir import IMGS_DIR, MODELS_DIR


def main_test():
    model_path = os.path.join(MODELS_DIR, 'model_19_20190620104255.pth')
    mi = MyInfer(model_path)

    img_path = os.path.join(IMGS_DIR, 'male.jpg')
    img_pil = Image.open(img_path)
    print('[Info] 原始图片尺寸: {}'.format(img_pil.size))

    output = mi.predict(img_pil)
    print('[Info] 输出维度: {}'.format(output.shape))
    print('[Info] 输出结果: {}'.format(output.detach().numpy()))


def main():
    main_test()


if __name__ == '__main__':
    main()
