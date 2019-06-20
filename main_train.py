#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18

参考:
NumPy FutureWarning
https://stackoverflow.com/questions/48340392/futurewarning-conversion-of-the-second-argument-of-issubdtype-from-float-to
"""

import numpy as np

from data_loaders.my_data_loader import MyDataLoader
from models.my_model import MyModel
from trainers.my_trainer import MyTrainer
from utils.config_utils import init_config


def main_train():
    """
    训练模型

    :return:
    """
    print('[INFO] 解析配置...')
    config = init_config('configs/my_config.json')

    np.random.seed(47)  # 固定随机数

    print('[INFO] 构造网络...')
    model = MyModel(config=config)

    print('[INFO] 加载数据...')
    dl = MyDataLoader(input_size=model.input_size, config=config)

    print('[INFO] 训练网络...')
    trainer = MyTrainer(
        model=model.model,
        data=dl.get_data_dict(),
        config=config)
    trainer.train()
    print('[INFO] 训练完成...')


if __name__ == '__main__':
    main_train()
    # test_main()
