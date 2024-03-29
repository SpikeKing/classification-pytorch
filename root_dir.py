#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/23
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 存储项目所在的绝对路径

# DATA_DIR = '/Users/wang/workspace/data_set/'
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# DATASET = os.path.join(DATA_DIR, 'person-test')
DATASET = '/Users/wang/workspace/data_set/person-test'
MODELS_DIR = os.path.join(DATA_DIR, 'models')
IMGS_DIR = os.path.join(DATA_DIR, 'imgs')
