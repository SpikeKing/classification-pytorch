# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from __future__ import division
from __future__ import print_function

import copy
import time

import os
import torch
import torch.nn as nn
import torch.optim as optim

from bases.trainer_base import TrainerBase
from root_dir import MODELS_DIR
from utils.utils import get_current_time_str


class MyTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(MyTrainer, self).__init__(model, data, config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('[Info] device: {}'.format(self.device))

        self.config = config

        self.model_ft = model
        self.dataloaders_dict = data

        self.feature_extract = True if config.feature_extract == 1 else False
        self.num_epochs = config.num_epochs
        self.lr = config.lr

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            # 存储模型
            save_path = os.path.join(MODELS_DIR, 'model_{}_{}.pth'.format(epoch, get_current_time_str()))
            torch.save(model, save_path)
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def train(self):
        # Send the model to GPU
        model_ft = self.model_ft.to(self.device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.RMSprop(params_to_update, lr=self.lr)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = self.train_model(model_ft, self.dataloaders_dict, criterion, optimizer_ft,
                                          num_epochs=self.num_epochs,
                                          is_inception=False)
