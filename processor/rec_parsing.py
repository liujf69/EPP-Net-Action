import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from .processor import Processor

import sys
import matplotlib.pyplot as plt
from PIL import Image
import time
import pickle

class REC_Processor(Processor):

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k, phase):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
        if phase == 'eval':
            result_dict = dict(zip(self.data_loader['test'].dataset.sample_name, self.result))
            self.io.save_pkl(result_dict, 'tmp_test_result.pkl')
            r3 = open(os.path.join(self.arg.work_dir,'tmp_test_result.pkl'), 'rb')
            r3 = list(pickle.load(r3).items())
            total_num = right_num_5 = 0
            for i,l in enumerate(self.label):
                _, r33 = r3[i]
                r1r3 = r33
                rank_5 = r1r3.argsort()[-5:]
                right_num_5 += int(int(l) in rank_5)
                r1r3 = np.argmax(r1r3)
                total_num += 1
            acc5 = right_num_5 / total_num
            self.io.print_log('Top 1:                        {}'.format(accuracy))
            self.io.print_log('Top 5:                        {}'.format(acc5))
        if k == 1:
            self.progress_info[int(self.meta_info['epoch']/self.arg.eval_interval), 2]  =  100 * accuracy

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        result_frag = []
        label_frag = []
        loss_value = []

        for rgb, label in loader:
            label = label.long().to(self.dev)
            rgb = rgb.float().to(self.dev)
            output = self.model(rgb)
            output = output.logits # inceptionV3

            ls_cls = self.loss(output, label)
            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())
            loss = ls_cls

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['ls_cls'] = ls_cls.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['ls_cls'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['ls_cls']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        if ((self.meta_info['epoch'] + 1) % self.arg.eval_interval == 0):
            for k in self.arg.show_topk:
                self.show_topk(k, 'train')

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for rgb, label in loader:
            label = label.long().to(self.dev)
            rgb = rgb.float().to(self.dev)

            with torch.no_grad():
                output = self.model(rgb)

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                ls_cls = self.loss(output, label)
                loss = ls_cls
                self.iter_info['ls_cls'] = ls_cls.data.item()
                loss_value.append(ls_cls.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['ls_cls']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k, 'eval')

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--show_topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--temporal_positions', default=None, help='temporal positions for calculating the joint weights')

        return parser
