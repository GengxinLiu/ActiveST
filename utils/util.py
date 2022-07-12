import math
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import numpy as np
import random
import os

LOGGING_PATH = ''


class Meter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def logging(message, end='\n', out_print=True, save_log=True):
    if out_print:
        print(message, end=end)
    if save_log:
        f = open(LOGGING_PATH, 'a+')
        f.write(message + end)
        f.close()


def get_timestamp():
    # return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return datetime.now().strftime('%Y_%m_%d_%H_%M')


def get_exp_id(exp_name):
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
        return str(1)
    return str(max([int(f) for f in os.listdir(exp_name)]) + 1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs (one-thing-one-click)"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ============ print tool ============
def bar(message, now: int, total: int):
    """
    :param message: string to print.
    :param now: the i-th iteration.
    :param total: total iteration num.
    :return:
    """
    r = f'\r[{now:03d}/{total:03d}]\t{message}'
    sys.stdout.write(r)
    sys.stdout.flush()


# ============ save log ============
def savetxt(loginfo, path_to_save, end='\n'):
    """
    :param loginfo: log
    :param path_to_save: save path
    :param end: end str
    """
    f = open(path_to_save, 'a+')
    f.write(loginfo + end)
    f.close()


# ============ plot training ============
def plot_xy(x, y, path_to_save, xlabel, ylabel, title=None):
    """
    :param x: [n] x var.
    :param y: [n] y var.
    :param path_to_save: save path
    :param xlabel:
    :param ylabel:
    :param title: figure title
    """
    fig = plt.figure()
    if title is not None:
        plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.close(fig)


def plot_epoch_loss(path_to_txt, epoch_idx, loss_idx, path_to_save_png):
    f = open(path_to_txt, 'r')
    lines = f.readlines()
    epoch, loss = [], []
    for l in lines:
        words = l.split()
        epoch.append(int(words[epoch_idx]))
        loss.append(float(words[loss_idx]))
    plot_xy(epoch, loss, path_to_save_png, 'epoch', 'loss', 'loss')


def plot_epoch_miou(path_to_txt, epoch_idx, miou_idx, path_to_save_png):
    f = open(path_to_txt, 'r')
    lines = f.readlines()
    epoch, miou = [], []
    for l in lines:
        words = l.split()
        epoch.append(int(words[epoch_idx]))
        miou.append(float(words[miou_idx]))
    plot_xy(epoch, miou, path_to_save_png, 'epoch', 'miou', 'miou')
