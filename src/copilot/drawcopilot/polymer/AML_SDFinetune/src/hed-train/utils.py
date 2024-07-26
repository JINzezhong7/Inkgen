"""
Utility functions for training

Author: Zhuo Su, Wenzhe Liu
Date: Aug 22, 2020
"""
import numpy as np
import torch
import torch.nn.functional as F


######################################
#       measurement functions        #
######################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        #self.sum += val * n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

######################################
#     edge specific functions        #
######################################

def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy_with_logits(
            prediction, labelf, weight=mask, reduction='sum')

    return cost

