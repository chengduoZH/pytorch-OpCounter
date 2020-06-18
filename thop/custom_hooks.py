import torch
import torch.nn as nn


def count_layer_norm(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_ops += torch.DoubleTensor([int(total_ops)])
