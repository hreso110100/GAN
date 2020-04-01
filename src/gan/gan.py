import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


class GAN(nn.Module):

    def __init__(self):
        super(GAN, self).__init__()
