import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from zipfile import ZipFile
from io import BytesIO
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm


### Model with modules

class RCat(nn.Module):
    def __init__(self, f):
        super(RCat, self).__init__()
        self.conv1 = nn.Conv2d(f, f//4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(f, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(f//2, f//4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        x1 = F.relu(self.conv6(x1))
        c1 = torch.cat((x1, y1), dim=1)
        c2 = F.relu(self.conv7(c1))
        return c2 + y1

class RDN(nn.Module):
    def __init__(self, f):
        super(RDN, self).__init__()
        self.conv1 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(3*f, f//4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        a1 = y1 + y2
        y3 = self.conv3(a1)
        a2 = y3 + a1
        y4 = self.conv4(a2)
        a3 = a1 + a2 + y4
        c = torch.cat((a1, a2, a3), dim=1)
        return self.conv5(c)


class Runt(nn.Module):
    def __init__(self, f):
        super(Runt, self).__init__()
        self.conv1 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(f, f//2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f//2, f//2, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(f//2, f//4, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(f//4, f//4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(f//4, f//8, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(f//8, f//8, kernel_size=5, padding=2)
        self.conv9 = nn.Conv2d(f//8, f//8, kernel_size=1)
        self.conv10 = nn.Conv2d(f//4, f//4, kernel_size=1)
        self.conv11 = nn.Conv2d(f//2, f//2, kernel_size=1)
        self.conv12 = nn.Conv2d(f, f, kernel_size=1)
        self.conv13 = nn.Conv2d(f, 3, kernel_size=1)

    def forward(self, x):
        y1 = F.relu(self.conv1(x))
        y1 = F.relu(self.conv2(y1))
        y2 = F.relu(self.conv3(y1))
        y2 = F.relu(self.conv4(y2))
        y3 = F.relu(self.conv5(y2))
        y3 = F.relu(self.conv6(y3))
        y4 = F.relu(self.conv7(y3))
        y4 = F.relu(self.conv8(y4))
        y5 = F.relu(self.conv9(y4))
        c1 = torch.cat([y5, y4], dim=1)
        y6 = F.relu(self.conv10(c1))
        c2 = torch.cat([y6, y3], dim=1)
        y7 = F.relu(self.conv11(c2))
        c3 = torch.cat([y7, y2], dim=1)
        y8 = F.relu(self.conv12(c3))
        y8 = y8 + y1
        y9 = self.conv13(y8)
        return y9



class Den(nn.Module):
    def __init__(self):
        super(Den, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)
        self.actc = ACTC()  # Assuming ACTC is a defined PyTorch module
        self.mdr_inp = nn.Conv2d(64, 3, kernel_size=1) 
        self.mdsr1 = MDSR1(32)  # Assuming MDSR1 is a defined PyTorch module
        self.rdn_inp = nn.Conv2d(64, 128, kernel_size=1) 
        self.rdn = RDN(128)  # Assuming RDN is a defined PyTorch module
        self.conv4 = nn.Conv2d(102, 3, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        f1 = self.actc(x)
        md_in = self.mdr_inp(x)
        f2 = self.mdsr1(md_in)
        rd_in = self.rdn_inp(x)
        f3 = self.rdn(rd_in)
        inp = torch.cat([f1, f2, f3, x], dim=1)
        x = self.conv4(inp)
        return x


class ACTC(nn.Module):
    def __init__(self):
        super(ACTC, self).__init__()
        self.conv1 = nn.Conv2d(64, 7, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 7, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 7, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(21, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(x)
        x2 = torch.sigmoid(x)
        x3 = x2 * x
        x4 = F.softplus(x)
        x4 = torch.tanh(x4)
        x5 = x4 * x
        c1 = self.conv1(x1)
        c2 = self.conv2(x3)
        c3 = self.conv3(x5)
        cx = torch.cat([c1, c2, c3], dim=1)
        y = self.conv4(cx)
        return y


class R1(nn.Module):
    def __init__(self, features):
        super(R1, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)

    def forward(self, input_tensor):
        x = F.relu(self.conv1(input_tensor))
        x = self.conv2(x)
        return x + input_tensor



class MDSR1(nn.Module):
    def __init__(self, f):
        super(MDSR1, self).__init__()
        self.conv1 = nn.Conv2d(3, f, kernel_size=3, padding=1)
        self.r1 = R1(f)  # Assuming R1 is a defined PyTorch module
        self.conv2 = nn.Conv2d(4*f, 3, kernel_size=3, padding=1)

    def forward(self, ix):
        x = F.relu(self.conv1(ix))
        x1 = self.r1(x)
        x1 = self.r1(x1)
        x2 = self.r1(x)
        x2 = self.r1(x2)
        x3 = self.r1(x)
        x3 = self.r1(x3)
        x = x1 + x2 + x3
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv2(x)
        return x


### Final Model

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=0, dilation=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0, dilation=4)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, padding=0, dilation=16)
        self.den_inp = nn.Conv2d(64, 3, kernel_size=1) 
        self.den = Den()  # Assuming Den is a defined PyTorch module
        self.runt_inp = nn.Conv2d(64, 16, kernel_size=1) 
        self.runt = Runt(16)  # Assuming Runt is a defined PyTorch module
        self.conv4 = nn.Conv2d(6, 3, kernel_size=1, padding=1, dilation=8)  # Adjusted number of input channels
        self.runt_inp2 = nn.Conv2d(3, 32, kernel_size=1)
        self.runt2 = Runt(32)  # Assuming Runt is a defined PyTorch module
        self.conv5 = nn.Conv2d(6, 3, kernel_size=1, padding=1, dilation=8)  # Adjusted number of input channels

    def forward(self, input_im):
        x = F.relu(self.conv1(input_im))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        den_in = self.den_inp(x) 
        x1 = self.den(den_in)
        runt_in = self.runt_inp(x)
        x2 = self.runt(runt_in)
        x = torch.cat([x1, x2], dim=1)
        # print(x.shape)
        x = F.relu(self.conv4(x))
        x1 = self.den(x)
        runt_in = self.runt_inp2(x)
        x2 = self.runt2(runt_in)
        x = torch.cat([x1, x2], dim=1)
        y = self.conv5(x)
        # Upsample the output to be 4 times the size of the input
        y = F.interpolate(y, scale_factor=4, mode='bicubic', align_corners=False)
        return y