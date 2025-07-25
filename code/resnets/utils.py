import matplotlib.pyplot as plt
import numpy as np

import random
import torch
from typing import List

def list_check(obj)->List:
    return obj if isinstance(obj,list) else [obj]

def seed_everything(seed = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

