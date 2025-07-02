import random
import torch
import numpy as np
from typing import List

def list_check(obj)->List:
    return obj if isinstance(obj,list) else [obj]

def seed_everything(seed = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False