import torch.nn as nn
import math 

class SwishAlt():
    def __init__(self):
        self.b=1.0
    def ActSwish (self, x):
        return x / (1 + math.exp(-self.b * x))