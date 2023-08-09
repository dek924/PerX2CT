import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

### From Autoint
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.sigmoid(input)


# Model
class DummyNeRF(nn.Module):
    def __init__(self, cfg):
        """ """
        super(DummyNeRF, self).__init__()
        self.cfg = cfg
        self.input_ch = cfg['input_ch']
        self.output_ch = cfg['output_ch']
        print(f"[DummyNeRF] Input ch : {self.input_ch}, Output ch : {self.output_ch}")

        ## for dimension reduction
        if self.input_ch != self.output_ch:
            self.linear = nn.Linear(self.input_ch, self.output_ch)

    def forward(self, x):
        if self.input_ch != self.output_ch:
            x = self.linear(x)
        return {'outputs': x}
