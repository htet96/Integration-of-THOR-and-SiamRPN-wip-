# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from trackers.SiamRPNpp.models.neck.neck import AdjustLayer, AdjustAllLayer

NECKS = {
         'AdjustLayer': AdjustLayer,
         'AdjustAllLayer': AdjustAllLayer
        }

#edited for THOR

def get_neck(name, **kwargs):
    return NECKS[name]([512, 1024, 2048], [256, 256, 256], **kwargs)
