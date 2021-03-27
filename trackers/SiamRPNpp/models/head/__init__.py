# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from trackers.SiamRPNpp.models.head.mask import MaskCorr, Refine
from trackers.SiamRPNpp.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN
       }

MASKS = {
         'MaskCorr': MaskCorr,
        }

REFINE = {
          'Refine': Refine,
         }


def get_rpn_head(name, **kwargs):
    return RPNS[name](5, [256, 256, 256], **kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()
