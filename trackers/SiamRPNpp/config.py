# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Revised for THOR by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------
from yacs.config import CfgNode as CN

class TrackerConfig(object):
    #windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    #instance_size_glob = 767
    stride = 8
    #score_size = (instance_size-exemplar_size)/total_stride+1
    #score_size_glob = (instance_size_glob-exemplar_size)/total_stride+1
    context_amount = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    anchor_glob = []
    penalty_k = 0.04
    window_influence = 0.44
    lr = 0.4
    #adaptive = True
    base_size = 8
    lost_instance_size = 831
    confidence_low = 0.85
    confidence_high = 0.998
    mask_threshold = 0.30
    mask_output_size = 127
    score_size = (instance_size - exemplar_size) // \
            stride + 1 + base_size

    #from SiamRPNpp
    # ------------------------------------------------------------------------ #
    # Backbone options
    # ------------------------------------------------------------------------ #

    BACKBONE = CN()

    # Backbone type, current only support resnet18,34,50;alexnet;mobilenet
    BACKBONE.TYPE = 'res50'

    BACKBONE.KWARGS = CN(new_allowed=True)

    # Pretrained backbone weights
    BACKBONE.PRETRAINED = ''

    # Train layers
    BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

    # Layer LR
    BACKBONE.LAYERS_LR = 0.1

    # Switch to train layer
    BACKBONE.TRAIN_EPOCH = 10 #第10层开始训练

    # ------------------------------------------------------------------------ #
    # Adjust layer options
    # ------------------------------------------------------------------------ #
    ADJUST = CN()

    # Adjust layer
    ADJUST.ADJUST = True

    ADJUST.KWARGS = CN(new_allowed=True)

    # Adjust layer type
    ADJUST.TYPE = "AdjustAllLayer"

    # ------------------------------------------------------------------------ #
    # RPN options
    # ------------------------------------------------------------------------ #
    RPN = CN()

    # RPN type
    RPN.TYPE = 'MultiRPN'

    RPN.KWARGS = CN(new_allowed=True)

    # ------------------------------------------------------------------------ #
    # mask options
    # ------------------------------------------------------------------------ #
    MASK = CN()

    # Whether to use mask generate segmentation
    MASK.MASK = False

    # Mask type
    MASK.TYPE = "MaskCorr"

    MASK.KWARGS = CN(new_allowed=True)

    REFINE = CN()

    # Mask refine
    REFINE.REFINE = False

    # Refine type
    REFINE.TYPE = "Refine"


    # TRACK.TYPE = 'SiamRPNTracker'



    def update(self, cfg):
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.stride + 1 + self.base_size
