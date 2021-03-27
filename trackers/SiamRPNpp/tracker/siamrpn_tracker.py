# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from ipdb import set_trace

from trackers.SiamRPNpp.core.config import cfg
from trackers.SiamRPNpp.utils.anchor import Anchors
from trackers.SiamRPNpp.config import TrackerConfig

# directly put in the SiameseTracker functions in this file
# from siamrpnpp.tracker.base_tracker import SiameseTracker

def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch

def generate_anchor():
    p = TrackerConfig()
    anchors = Anchors(p.stride,
                      p.ratios,
                      p.scales)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
    total_stride = anchors.stride
    anchor_num = anchor.shape[0]
    anchor = np.tile(anchor, p.score_size * p.score_size).reshape((-1, 4))
    ori = - (p.score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(p.score_size)],
                         [ori + total_stride * dy for dy in range(p.score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
        np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def SiamRPNpp_init(im, target_pos, target_sz, cfg):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    p = TrackerConfig()
    p.update(cfg)

    p.anchor = generate_anchor()
    avg_chans = np.mean(im, axis=(0, 1))

    hanning = np.hanning(p.score_size)
    window = np.outer(hanning, hanning)
    window = np.tile(window.flatten(), p.anchor_num)

    state['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state['p'] = p
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    # state['score'] = 1.0
    return state

def SiamRPNpp_track(state, im, temp_mem):
    p = state['p']
    avg_chans = state['avg_chans']
    window = state['window']
    old_pos = state['target_pos']
    old_sz = state['target_sz']
    dev = state['device']

    # calculate z crop size
    wc_z = old_sz[1] + p.context_amount * np.sum(old_sz)
    hc_z = old_sz[0] + p.context_amount * np.sum(old_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # not sure if below is needed
    s_z = min(s_z, min(state['im_h'], state['im_w']))

    scale_z = p.exemplar_size / s_z
    s_x = round(s_z * (p.instance_size / p.exemplar_size))

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow(im, old_pos, p.instance_size,\
                                             round(s_x), avg_chans).unsqueeze(0))

    # track
    target_pos, target_sz, score = temp_mem.batch_evaluate(x_crop.to(dev), old_pos, old_sz * scale_z, window, scale_z, p)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['crop'] = x_crop
    return state
