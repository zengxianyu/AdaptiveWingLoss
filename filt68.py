from __future__ import print_function, division
import torch
import pdb
import argparse
import numpy as np
import torch.nn as nn
import time
import os
from PIL import Image, ImageFilter
from torchvision import transforms
from core.evaler import eval_model
from core.dataloader import get_dataset
from utils.utils import fan_NME, show_landmarks, get_preds_fromhm
from core import models
import matplotlib.pyplot as plt
from types import SimpleNamespace

from collections import OrderedDict
DLIB_68_TO_WFLW_98_IDX_MAPPING = OrderedDict()
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(0,17),range(0,34,2)))) # jaw | 17 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(17,22),range(33,38)))) # left upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(22,27),range(42,47)))) # right upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(27,36),range(51,60)))) # nose points | 9 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({36:60}) # left eye points | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({37:61})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({38:63})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({39:64})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({40:65})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({41:67})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({42:68}) # right eye | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({43:69})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({44:71})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({45:72})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({46:73})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({47:75})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(48,68),range(76,96)))) # mouth points | 20 pts

WFLW_98_TO_DLIB_68_IDX_MAPPING = {v:k for k,v in DLIB_68_TO_WFLW_98_IDX_MAPPING.items()}

def get_landmakrs(img_folder, lmks_folder):
    if not os.path.exists(lmks_folder):
        os.mkdir(lmks_folder)
    imgs = []

    for fname in os.listdir(img_folder):
        if not fname.endswith(".npy"):
            continue
        src_img_path = os.path.join(img_folder, fname)
        imgs.append((fname,src_img_path))
    for fname, src_img_path in imgs:
        pred_landmarks = np.load(src_img_path)
        lmks68 = np.zeros([68, 2])
        for i98,i68 in WFLW_98_TO_DLIB_68_IDX_MAPPING.items():
            lmks68[i68,:] = pred_landmarks[i98,:]
        save_npy_path = os.path.join(lmks_folder, fname)
        print(save_npy_path)
        np.save(save_npy_path, lmks68)

get_landmakrs("../celebhq256x256/lmk_awl",
        "../celebhq256x256/lmk_awl68")
