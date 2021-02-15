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

def get_landmakrs(img_folder, lmks_folder, img_listfile=None):
    if not os.path.exists(lmks_folder):
        os.mkdir(lmks_folder)
    imgs = []

    if img_listfile is None:
        for fname in os.listdir(img_folder):
            ext = fname.split('.')[-1]
            if ext not in ["png", "jpg"]:
                continue
            src_img_path = os.path.join(img_folder, fname)
            imgs.append((fname,src_img_path))
    else:
        with open(img_listfile, "r") as f:
            names = f.readlines()
        names = [l.strip("\n") for l in names]
        for fname in names:
            ext = fname.split('.')[-1]
            if ext not in ["png", "jpg"]:
                continue
            src_img_path = os.path.join(img_folder, fname)
            imgs.append((fname,src_img_path))
            prefix = "/".join(fname.split("/")[:-1])
            if not os.path.exists(os.path.join(lmks_folder, prefix)):
                os.makedirs(os.path.join(lmks_folder, prefix))

    imgs.sort()

    args = SimpleNamespace()
    args.val_img_dir='./dataset/WFLW_test/images/'
    args.val_landmarks_dir='./dataset/WFLW_test/landmarks/'
    args.ckpt_save_path='./experiments/eval_iccv_0620'
    args.hg_blocks=4
    args.pretrained_weights='./ckpt/WFLW_4HG.pth'
    args.num_landmarks=98
    args.end_relu='False'
    args.batch_size=1
    args.gray_scale=False
    args.save_vis = True

    VAL_IMG_DIR = args.val_img_dir
    VAL_LANDMARKS_DIR = args.val_landmarks_dir
    CKPT_SAVE_PATH = args.ckpt_save_path
    BATCH_SIZE = args.batch_size
    PRETRAINED_WEIGHTS = args.pretrained_weights
    GRAY_SCALE = False if args.gray_scale == 'False' else True
    HG_BLOCKS = args.hg_blocks
    END_RELU = False if args.end_relu == 'False' else True
    NUM_LANDMARKS = args.num_landmarks

    device = "cuda"
    use_gpu = torch.cuda.is_available()
    print("use_gpu", use_gpu)
    model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)

    if PRETRAINED_WEIGHTS != "None":
        checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=torch.device('cpu'))
        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)

    to_tensor = transforms.Compose([
         transforms.Resize(256),
         transforms.ToTensor()
        ])
    model_ft = model_ft.to(device)
    with torch.no_grad():
        for fname, src_img_path in imgs:
            image = Image.open(src_img_path)
            image = image.convert('RGB')
            w,h = image.size

            # padding if neccesary
            desired_size = 640
            ratio = desired_size/float(min(w,h))
            w2, h2 = int(w*ratio), int(h*ratio)
            image = image.resize((w2, h2))
            image_raw = image

            old_size = image.size  # old_size[0] is in (width, height) format
            new_im = Image.new("RGB", (desired_size, desired_size))
            x0 = (desired_size-old_size[0])//2
            y0 = (desired_size-old_size[1])//2
            new_im.paste(image, (x0,
                                y0))
            image = new_im

            isz = image.size[0]
            inputs = to_tensor(image).unsqueeze(0)
            inputs = inputs.to(device)
            outputs, boundary_channels = model_ft(inputs)
            pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
            pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
            pred_landmarks = pred_landmarks.squeeze()*4 / 256.0 * isz

            lmks68 = torch.zeros([68, 2])
            for i98,i68 in WFLW_98_TO_DLIB_68_IDX_MAPPING.items():
                lmks68[i68,:] = pred_landmarks[i98,:]
            #lmks68 = pred_landmarks[:-2]
            lmks68[:,0] -= x0
            lmks68[:,1] -= y0
            lmks68 /= ratio

            ext = fname.split('.')[-1]
            save_npy_path = os.path.join(lmks_folder, fname.replace(ext, "npy"))
            print(save_npy_path)
            np.save(save_npy_path, lmks68)
            image = image_raw.resize((w,h))
            if args.save_vis:
                plt.figure()
                plt.imshow(image)
                plt.scatter(lmks68[:, 0], lmks68[:, 1], s=5, marker='.', c='r')
                save_vis_path = os.path.join(lmks_folder, fname)
                plt.savefig(save_vis_path)
            print(save_vis_path)

get_landmakrs("../ssddata/FaceWarehouse_Data_0_cropseg",
        "../ssddata/FaceWarehouse_Data_0_adlmk", "../ssddata/fwh_imagelist.txt")
