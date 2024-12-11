import copy
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

from config import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics import *
from renderer import *
from templates import *

checkpoint_path = '/data/jincheol/diffae'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model_x0_config = retina_autoenc()
base_model_x0 = LitModel(base_model_x0_config)
state = torch.load(checkpoint_path + '/checkpoints/retina_autoenc_color_mask_attention/epoch=60-step=2249.ckpt')
base_model_x0.load_state_dict(state["state_dict"], strict=False)
base_model_x0.ema_model.eval()
base_model_x0 = base_model_x0.to(device)

breakpoint()

base_data_x0 = Image.open(checkpoint_path + '/imgs/FIVES-1.png')
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
base_data_x0 = transform(base_data_x0)
base_data_x0 = base_data_x0.unsqueeze(0)
base_data_x0 = base_data_x0.to(device)



def predict_x0(x_start):
    
    base_model_x0.model.eval()

    with torch.no_grad():

        for i in range(40):

            x_T = torch.randn(1, 3, 256, 256)
            x_T = x_T.to(device)
            loader = DataLoader(x_T, batch_size=1)
   
            cond = None

            gen = base_model_x0.eval_sampler.sample(model=base_model_x0.model, noise=x_T, cond=cond, x_start=base_data_x0)

            result = gen
            result = result.squeeze(0)
            result = (result + 1) / 2
            result = result.permute(1, 2, 0).cpu().numpy()
   
            plt.imsave(f'/DATA/sunghun/diffae/image_x0_40_3/image{i}.png', result)

    base_model_x0.model.train()
        
predict_x0(x_start=base_data_x0)