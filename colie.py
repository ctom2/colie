from utils import *
from loss import *
from siren import INF
from color import rgb2hsv_torch, hsv2rgb_torch

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description='CoLIE')
parser.add_argument('--input_folder', type=str, default='input/')
parser.add_argument('--output_folder', type=str, default='output/')
parser.add_argument('--down_size', type=int, default=256, help='downsampling size')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--window', type=int, default=1, help='context window size')
parser.add_argument('--L', type=float, default=0.5)
# loss fuction weigth parameters
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--beta', type=float, required=True)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--delta', type=float, required=True)
opt = parser.parse_args()


if not os.path.exists(opt.input_folder):
    print('input folder: {} does not exist'.format(opt.input_folder))
    exit()

if not os.path.exists(opt.output_folder):
    os.makedirs(opt.output_folder)


print(' > running')
for PATH in tqdm(np.sort(os.listdir(opt.input_folder))):
    img_rgb = get_image(os.path.join(opt.input_folder, PATH))
    img_hsv = rgb2hsv_torch(img_rgb)

    img_v = get_v_component(img_hsv)
    img_v_lr = interpolate_image(img_v, opt.down_size, opt.down_size)
    coords = get_coords(opt.down_size, opt.down_size)
    patches = get_patches(img_v_lr, opt.window)


    img_siren = INF(patch_dim=opt.window**2, num_layers=4, hidden_dim=256, add_layer=2)
    img_siren.cuda()

    optimizer = torch.optim.Adam(img_siren.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)

    l_exp = L_exp(16,opt.L)
    l_TV = L_TV()

    for epoch in range(opt.epochs):
        img_siren.train()
        optimizer.zero_grad()

        illu_res_lr = img_siren(patches, coords)
        illu_res_lr = illu_res_lr.view(1,1,opt.down_size,opt.down_size)
        illu_lr = illu_res_lr + img_v_lr

        img_v_fixed_lr = (img_v_lr) / (illu_lr + 1e-4)

        loss_spa = torch.mean(torch.abs(torch.pow(illu_lr - img_v_lr, 2))) * opt.alpha
        loss_tv  = l_TV(illu_lr) * opt.beta
        loss_exp = torch.mean(l_exp(illu_lr)) * opt.gamma
        loss_sparsity = torch.mean(img_v_fixed_lr) * opt.delta


        loss = loss_spa * opt.alpha + loss_tv * opt.beta + loss_exp * opt.gamma + loss_sparsity * opt.delta
        loss.backward()
        optimizer.step()


    img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v)
    img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
    img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)

    Image.fromarray(
        (torch.movedim(img_rgb_fixed,1,-1)[0].detach().cpu().numpy() * 255).astype(np.uint8)
    ).save(os.path.join(opt.output_folder, PATH))

print(' > reconstruction done')
