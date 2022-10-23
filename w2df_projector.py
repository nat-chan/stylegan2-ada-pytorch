# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
from ast import Call
from email.policy import default
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from pathlib import Path
from nokogiri.working_dir import working_dir
from nokogiri.defaultdotdict import defaultdotdict
import json
import random
from typing import Union, Callable, Optional

def make_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

import dnnlib

def check(arg):
    print("\x1b[32m"+str(arg.dtype).split('.')[-1] + str(tuple(arg.shape))+"\x1b[m")

def project(
    w2df,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    dist_weight                = .5,
    additional_weight          = .5,
    w_avg                      = None,
    w_std                      = 0, # 20.59920993630580,
    custom_lr                  = -1,
    device: torch.device
):
    G = copy.deepcopy(w2df.net.decoder.G).eval().requires_grad_(False).to(device) # type: ignore
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    def additional_feat(target): # float32(1,3,512,512)
        df = w2df(target[0].permute(1,2,0), imode="illust(512,512,3)")
        return df # float32(1, 1, 512, 512)

    def additional_loss(target, synth):
        """
        .5, .5の設定で
        step    1/1000: dist 0.46 loss 24976.81
        step 1000/1000: dist 0.05 loss 2.76

        .5, .5の設定で 入力sim
        step 1000/1000: dist 0.23 loss 22.98
        step 1000/1000: dist 0.18 loss 12.93
        step 1000/1000: dist 0.24 loss 34.01 | w_std=0
        step   100/100: dist 0.32 loss 166.92  dist2 332.73 | num_steps=100 Elapsed: 28.4 s
        step 1000/1000: dist 0.24 loss 27.90   dist2 55.56 | .5, .5 make_deterministic Elapsed: 268.0 s
        step 1000/1000: dist 0.52 loss 1082.32 dist2 1082.3 | 0, 1
        step 1000/1000: dist 0.22 loss 0.22    dist2 196.1  | 1, 0
        step 1000/1000: dist 0.09 loss 10.06   dist2 20.03  | global_avg 無彩色achromatic std = 20
        step 1000/1000: dist 0.07 loss 8.26    dist2 16.45  | global_avg std = 0
        """
        return (target-synth).square().mean()
    

    # Compute w stats.
    logprint(f'load W midpoint and stddev from w2df')
    if w_avg is None:
        w_avg = w2df.net.latent_avg[0][None,None,:] # [1, 1, C] # float32(1, 1, 512)

    # Setup noise inputs.
    # FIX RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
    noise_bufs = { name: torch.randn_like(buf, requires_grad=True) for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) # float32(1, 3, 512, 512)
    additional_target_features = additional_feat(target_images)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')# float32(1, 3, 256, 256)
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)# float32(1, 7995392)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # float32(1, 1, 512)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
#    for buf in noise_bufs.values():
#        buf[:] = torch.randn_like(buf)
#        buf.requires_grad = True
#
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        if custom_lr != -1:
            if callable(custom_lr):
                lr = custom_lr(t)
            else:
                lr = custom_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale # float32(1, 1, 512)
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1]) # float32(1, 16, 512)
        synth_images = G.synthesis(ws, noise_mode='const')# float32(1, 3, 512, 512)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        synth_np = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        
        additional_synth_features = additional_feat(synth_images)
        additional_dist = additional_loss(additional_target_features, additional_synth_features)

        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')# float32(1, 3, 256, 256)

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)# float32(1, 7995392)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2# float32()
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2# float32()
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist * dist_weight + additional_dist * additional_weight + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        log_txt = f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f} dist2 {additional_dist:<4.2f}'
        logprint(log_txt)

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        yield w_opt.detach()[0].repeat([G.mapping.num_ws, 1]), synth_np, additional_dist

#----------------------------------------------------------------------------

@click.command()
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num_steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--save_video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--sim',                    help='extract skech and simplify input image', type=bool, default=False)
@click.option('--verbose',                help='verbose', type=bool, default=True)
@click.option('--dist_weight',            help='dist_weight', type=float, default=0.5)
@click.option('--additional_weight',      help='additional_weight', type=float, default=0.5)
@click.option('--w_std',                  help='w_std', type=float, default=0)
@click.option('--custom_lr',              help='custom_lr', type=float, default=-1)
def run_projection(
    target_fname: str,
    outdir: str,
    save_video: bool,
    num_steps: int,
    sim: bool,
    verbose: bool,
    dist_weight: float,
    additional_weight: float,
    w_std:  float,
    custom_lr: Union[Callable, float],
):
    make_deterministic()
    stem = Path(target_fname).stem
    device = "cuda"

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    if sim:
        target_tensor = torch.Tensor(target_uint8).to(device)
        target_uint8 = (255.0*w2df(target_tensor, imode="illust(512,512,3)", omode="sim(1,1,512,512)")[0,0,:,:].cpu().detach().numpy()).astype(np.uint8)[:,:,None][:,:,[0,0,0]]

    # Optimize projection.
    projected_w_steps = project(
        w2df,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=verbose,
        dist_weight=dist_weight,
        additional_weight=additional_weight,
        w_std=w_std,
        custom_lr=custom_lr,
    )

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/{stem}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
    start_time = perf_counter()
    for i, (projected_w, synth_np, additional_dist) in enumerate(projected_w_steps):
        if save_video:
            video.append_data(np.concatenate([target_uint8, synth_np], axis=1))# uint8(512, 1024, 3)
        #if i == 0: break
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
    if save_video:
        video.close()

    # Save final projected frame and W vector.
    PIL.Image.fromarray(synth_np, 'RGB').save(f'{outdir}/{stem}.png')
    np.savez(f'{outdir}/{stem}.npz', w=projected_w.unsqueeze(0).cpu().numpy())
    with open(f'{outdir}/{stem}.txt', "w") as f:
        f.write(str(float(additional_dist)))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    with working_dir("/home/natsuki/pixel2style2pixel"):
        from script_w2df import W2DF
        w2df = W2DF()

    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
