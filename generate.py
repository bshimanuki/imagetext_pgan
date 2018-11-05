import os

import imageio
import numpy as np
import skimage
import torch
import torchvision
import pro_gan_pytorch.PRO_GAN as pg

PARENT_DIR = os.path.dirname(__file__)

# select the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = 'ckpt-coco'
out_dir = 'generated-coco'


def load_checkpoint(pro_gan, ckpt_dir, basename=None):
    if basename is None:
        checkpoint_file = os.path.join(PARENT_DIR, ckpt_dir, 'checkpoint')
        if not os.path.isfile(checkpoint_file):
            return None
        with open(checkpoint_file) as f:
            basename = f.read()
    filename = os.path.join(PARENT_DIR, ckpt_dir, basename)
    state = torch.load(filename)
    pro_gan.gen.load_state_dict(state['gen'])
    pro_gan.gen_optim.load_state_dict(state['gen_optim'])
    pro_gan.dis.load_state_dict(state['dis'])
    pro_gan.dis_optim.load_state_dict(state['dis_optim'])
    current_depth = state['current_depth']
    epoch = state['epoch']
    print('Loaded {}'.format(basename))
    return current_depth, epoch


if __name__ == '__main__':

    # some parameters:
    depth = 6
    latent_size = 512


    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = pg.ProGAN(depth=depth, latent_size=latent_size, device=device)
    # gen = pg.Generator(depth=depth, latent_size=latent_size, use_eql=False).to(device)
    # ======================================================================

    current_depth, epoch = load_checkpoint(pro_gan, ckpt_dir)
    # if epoch != 10:
        # current_depth -= 1

    # OUT_DIR = out_dir
    # for current_depth in range(depth):
    current_depth -= 1
    if True:
        load_checkpoint(pro_gan, ckpt_dir, basename='checkpoint-{}-10.ckpt'.format(current_depth))
        # out_dir = '{}-{}'.format(OUT_DIR, current_depth)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(10):
            noise = torch.randn(1, latent_size).to(device)

            sample_image = pro_gan.gen(noise, depth=current_depth, alpha=1).detach()
            # sample_image = gen(noise, depth=3, alpha=1).detach()
            img = sample_image[0].permute(1, 2, 0) / 2 + 0.5
            # img = sample_image[0].permute(1, 2, 0)

            img = img.cpu().numpy()
            img = np.clip(img, 0, 1)
            img = skimage.img_as_ubyte(img)
            path = os.path.join(out_dir, '{}.png'.format(i))
            imageio.imwrite(path, img)
