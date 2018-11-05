import os
import time

import torch
import torchvision
import pro_gan_pytorch.PRO_GAN as pg

from dataset import TextDataset

PARENT_DIR = os.path.dirname(__file__)

# select the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = 'ckpt-coco'

def setup_data(batch_size, size):
    """
    setup the dataset for training the CNN
    :param batch_size: batch_size for sgd
    :return: classes, trainloader, testloader => training and testing data loaders
    """

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    trainset = TextDataset(path=os.path.join(PARENT_DIR, 'coco_captions.txt'), size=size, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    # testset = torchvision.datasets.CIFAR10(root=data_path,
                                  # transform=transforms, train=False,
                                  # download=False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          # shuffle=True,
                                          # num_workers=num_workers)

    return trainloader#, testloader


def save_checkpoint(pro_gan, current_depth, epoch, ckpt_dir):
    state = {
        'gen':pro_gan.gen.state_dict(),
        'gen_optim':pro_gan.gen_optim.state_dict(),
        'dis':pro_gan.dis.state_dict(),
        'dis_optim':pro_gan.dis_optim.state_dict(),
        'current_depth': current_depth,
        'epoch': epoch,
    }
    os.makedirs(os.path.join(PARENT_DIR, ckpt_dir), exist_ok=True)
    basename = 'checkpoint-{}-{}.ckpt'.format(current_depth, epoch)
    filename = os.path.join(PARENT_DIR, ckpt_dir, basename)
    torch.save(state, filename)
    checkpoint_file = os.path.join(PARENT_DIR, ckpt_dir, 'checkpoint')
    with open(checkpoint_file, 'w') as f:
        f.write(basename)
    print('Saved {}'.format(basename))


def load_checkpoint(ckpt_dir):
    checkpoint_file = os.path.join(PARENT_DIR, ckpt_dir, 'checkpoint')
    if not os.path.isfile(checkpoint_file):
        return None
    with open(checkpoint_file) as f:
        basename = f.read()
    filename = os.path.join(PARENT_DIR, ckpt_dir, basename)
    state = torch.load(filename)
    print('Loaded {}'.format(basename))
    return state


if __name__ == '__main__':

    # some parameters:
    depth = 6
    # num_epochs = 100  # number of epochs per depth (resolution)
    num_epochs = 10  # number of epochs per depth (resolution)
    latent_size = 512

    # get the data. Ignore the test data and their classes
    # _, train_data_loader, _ = setup_data(batch_size=32, num_workers=3, download=True)
    train_data_loader = setup_data(batch_size=128, size=128000)

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = pg.ProGAN(depth=depth, latent_size=latent_size, device=device)
    # ======================================================================

    start = time.time()

    # train the pro_gan using the cifar-10 data
    for current_depth in range(depth):
        # print("working on depth:", current_depth)

        # note that the rest of the api indexes depth from 0
        for epoch in range(1, num_epochs + 1):
            # print("\ncurrent_epoch: ", epoch)

            # calculate the value of aplha for fade-in effect
            # alpha = epoch / num_epochs
            # print("value of alpha:", alpha)

            # iterate over the dataset in batches:
            for i, batch in enumerate(train_data_loader, 1):
                alpha = (epoch - 1 + i / len(train_data_loader)) / num_epochs

                images = batch
                images = images.to(device)
                # generate some random noise:
                noise = torch.randn(images.shape[0], latent_size).to(device)

                # optimize discriminator:
                dis_loss = pro_gan.optimize_discriminator(noise, images, current_depth, alpha)

                # optimize generator:
                gen_loss = pro_gan.optimize_generator(noise, current_depth, alpha)

                end = time.time()
                delay = end - start
                start = end

                print("Depth: %d Epoch: %d Batch: %d  dis_loss: %.3f  gen_loss: %.3f  time: %.3f"
                      % (current_depth, epoch, i, dis_loss, gen_loss, delay))

            # print("epoch finished ...")
            if epoch % 1 == 0:
                save_checkpoint(pro_gan, current_depth, epoch, ckpt_dir)
    print("training complete ...")
    save_checkpoint(pro_gan, current_depth, epoch, ckpt_dir)
