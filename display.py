import torch
import matplotlib.pyplot as plt
import argparse
from trainer import Trainer
from dataset import My_Dataset
from utils import get_config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = get_config('./config.yaml')

'''
Option:
    1. digital : synthetic-to-synthetic study on digital phantoms, including 2 example images
    2. agar: synthetic-to-real study on agar phantoms, including 2 example images
    3. mouse: synthetic-to-real study on in vivo mice, including 2 example images
'''
dataset_type = 'digital'
example_index = 1  # choose a certain example image: 1 or 2


trainer = Trainer(config)
trainer = trainer.to(device)

# load network parameters
loadcp = True
if loadcp:
    checkpoint = torch.load('./data/' + dataset_type + '/model/model.ckpt')
    trainer.enc.load_state_dict(checkpoint['enc'])
    trainer.pre.load_state_dict(checkpoint['pre'])

test_tgt = My_Dataset('./data/' + dataset_type + '/', src=False, train=False)
test_batch = 1  # Get one image by one iteration
test_loader_tgt = torch.utils.data.DataLoader(test_tgt, batch_size=test_batch,
                                              shuffle=True, drop_last=True)
iterator = iter(test_loader_tgt)

print('Number of images:{}'.format(test_loader_tgt.__len__()))

with torch.no_grad():
    for i in range(example_index):
        (ua_t, p0_t) = next(iterator)  # torch.Size([batch, 1, H, W])

    p0_t = p0_t.to(device)
    p0_t = (p0_t-0.5) * 2
    outputs = trainer.pre(trainer.enc(p0_t))
    ua_recon = outputs.squeeze().cpu()

    if dataset_type == 'digital':
        vmax = 0.32
    elif dataset_type == 'agar':
        vmax = 0.3
    else:
        vmax = 0.7

    p0_t = p0_t.squeeze().cpu()
    p0_t = p0_t/2 + 0.5

    if dataset_type != 'mouse':
        ua_t = ua_t.squeeze()  # Ground Truth

    # Visualization
    plt.figure()
    plt.subplot(1, 3, 1)
    if dataset_type != 'mouse':
        plt.imshow(p0_t, vmin=0, vmax=1, cmap=plt.cm.jet, interpolation='none')
    else:
        plt.imshow(p0_t, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='none')
    plt.title("p0")

    if dataset_type != 'mouse':
        plt.subplot(1, 3, 2)
        if dataset_type == 'digital':
            plt.imshow(ua_t, vmin=0, vmax=vmax, cmap=plt.cm.jet, interpolation='none')
            plt.title("Real ua")
        else:
            plt.imshow(ua_t, vmin=0, vmax=vmax, interpolation='none')
            plt.title("Reference ua")

    plt.subplot(1, 3, 3)
    if dataset_type == 'digital':
        plt.imshow(ua_recon, vmin=0, vmax=vmax, cmap=plt.cm.jet, interpolation='none')
    else:
        plt.imshow(ua_recon, vmin=0, vmax=vmax, interpolation='none')

    plt.minorticks_on()
    plt.title("Reconstructed ua")
    plt.show()

