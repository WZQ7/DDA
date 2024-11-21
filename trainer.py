import torch
import torch.nn as nn
from network import Enc, Dec, Dis
from utils import get_scheduler, weights_init
import numpy as np


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        lr = config['lr']

        self.enc = Enc(config['enc'])
        self.pre = Dec(config['dec'])
        self.apply(weights_init(config['init']))
        params = list(self.enc.parameters()) + list(self.pre.parameters())

        self.dec = Dec(config['dec'])
        self.dec.apply(weights_init(config['init']))
        params += list(self.dec.parameters())

        dis_params = []

        self.dis_f = Dis(1, config['dis'])
        self.dis_f.apply(weights_init('gaussian'))
        dis_params += list(self.dis_f.parameters())

        self.dis_p = Dis(1, config['dis'])
        self.dis_p.apply(weights_init('gaussian'))
        dis_params += list(self.dis_p.parameters())

        # Setup the optimizers
        beta1 = config['beta1']
        beta2 = config['beta2']
        weight_decay = config['weight_decay']

        self.opt = torch.optim.Adam([p for p in params if p.requires_grad],
                                     lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

        self.scheduler = get_scheduler(self.opt, params)
        self.dis_scheduler = get_scheduler(self.dis_opt, params)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def update(self, s, gt, t, hyper):

        device = s.device
        s = (s - 0.5) * 2  # [-1,1]
        t = (t - 0.5) * 2  # [-1,1]

        f_s = self.enc(s)
        f_t = self.enc(t)

        h_t = f_t.mean(dim=1, keepdim=True)
        loss_adv_f = self.dis_f.calc_enc_loss(h_t)

        t2t = self.gen(f_t)  # [-1,1]
        loss_ae = self.recon_criterion(t2t, t)

        s2t = self.dec(f_s)
        loss_adv_p = self.dis_p.calc_enc_loss(s2t)

        f_s2t = self.enc(s2t)
        loss_cyc = self.recon_criterion(f_s2t, f_s)

        a = torch.FloatTensor(np.random.random((f_s2t.size(0), 1, 1, 1))).to(device) * self.alpha
        f_mix = (a * f_s2t + ((1 - a) * f_s))

        ua_estim = self.pre(f_mix)
        loss_estim = self.recon_criterion(ua_estim, gt)
        self.opt.zero_grad()

        loss = hyper['estim'] * loss_estim + hyper['ae'] * loss_ae + \
               hyper['cyc'] * loss_cyc + \
               hyper['gan_f'] * loss_adv_f + hyper['gan_p'] * loss_adv_p

        loss.backward()

        self.opt.step()

        return loss_estim, loss_ae, loss_cyc, loss_adv_f, loss_adv_p

    def dis_update(self, s, t, hyper):

        s = (s - 0.5) * 2  # [-1,1]
        t = (t - 0.5) * 2  # [-1,1]
        f_s = self.enc(s)
        f_t = self.enc(t)

        h_s = f_s.mean(dim=1, keepdim=True)
        h_t = f_t.mean(dim=1, keepdim=True)
        loss_dis_f = self.dis_f.calc_dis_loss(h_t.detach(), h_s.detach())

        s2t = self.gen(f_s)
        loss_dis_p = self.dis_p.calc_dis_loss(s2t.detach(), t)

        self.dis_opt.zero_grad()

        loss = hyper['gan_f'] * loss_dis_f + hyper['gan_p'] * loss_dis_p

        loss.backward()

        self.dis_opt.step()

        return loss_dis_f, loss_dis_p







