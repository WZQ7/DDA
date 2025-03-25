import torch
import torch.nn as nn
from network import Enc, Dec, Dis
from utils import get_scheduler, weights_init

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        lr = config['lr']

        self.enc = Enc(config['enc'])
        self.pre = Dec(config['dec'])
        self.apply(weights_init(config['init']))
        params = list(self.enc.parameters()) + list(self.pre.parameters())

        # decoder for reconstructing target images or translating source images into target-like images
        self.gen = Dec(config['gen'])
        self.gen.apply(weights_init(config['init']))
        params += list(self.gen.parameters())

        dis_params = []
        self.dis_f = Dis(2, config['dis'])
        self.dis_f.apply(weights_init('gaussian'))
        dis_params += list(self.dis_f.parameters())

        self.dis_p = Dis(1, config['dis'])
        self.dis_p.apply(weights_init('gaussian'))
        dis_params += list(self.dis_p.parameters())

        adv_params = list(self.enc.parameters()) + list(self.gen.parameters())

        for layer in self.modules():
            if isinstance(layer, nn.InstanceNorm2d):
                nn.init.constant_(layer.bias, 0)  # beta = 0
                nn.init.constant_(layer.weight, 1)  # gamma = 1

        # Setup the optimizers
        beta1 = config['beta1']
        beta2 = config['beta2']

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2))

        self.adv_opt = torch.optim.Adam([p for p in adv_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2))

        self.opt = torch.optim.Adam([p for p in params if p.requires_grad],
                                    lr=lr, betas=(beta1, beta2))

        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.adv_scheduler = get_scheduler(self.adv_opt, config)
        self.scheduler = get_scheduler(self.opt, config)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.adv_scheduler is not None:
            self.adv_scheduler.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def update(self, s, gt, t, hyper):

        s = (s - 0.5) * 2  # [-1,1]
        t = (t - 0.5) * 2  # [-1,1]

        f_s = self.enc(s)
        f_t = self.enc(t)

        s2t = self.gen(f_s)  # [-1,1]
        f_s2t = self.enc(s2t)

        pred_s = self.pre(f_s)
        pred_s2t = self.pre(f_s2t)

        loss_pred = 0.5 * (self.recon_criterion(pred_s, gt) + self.recon_criterion(pred_s2t, gt))

        t2t = self.gen(f_t)  # [-1,1]
        loss_idt = self.recon_criterion(t2t, t)

        self.opt.zero_grad()

        loss = hyper['loss_hyper']['estim'] * loss_pred + hyper['loss_hyper']['recon'] * loss_idt

        loss.backward()

        self.opt.step()

        return loss_pred, loss_idt

    def gan_update(self, s, t, hyper):

        s = (s - 0.5) * 2  # [-1,1]
        t = (t - 0.5) * 2  # [-1,1]

        # update discriminators
        f_s = self.enc(s)
        s2t = self.gen(f_s)

        f_s2t = self.enc(s2t)

        f_t = self.enc(t)

        h_s_avg = torch.mean(f_s, dim=1, keepdim=True)
        h_s_max = torch.max(f_s, dim=1, keepdim=True)[0]
        h_s = torch.cat([h_s_avg, h_s_max], dim=1)

        f = torch.cat((f_t, f_s2t), dim=0)
        h_avg = torch.mean(f, dim=1, keepdim=True)
        h_max = torch.max(f, dim=1, keepdim=True)[0]
        h = torch.cat([h_avg, h_max], dim=1)

        loss_dis_f = self.dis_f.calc_dis_loss(h.detach(), h_s.detach())  # false true
        loss_dis_p = self.dis_p.calc_dis_loss(s2t.detach(), t)

        self.dis_opt.zero_grad()

        loss = hyper['loss_hyper']['gan_p'] * loss_dis_p + hyper['loss_hyper']['gan_f'] * loss_dis_f

        loss.backward()

        self.dis_opt.step()

        # update for adversarial training
        f_s = self.enc(s)
        s2t = self.gen(f_s)

        f_s2t = self.enc(s2t)

        f_t = self.enc(t)

        f = torch.cat((f_t, f_s2t), dim=0)
        h_avg = torch.mean(f, dim=1, keepdim=True)
        h_max = torch.max(f, dim=1, keepdim=True)[0]
        h = torch.cat([h_avg, h_max], dim=1)

        loss_adv_f = self.dis_f.calc_enc_loss(h)
        loss_adv_p = self.dis_p.calc_enc_loss(s2t)

        self.adv_opt.zero_grad()

        loss = hyper['loss_hyper']['gan_p'] * loss_adv_p + hyper['loss_hyper']['gan_f'] * loss_adv_f
        loss.backward()

        self.adv_opt.step()

        return loss_dis_p, loss_dis_f, loss_adv_p, loss_adv_f


from network2 import Enc_Net, Aug_dec, PatchGAN



class Trainer_0218(nn.Module):
    def __init__(self, hyperparameters):
        super(Trainer_0218, self).__init__()
        lr = hyperparameters['lr']
        self.gan_f = hyperparameters['gan_f_w'] != 0
        self.gan_p = hyperparameters['gan_p_w'] != 0
        self.idt = hyperparameters['idt_w'] != 0

        print(f'feature-level adaptation {self.gan_f}')
        print(f'Reconstruct target images{self.idt}')
        print(f'pixel-level adaptation   {self.gan_p}')

        self.enc = Enc_Net(hyperparameters['input_dim'], hyperparameters['fr'])
        # decoder for task
        self.pre = Aug_dec(hyperparameters['input_dim'], hyperparameters['aug_dec'])
        self.apply(weights_init(hyperparameters['init']))
        params = list(self.enc.parameters()) + list(self.pre.parameters())

        dis_params = []
        if self.gan_f:
            self.dis_f = PatchGAN(2, hyperparameters['dis'])
            self.dis_f.apply(weights_init('gaussian'))
            dis_params += list(self.dis_f.parameters())

        # decoder for autoregression
        self.gen = Aug_dec(hyperparameters['input_dim'], hyperparameters['gen'])
        self.gen.apply(weights_init(hyperparameters['init']))
        params += list(self.gen.parameters())

        if self.gan_p:
            self.dis_p = PatchGAN(1, hyperparameters['dis'])
            self.dis_p.apply(weights_init('gaussian'))
            dis_params += list(self.dis_p.parameters())

        adv_params = list(self.enc.parameters()) + list(self.gen.parameters())

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2))

        self.adv_opt = torch.optim.Adam([p for p in adv_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2))

        self.opt = torch.optim.Adam([p for p in params if p.requires_grad],
                                    lr=lr, betas=(beta1, beta2))

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.adv_scheduler = get_scheduler(self.adv_opt, hyperparameters)
        self.scheduler = get_scheduler(self.opt, hyperparameters)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.adv_scheduler is not None:
            self.adv_scheduler.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def update(self, s, gt, t, hyper):

        s = (s - 0.5) * 2  # [-1,1]
        t = (t - 0.5) * 2  # [-1,1]

        f_s = self.enc(s)
        f_t = self.enc(t)

        s2t = self.gen(f_s)  # [-1,1]
        f_s2t = self.enc(s2t)

        pred_s = self.pre(f_s)
        pred_s2t = self.pre(f_s2t)
        if hyper['trainer'] == 'Trainer_0218':
            loss_pred = 0.5 * (hyper['pred_s'] * self.recon_criterion(pred_s, gt) + hyper['pred_s2t'] * self.recon_criterion(pred_s2t, gt))
        else:
            # mask = (gt > 0.041).to(torch.float32) * 0.7 + (gt > 0).to(torch.float32) * 0.3
            mask = (gt > 0.01).to(torch.float32) # for mouse
            loss_pred = 0.5 * ((pred_s*mask - gt*mask).abs().sum() + (pred_s2t*mask - gt*mask).abs().sum())/mask.sum()

        if not self.idt:
            loss_idt = torch.tensor(0)
        else:
            t2t = self.gen(f_t)  # [-1,1]
            loss_idt = self.recon_criterion(t2t, t)

        self.opt.zero_grad()

        loss = hyper['pred_w'] * loss_pred + hyper['idt_w'] * loss_idt

        loss.backward()

        self.opt.step()

        return loss_pred, loss_idt

    def gan_update(self, s, t, hyper):

        s = (s - 0.5) * 2  # [-1,1]
        t = (t - 0.5) * 2  # [-1,1]

        # the first forward calculation
        f_s = self.enc(s)
        s2t = self.gen(f_s)

        f_s2t = self.enc(s2t)

        f_t = self.enc(t)

        h_s_avg = torch.mean(f_s, dim=1, keepdim=True)
        h_s_max = torch.max(f_s, dim=1, keepdim=True)[0]
        h_s = torch.cat([h_s_avg, h_s_max], dim=1)

        f = torch.cat((f_t, f_s2t), dim=0)
        h_avg = torch.mean(f, dim=1, keepdim=True)
        h_max = torch.max(f, dim=1, keepdim=True)[0]
        h = torch.cat([h_avg, h_max], dim=1)

        loss_dis_f = self.dis_f.calc_dis_loss(h.detach(), h_s.detach())  # false true
        loss_dis_p = self.dis_p.calc_dis_loss(s2t.detach(), t)

        # update discriminators
        self.dis_opt.zero_grad()

        loss = hyper['gan_p_w'] * loss_dis_p + hyper['gan_f_w'] * loss_dis_f

        loss.backward()

        self.dis_opt.step()

        # the second forward calculation
        f_s = self.enc(s)
        s2t = self.gen(f_s)

        f_s2t = self.enc(s2t)

        f_t = self.enc(t)

        f = torch.cat((f_t, f_s2t), dim=0)
        h_avg = torch.mean(f, dim=1, keepdim=True)
        h_max = torch.max(f, dim=1, keepdim=True)[0]
        h = torch.cat([h_avg, h_max], dim=1)

        loss_adv_f = self.dis_f.calc_enc_loss(h)
        loss_adv_p = self.dis_p.calc_enc_loss(s2t)

        # update for adversarial training
        self.adv_opt.zero_grad()

        loss = hyper['gan_p_w'] * loss_adv_p + hyper['gan_f_w'] * loss_adv_f
        loss.backward()

        self.adv_opt.step()

        return loss_dis_p, loss_dis_f, loss_adv_p, loss_adv_f