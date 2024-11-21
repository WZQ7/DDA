import torch
from scipy.io import savemat
import argparse
from trainer import Trainer
from dataset import My_Dataset
import torch.backends.cudnn as cudnn
from utils import get_config
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

loadckpt = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to the config file.')

opts = parser.parse_args()
config = get_config(opts.config)

root = config['dataset']
src_root = root + config['src_root']
tgt_root = root + config['tgt_root']

print(f'src_root is {src_root}')
print(f'tgt_root is {tgt_root}')

##################################################################################
# Initialize Dataset
##################################################################################
start_epoch = config['start_epoch']

train_src = My_Dataset(src_root, src=True, train=True)
test_src = My_Dataset(src_root, src=True, train=False)
train_tgt = My_Dataset(tgt_root, src=False, train=True)

train_loader_src = torch.utils.data.DataLoader(train_src, batch_size=config['batch_size_src'],
                                               shuffle=True, drop_last=True)
test_loader_src = torch.utils.data.DataLoader(test_src, batch_size=config['test_batch_size'],
                                              shuffle=True, drop_last=True)
train_loader_tgt = torch.utils.data.DataLoader(train_tgt, batch_size=config['batch_size_tgt'],
                                               shuffle=True, drop_last=True)


# test_loader_tgt is used only in the digital phantom experiment, which includes a validation set for the target domain.
test_tgt = My_Dataset(tgt_root, src=False, train=False)
test_loader_tgt = torch.utils.data.DataLoader(test_tgt, batch_size=config['test_batch_size'],
                                              shuffle=True, drop_last=True)

cudnn.benchmark = True

total_step = min(len(train_loader_src), len(train_loader_tgt))


##################################################################################
# Initialize Trainer
##################################################################################


def main(index):
    trainer = Trainer(config).to(device)

    # Train
    epoch = config['start_epoch']

    while epoch < config['max_epoch']:
        opt_lrs = [group['lr'] for group in trainer.opt.param_groups]
        dis_lrs = [group['lr'] for group in trainer.dis_opt.param_groups]
        print(f"Epoch {epoch + 1}: Learning Rates = {opt_lrs},{dis_lrs}")
        config['curr_epoch'] = epoch

        for batch_idx, ((ua_s, p0_s), (p0_t)) in enumerate(zip(train_loader_src, train_loader_tgt)):

            # labeled source data
            ua_s = ua_s.to(device)
            p0_s = p0_s.to(device)

            #  unlabeled target data
            p0_t = p0_t.to(device)

            # update discriminators
            loss_dis_f, loss_dis_p = trainer.dis_update(p0_s, p0_t, config['loss_hyper'])

            # update encoder, predictor and decoder
            loss_estim, loss_ae, loss_cyc, loss_adv_f, loss_adv_p = \
                trainer.update(p0_s, ua_s, p0_t, config)

            if (batch_idx + 1) % config['display_iter'] == 0:
                print(f'Epoch [{epoch + 1}], Start [{start_epoch}], Step [{batch_idx + 1}/{total_step}], '
                      f'loss_estim:{(loss_estim.item()):.4f}, loss_ae:{(loss_ae.item()): .4f}, '
                      f'loss_cyc:{(loss_cyc.item()):.4f}, '
                      f'dis:[{(loss_dis_p.item()):.4f}, {(loss_dis_f.item()):.4f}], '
                      f'adv:[{(loss_adv_p.item()):.4f}, {(loss_adv_f.item()):.4f}]')

            if (batch_idx + 1) % config['test_iter'] == 0:
                print('------------------------------test --------------------------------')
                with torch.no_grad():
                    s_loss = torch.tensor(0.0).to(device)
                    s2t_loss = torch.tensor(0.0).to(device)
                    t_loss = torch.tensor(0.0).to(device)

                    num = torch.tensor(0.0).to(device)
                    for test_idx, ((ua_s, p0_s), (ua_t, p0_t)) in enumerate(zip(test_loader_src, test_loader_tgt)):
                        ua_s = ua_s.to(device)
                        p0_s = p0_s.to(device)
                        p0_s = (p0_s - 0.5) * 2

                        f_s = trainer.enc(p0_s)

                        pred_s = trainer.pre(f_s)

                        s_loss += trainer.recon_criterion(pred_s, ua_s)

                        s2t = trainer.gen(f_s)
                        f_s2t = trainer.enc(s2t)
                        pred_s2t = trainer.pre(f_s2t)
                        s2t_loss += trainer.recon_criterion(pred_s2t, ua_s)

                        # The following code is enabled only in the digital phantom experiment
                        # which includes a validation set for the target domain.
                        ua_t = ua_t.to(device)
                        p0_t = p0_t.to(device)
                        p0_t = (p0_t - 0.5) * 2
                        f_t = trainer.enc(p0_t)
                        pred_t = trainer.pre(f_t)
                        t_loss += trainer.recon_criterion(pred_t, ua_t)

                        num = num + 1

                    print(f'Epoch [{epoch + 1}], s: {s_loss/num:.4f}, s2t: {s2t_loss/num:.4f}')

        if (epoch + 1) % config['save_epoch'] == 0:
            save_path = root + config['ckpt_root'] + config['trainer']
            if not os.path.exists(save_path + f'/checkpoint/{index}'):
                os.makedirs(save_path + f'/checkpoint/{index}')
            torch.save({'epoch': epoch+1,
                        'enc': trainer.enc.state_dict(),
                        'dec': trainer.gen.state_dict(),
                        'pre': trainer.pre.state_dict(),
                        'dis_p': trainer.dis_p.state_dict(),
                        'dis_f': trainer.dis_f.state_dict(),
                        }, save_path + f'/checkpoint/{index}'+'/{}.ckpt'.format(epoch+1))

        epoch = epoch + 1

        if config['lr_policy'] == 'cos_wp':
            trainer.update_learning_rate()


#########
# main program
main()



