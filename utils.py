import yaml
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import math
from torch.optim import lr_scheduler
from typing import Optional, Any, Tuple
from torch.autograd import Variable, Function


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'cos_wp':
        warm_up_iter = hyperparameters['warm_up_iter']
        T_max = hyperparameters['T_max']  # 周期
        lr_start = hyperparameters['lr_start']
        lr_max = hyperparameters['lr_max']
        lr_min = hyperparameters['lr_min']
        start_epoch = hyperparameters['start_epoch']
        lambda0 = lambda cur_iter: (1-lr_start)*((cur_iter+start_epoch) / warm_up_iter) + lr_start\
            if (cur_iter+start_epoch) < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos(((cur_iter+start_epoch)-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


# B C H W
# mean 和 sum 函数指定在多个维度上进行， 但是 max 和 min 一次只能处理一个维度，如果不想连续的使用可以把要求最大值的多个维度合并为一个维度
# reshape 和 view 都是内存视图操作，它们不会复制数据，而是返回一个新的张量视图，这些视图共享底层数据。
def cal_mae(input, target, mask):
    err = torch.abs(input - target)*mask
    return torch.sum(err,dim=[1,2,3])/mask.sum(dim=[1,2,3])

def cal_mse(input, target, mask):
    err = (input - target)**2*mask
    return torch.sum(err,dim=[1,2,3])/mask.sum(dim=[1,2,3])

def cal_mre(input, target, mask):
    re_err = torch.abs(input - target)*mask/(target+(mask==0)*1e-8)
    return torch.sum(re_err,dim=[1,2,3])/mask.sum(dim=[1,2,3])

def cal_psnr(input, target, mask):
    b, *_ = input.shape
    rmse = torch.sqrt(torch.sum((input - target)**2*mask,dim=[1,2,3])/mask.sum(dim=[1,2,3]))
    max_ = torch.max(target.view(b,-1),dim=1)[0]
    return 20*torch.log10(max_/rmse)
