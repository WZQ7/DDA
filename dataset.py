import os
import os.path
import torch
import torch.utils.data as data
import numpy as np
import scipy.io as scio


def max_norm(image):

    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
            _max = image.max()
            narmal_image = image/_max
    else:
        num = image.shape[0]
        for i in range(num):
            _max = image[i].max()
            image[i] = image[i]/_max

        narmal_image = image

    return narmal_image, _max


class My_Dataset(data.Dataset):
    def __init__(self, root, src=True, train=True):
        self.src = src
        self.train = train
        if self.src:
            self.__p0_s = []
            self.__ua_s = []
            self.__mask = []
            self.root = os.path.expanduser(root)
        else:
            self.__p0_t = []
            self.__ua_t = []
            self.root = os.path.expanduser(root)

        if self.train:
            folder = self.root + "train_data/"
        else:
            folder = self.root + "test_data/"

        if self.src:
            for file in os.listdir(folder):
                matdata = scio.loadmat(folder + file)
                p0_s = matdata['p0_data']
                ua_s = matdata['ua_data']
                __len = p0_s.shape[0]
                for i in np.arange(int(__len)):
                    self.__ua_s.append(ua_s[i][np.newaxis, :, :])  # 1,batch_size,H,W
                    self.__p0_s.append(p0_s[i][np.newaxis, :, :])


        else:
            for file in os.listdir(folder):
                matdata = scio.loadmat(folder + file)

                p0_t = matdata['p0_recon_data']

                __len = p0_t.shape[0]

                if not train:
                    ua_t = matdata['ua_data']  # target domain labels only used for evaluation

                __len = p0_t.shape[0]
                for i in np.arange(int(__len)):
                    self.__p0_t.append(p0_t[i][np.newaxis, :, :])
                    if not train:
                        self.__ua_t.append(ua_t[i][np.newaxis, :, :])

    def __getitem__(self, index):
        if self.src:
            p0_s = self.__p0_s[index]
            p0_s, scale = max_norm(p0_s)
            p0_s = torch.Tensor(p0_s)

            ua_s = self.__ua_s[index]
            ua_s = torch.Tensor(ua_s)

            return ua_s, p0_s

        else:
            p0_t = self.__p0_t[index]
            p0_t, scale = max_norm(p0_t)
            p0_t = torch.Tensor(p0_t)

            if not self.train:
                ua_t = self.__ua_t[index]
                ua_t = torch.Tensor(ua_t)

            return p0_t if self.train else (ua_t, p0_t)

    def __len__(self):
        return len(self.__p0_s) if self.src else len(self.__p0_t)


