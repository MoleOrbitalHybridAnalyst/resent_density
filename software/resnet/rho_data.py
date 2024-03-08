import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class RhoData(Dataset):
    def __init__(self, list_data, list_label, list_data_gridsizes, list_label_gridsizes, data_augmentation=True, downsample_data=1, downsample_label=1):
        self.ds_data = downsample_data
        self.ds_label = downsample_label
        self.da = data_augmentation

        self.list_data = np.genfromtxt(
            list_data, dtype=str)
        self.list_data_gs = np.genfromtxt(
            list_data_gridsizes, dtype=str)

        self.list_label = np.genfromtxt(
            list_label, dtype=str)
        self.list_label_gs = np.genfromtxt(
            list_label_gridsizes, dtype=str)

        assert self.list_data.size == self.list_data_gs.size
        assert self.list_data.size == self.list_label.size
        assert self.list_data.size == self.list_label_gs.size

    def __len__(self):
        return self.list_data.size

    def rotate_x(self, data_in):
        '''
        rotate 90 by x axis
        '''
        return data_in.transpose(-1,-2).flip(-1)

    def rotate_y(self, data_in):
        return data_in.transpose(-1,-3).flip(-1)

    def rotate_z(self, data_in):
        return data_in.transpose(-2,-3).flip(-2)

    def rand_rotate(self, data_lst):
        rint = np.random.randint(3)
        if rint == 0:
            rotate = lambda d: self.rotate_x(d)
        elif rint == 1:
            rotate = lambda d: self.rotate_y(d) 
        else:
            rotate = lambda d: self.rotate_z(d)
        r = np.random.rand()
        if r < 0.1:
            return data_lst
        elif r < 0.4:
            return [rotate(d) for d in data_lst]
        elif r < 0.7:
            return [rotate(rotate(d)) for d in data_lst]
        else:
            return [rotate(rotate(rotate(d))) for d in data_lst]

    def __getitem__(self, idx):
        rho1 = torch.tensor(
            np.load(self.list_data[idx]), dtype=torch.float32)
        size = np.loadtxt(self.list_data_gs[idx], dtype=int)
        rho1 = rho1.reshape(1, *size)

        rho2 = torch.tensor(
            np.load(self.list_label[idx]), dtype=torch.float32)
        size = np.loadtxt(self.list_label_gs[idx], dtype=int)
        rho2 = rho2.reshape(1, *size)

        if self.da:
            rho1, rho2 = self.rand_rotate([rho1, rho2])

        ds1 = self.ds_data
        ds2 = self.ds_label
        nx, ny, nz = rho1.size()[-3:]
        nx = nx // ds1 * ds1
        ny = ny // ds1 * ds1
        nz = nz // ds1 * ds1
        rho1 = rho1[..., :nx:ds1,:ny:ds1,:nz:ds1]
        nx, ny, nz = rho2.size()[-3:]
        nx = nx // ds1 * ds1
        ny = ny // ds1 * ds1
        nz = nz // ds1 * ds1
        rho2 = rho2[..., :nx:ds2,:ny:ds2,:nz:ds2]

        return rho1, rho2
