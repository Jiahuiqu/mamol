import scipy.io as sio
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import random
from matplotlib import pyplot as plt
import h5py

class LIDARHS(Dataset):
    def __init__(self, patchsize, mode = 'train', classnum = 150):
        if mode == 'train':
            print('train')
        self.dataset = 3
        self.mode = mode
        self.padding_layers = int(patchsize /2) 
        self.lidar_mat = np.squeeze(sio.loadmat('/datasets/data_augsburg/LiDAR_data.mat')['LiDAR_data'].astype(np.float32))
        self.lidar_mat = np.pad(self.lidar_mat, self.padding_layers, mode='symmetric')
        self.sar_mat = np.squeeze(sio.loadmat('/datasets/data_augsburg/SAR_data.mat')['SAR_data'].astype(np.float32))
        self.sar_mat = np.pad(self.sar_mat, ((self.padding_layers, self.padding_layers), (self.padding_layers, self.padding_layers), (0, 0)),
                        mode='symmetric')
        self.HS_mat = np.squeeze(sio.loadmat('/datasets/data_augsburg/HSI_data.mat')['HSI_data'].astype(np.float32))

        self.HS_mat = np.pad(self.HS_mat, ((self.padding_layers, self.padding_layers), (self.padding_layers, self.padding_layers), (0, 0)),
                        mode='symmetric')
        self.test_lable_os = '/datasets/data_augsburg/All_Label.mat'
        self.lable_mat = np.squeeze(sio.loadmat(self.test_lable_os)['All_Label'].astype(np.float32))

        self.lidar = []
        self.HS = []
        self.sar = []
        self.lable = []
        self.ik = []
        for i in range(0, len(self.lable_mat)):
            for k in range(0, len(self.lable_mat[0])):
                ik_now = (i,k)
                lable_now = self.lable_mat[i][k]
                if lable_now != 0:
                    self.lable.append(lable_now)
                    lidar_now = self.lidar_mat[i:i + (self.padding_layers * 2), k:k + (self.padding_layers * 2)]

                    lidar_now = np.expand_dims(lidar_now, axis=0)
                    HS_now = self.HS_mat[i:i + (self.padding_layers*2), k:k + (self.padding_layers*2)]

                    HS_now = np.transpose(HS_now, (2, 0, 1))
                    sar_now = self.sar_mat[i:i + (self.padding_layers*2), k:k + (self.padding_layers*2)]
                    sar_now = np.transpose(sar_now, (2, 0, 1))
                    self.sar.append(sar_now)
                    self.lidar.append(lidar_now)
                    self.HS.append(HS_now)
                    self.ik.append(ik_now)

        if mode == 'train':

            self.train_hsi = []
            self.train_lidar = []
            self.train_sar = []
            self.train_lable = []
            self.train_ik = []
            for cls in range(1,int(max(self.lable)) + 1):
                indices = [index for index, value in enumerate(self.lable) if value == cls]
                random_indices = random.sample(indices, classnum)
                self.train_hsi += [self.HS[index] for index in random_indices]
                self.train_lidar += [self.lidar[index] for index in random_indices]
                self.train_sar += [self.sar[index] for index in random_indices]
                self.train_lable += [self.lable[index] for index in random_indices]
                self.train_ik += [self.ik[index] for index in random_indices]
        print(np.array(self.train_hsi).shape,np.array(self.train_sar).shape,np.array(self.train_lable).shape)
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_lable)
        else:
            return len(self.lable)
    def __getitem__(self, item):
        if self.mode == 'train':
            lidar, HS, sar, lable, ik = self.train_lidar[item], self.train_hsi[item], self.train_sar[item],self.train_lable[item], self.train_ik[item]
        else:
            lidar, HS, sar, lable, ik = self.lidar[item], self.HS[item],self.sar[item], self.lable[item], self.ik[item]
        sample = {
                "HS": torch.tensor(HS, dtype=torch.float32),
                "lidar": torch.tensor(lidar, dtype=torch.float32),
                "sar": torch.tensor(sar, dtype=torch.float32),
                "label": torch.tensor(int(lable) - 1, dtype=torch.long),
                "ik": ik,
            }
        if self.dataset == 0:return lidar, HS, lable, ik
        if self.dataset == 1:return sar, HS, lable, ik
        if self.dataset == 2:return lidar, HS, sar, lable, ik
        if self.dataset == 3:return sample
