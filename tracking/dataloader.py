from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
import numpy as np


class Gtdataset(Dataset):
    def __init__(self, data_path='../MOT17/train'):
        self.images = []
        self.target = []
        for vedio in os.listdir(data_path):
            video_path = os.path.join(data_path, vedio)
            f = open(os.path.join(video_path, 'gt', 'gt.txt'), 'r')
            txt = f.readlines()
            maps = {}
            for str_data in txt:
                data = str_data.split(',')
                data = list(map(float, data))
                frame = data[0]
                if frame not in maps:
                    maps[frame] = []
                if data[6]:
                    data[2] = data[2]/1920+data[4]/1920/2
                    data[3] = data[3]/1080+data[5]/1080/2
                    data[4] = data[4]/1920
                    data[5] = data[5]/1080
                    res_data = data[0:6]+data[7:8]
                    res_data[1], res_data[6] = res_data[6], res_data[1]
                    maps[frame].append(res_data)
            for img in os.listdir(os.path.join(video_path, 'img1')):
                if int(img.split('.')[0]) in maps:
                    self.images.append(os.path.join(video_path, 'img1', img))
                    data = np.array(maps[int(img.split('.')[0])])
                    b = np.zeros((1, 7))
                    rows = len(data)
                    for i in range(100-rows):
                        data = np.row_stack((data, b))
                    self.target.append(data[0:50])

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.resize(img, (640, 640))
        targets = self.target[index]
        targets = torch.from_numpy(targets).float()
        inpunt = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        inpunt = np.ascontiguousarray(inpunt)
        inpunt = torch.from_numpy(inpunt)
        inpunt = inpunt.cuda().float() / 255.0
        return inpunt, targets

    def __len__(self):
        return len(self.images)


class Detdataset(Dataset):
    def __init__(self, data_path='../MOT17/train'):
        self.images = []
        self.target = []
        for vedio in os.listdir(data_path):
            video_path = os.path.join(data_path, vedio)
            f = open(os.path.join(video_path, 'det', 'det.txt'), 'r')
            txt = f.readlines()
            maps = {}
            for str_data in txt:
                data = str_data.split(',')
                data = list(map(float, data))
                frame = data[0]
                if frame not in maps:
                    maps[frame] = []
                data[2] = data[2]/1920+data[4]/1920/2
                data[3] = data[3]/1080+data[5]/1080/2
                data[4] = data[4]/1920
                data[5] = data[5]/1080
                data[1], data[6] = data[6], data[1]
                maps[frame].append(data[0:7])
            for img in os.listdir(os.path.join(video_path, 'img1')):
                if int(img.split('.')[0]) in maps:
                    # for t in maps[int(img.split('.')[0])]:
                    self.images.append(os.path.join(video_path, 'img1', img))
                    data = np.array(maps[int(img.split('.')[0])])
                    b = np.zeros((1, 7))
                    rows = len(data)
                    for i in range(100-rows):
                        data = np.row_stack((data, b))
                    self.target.append(data[0:100])

    def __getitem__(self, index):

        img = cv2.imread(self.images[index])
        img = cv2.resize(img, (640, 640))
        targets = self.target[index]
        targets = torch.from_numpy(targets).float()
        inpunt = img.transpose(2, 0, 1)  # BGR to RGB
        inpunt = np.ascontiguousarray(inpunt)
        inpunt = torch.from_numpy(inpunt)
        inpunt = inpunt.cuda().float() / 255.0
        return inpunt, targets

    def __len__(self):
        return len(self.images)


class Shuffledataset(Dataset):
    def __init__(self, data_path='source'):
        self.data_path = data_path
        self.images = []
        self.target = []
        for f in os.listdir(data_path):
            if f.endswith('.jpg'):
                self.images.append(f)
                self.target.append(f.split('.')[0]+'.txt')

    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.data_path, self.images[index]))
        img = cv2.resize(img, (640, 640))
        targets = np.loadtxt(os.path.join(self.data_path, self.target[index]))
        targets = torch.from_numpy(targets).float()
        inpunt = img.transpose(2, 0, 1)  # BGR to RGB
        inpunt = np.ascontiguousarray(inpunt)
        inpunt = torch.from_numpy(inpunt)
        inpunt = inpunt.cuda().float() / 255.0
        return inpunt, targets

    def __len__(self):
        return len(self.images)
