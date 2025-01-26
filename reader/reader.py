import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[0]
    anno.ratio = line[9]
    anno.LeftEyeCorner, anno.RightEyeCorner, anno.FaceCorner = line[11], line[12], line[10]
    anno.point2d, anno.head2d = line[6], line[7]
    return anno

def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno

def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_RTGene(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.name = line[0]
    return anno

def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.ethtrain = Decode_ETH
    mapping.rtgene = Decode_RTGene
    return mapping

def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)

def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

class trainloader(Dataset): 
    def __init__(self, dataset):
        # Read source data
        self.data = edict() 
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        if isinstance(dataset.label, list):
            for i in dataset.label:
                with open(i) as f: 
                    line = f.readlines()
                if dataset.header: 
                    line.pop(0)
                self.data.line.extend(line)
        else:
            with open(dataset.label) as f: 
                self.data.line = f.readlines()
            if dataset.header: 
                self.data.line.pop(0)

        # build transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data.line)

    def __getitem__(self, idx):
        # Read source information
        line = self.data.line[idx]
        line = line.strip().split(" ")
        anno = self.data.decode(line)

        img_path = os.path.join(self.data.root, anno.face).replace("\\", "/")
        # print()
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Unable to read image at {img_path}. Skipping this image.")
            return self.__getitem__((idx + 1) % len(self.data.line))

        img = self.transforms(img)

        LeftEyeCorner = np.array(anno.LeftEyeCorner.split(",")).astype("float")
        RightCorner = np.array(anno.RightEyeCorner.split(",")).astype("float")
        FaceCorner = np.array(anno.FaceCorner.split(",")).astype("float")
        label = np.array(anno.point2d.split(",")).astype("float")
        ratio = np.array(anno.ratio.split(",")).astype("float")

        LeftEyeCorner = torch.from_numpy(LeftEyeCorner).type(torch.FloatTensor)
        RightCorner = torch.from_numpy(RightCorner).type(torch.FloatTensor)
        FaceCorner = torch.from_numpy(FaceCorner).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)
        ratio = torch.from_numpy(ratio).type(torch.FloatTensor)

        data = edict()
        data.face = img
        data.name = anno.name
        data.LeftEyeCorner = LeftEyeCorner
        data.RightEyeCorner = RightCorner
        data.FaceCorner = FaceCorner
        data.ratio = ratio

        return data, label

def loader(source, batch_size, shuffle=True, num_workers=0):
    dataset = trainloader(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load