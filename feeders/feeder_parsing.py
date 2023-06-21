import torch
import pickle
import numpy as np
from PIL import ImageFile
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from feeders import gen_parsing

class Feeder(torch.utils.data.Dataset):
    def __init__(self, sample_path, label_path, split, random_interval=False, temporal_rgb_frames=9, debug=False):
        self.debug = debug
        self.sample_path = sample_path
        self.label_path = label_path
        self.split = split
        self.random_interval = random_interval
        self.temporal_rgb_frames = temporal_rgb_frames

        self.load_data()
        if self.split.lower() == 'train':
            self.transform = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.Resize(size=(299)), # inceptionv3
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(299)), # inceptionv3
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.palette = self.get_palette(num_cls=20)
        self.aug = True if self.split.lower() == 'train' else False

    def load_data(self):
        self.sample_name = np.loadtxt(self.sample_path, dtype = str)
        self.label = np.loadtxt(self.label_path, dtype = int)
        if self.debug:
            self.label = self.label[0:100]
            self.sample_name = self.sample_name[0:100]
    
    def get_palette(self, num_cls):
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        label = self.label[index]
        filename = self.sample_name[index]
        img = gen_parsing.gen_featuremap(filename, self.palette, self.random_interval, self.temporal_rgb_frames, self.aug)
        img = self.transform(img)
        return img, label
