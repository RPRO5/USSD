import torch
import glob
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
import itertools
import matplotlib.pyplot as plt



class TwoStreamBatchSampler(Sampler):

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)



class Slit_loader(torch.utils.data.Dataset):

    def __init__(self, dataset_path, scale, k_fold_test=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.scale = scale
        if mode != 'test':
            self.img_path = dataset_path + '/train' + '/img'
            self.mask_path = dataset_path + '/train' + '/mask'
        else:
            self.img_path = dataset_path + '/test' + '/img'
            self.mask_path = dataset_path + '/test' + '/mask'
        self.image_lists, self.label_lists = self.read_list(self.img_path, k_fold_test=k_fold_test)
        self.resize_label = transforms.Resize(scale, Image.BILINEAR)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        self.to_gray = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor() 
        
    def __getitem__(self, index):
        img = Image.open(self.image_lists[index])
        img = self.resize_img(img)
        
        img = np.array(img).astype(np.uint8)
        labels = self.label_lists[index]
        if self.mode != 'test':
            label = Image.open(self.label_lists[index])
            label = self.resize_label(label)
            label = np.array(label).astype(np.uint8) 
            label = label.reshape(1, label.shape[0], label.shape[1])
            label = (label-127.5)/127.5
            label_img = torch.from_numpy(label.copy()).float()

            if self.mode == 'val' or self.mode == 'test':
                assert len(os.listdir(os.path.dirname(self.image_lists[index]))) == len(
                    os.listdir(os.path.dirname(labels)))
                img_num = len(os.listdir(os.path.dirname(labels)))
                labels = (label_img, img_num)  
            else:
                labels = label_img  

        img = (img.reshape(1, img.shape[0], img.shape[1])-127.5)/127.5
        img = torch.from_numpy(img.copy()).float()
        return img, labels,self.image_lists[index]

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path, k_fold_test=1):
        fold = sorted(os.listdir(image_path))
        print("fold =========== {}".format(fold))
        img_list = []
        label_list = []

        if self.mode == 'train':
            fold_r = list(fold)
            fold_r.remove('f' + str(k_fold_test))  # remove testdata
            for item in fold_r:
                    cube_path = os.path.join(image_path, item)
                    img_list += glob.glob(cube_path + '/*.png')
            label_list = [x.replace("img","mask") for x in img_list]

        elif self.mode == 'val':
            fold_s = fold[k_fold_test-1]
            cube_path = os.path.join(image_path, fold_s).replace('train', 'valid')
            img_list += glob.glob(cube_path + '/*.png')
            label_list = [x.replace("img","mask") for x in img_list]


        elif self.mode == 'test':
            fold_t = fold[k_fold_test-1]
            cube_path = os.path.join(image_path, fold_t).replace('train', 'valid')
            img_list += glob.glob(cube_path + '/*.png')
            label_list = [x.replace("img","mask") for x in img_list]

        assert len(img_list) == len(label_list)
        print('Total {} image is:{}'.format(self.mode, len(img_list)))

        return img_list, label_list
