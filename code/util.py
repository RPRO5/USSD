from __future__ import print_function
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

import numpy as np
from PIL import Image
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()  
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

from skimage import img_as_ubyte
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    new_size = (640, 480)  
    image_pil = image_pil.resize(new_size)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uint82bin(n, count=8):
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

from torch.utils.data.sampler import Sampler
import itertools
class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        super(TwoStreamBatchSampler, self).__init__()
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

def adjust_learning_rate(opt, optimizer, epoch):
    opt.lr_mode = 'poly'
    opt.lr = 0.0002
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    elif opt.lr_mode == 'normal':
        lr = opt.lr
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))
    for param_group in optimizer.param_groups:                                
        param_group['lr'] = lr                                            
    return lr

def save_checkpoint(state,epoch,checkpoint_path,stage="val",filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    print("{} Model Saving................".format(epoch))
    shutil.copyfile(filename, osp.join(checkpoint_path,'model_{}_{:03d}.pth.tar'.format(stage,(epoch + 1))))

def setup_seed(seed=1234):
    import random
    import imgaug as ia
    import torch.backends.cudnn as cudnn
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    ia.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
