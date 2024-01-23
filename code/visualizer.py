import numpy as np
import os
import ntpath
import torch
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  
except ImportError:
    from io import BytesIO         
import tensorflow as tf

class Visualizer():
    def __init__(self, opt):
        opt.tf_log = True
        opt.isTrain = True
        opt.no_html = False
        opt.display_winsize = 64
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        opt.checkpoints_dir = "./checkpoints"
        opt.name = 'model1'
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.create_file_writer(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                from PIL import Image
                image_pil = Image.fromarray(image_numpy)
                new_size = (640, 480)  
                image_pil = image_pil.resize(new_size)
                image_pil.save(s, format="png")
                img_sum = tf.summary.image(name='image_summary', data=image_numpy, step=step)
                img_summaries.append(tf.summary.image(name=label, data=img_sum))
        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.png' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                        ax1.imshow(im_convert(image_numpy))     
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)
            self.name = "Modell"
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.png' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()
        
    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            import tensorflow as tf
            from tensorflow import summary
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            for tag, value in errors.items():
                summary = self.tf.summary.scalar(name=tag, data=value)
                self.summary_writer.flush()

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

