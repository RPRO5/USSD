from models.models import create_model
import torch
import os
from dataprepare.slit_loader import Slit_loader
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import cv2
import util.util as u
from dataprepare.slit_loader import TwoStreamBatchSampler
from collections import OrderedDict
from util.visualizer import Visualizer
import util.util as util
from skimage import data, exposure, img_as_float

def val_Cycle(args, modelG_A, dataloader_val,epoch,k_fold):
    print('\n')
    print('Start Validation......')
    with torch.no_grad():  
        modelG_A.eval()  
        tbar = tqdm.tqdm(dataloader_val, desc='\r')  
        args.net_work = "GAN"
        if not os.path.exists(os.path.join(args.checkpoints_dir,args.net_work,"Gen_B/{}/{}".format(k_fold,epoch))):
            os.makedirs(os.path.join(args.checkpoints_dir,args.net_work,"Gen_B/{}/{}".format(k_fold,epoch)))
        for i, (data, _, _) in enumerate(tbar):  
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
            predict = modelG_A.forward(data,isTrain=False)
            predict = (predict+1)/2
            data_num = data.size(0)
            for idx in range(data_num):
                gen_B = np.array(predict[idx].squeeze(0).cpu().numpy()*255,dtype=np.uint8)
                new_size = (800, 540) 
                gen_B = cv2.resize(gen_B, new_size)
                cv2.imwrite(
                    os.path.join(args.checkpoints_dir,args.net_work,"Gen_B/{}/{}/{}_{}_{}.png".format(k_fold,epoch,epoch,i,idx)),
                    gen_B)

def train(opt, model, dataloader_train_Super,dataloader_train ,dataloader_val, optimizer_G,optimizer_D, k_fold):
    total_steps = 0.0
    opt.num_epochs = 200
    for epoch in range(opt.num_epochs):
        lr_G = u.adjust_learning_rate(opt,optimizer_G,epoch)    # update learning rates in the beginning of every epoch.
        lr_D = u.adjust_learning_rate(opt,optimizer_D,epoch)    # update learning rates in the beginning of every epoch.
        if epoch < 5:
            for i_Super, (data, label,img_list) in enumerate(dataloader_train_Super): # inner loop within one epoch
                model.module.netG.train()
                model.module.netD.train()
                total_steps += opt.batch_size
                opt.use_gpu =True
                if torch.cuda.is_available() and opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda() 
                losses_Pre, generated = model(data,label)   # calculate loss functions, get gradients, update network weights
                losses_Pre = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Pre]
                loss_dict_Pre = dict(zip(model.module.loss_names, losses_Pre))
                # calculate final loss scalar
                loss_D = (loss_dict_Pre['D_fake'] + loss_dict_Pre['D_real']) * 0.5
                loss_G = loss_dict_Pre['G_GAN'] + loss_dict_Pre.get('L1_Loss', 0) + loss_dict_Pre.get('SSIM_Loss', 0)
                errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                ############### Backward Pass ####################
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                model.module.netD.train()
                model.module.netG.eval()
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                save_fake = total_steps % 100 == 0
                if save_fake:
                    visualizer.plot_current_errors(errors_Pre, total_steps)
                    errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                    visuals = OrderedDict([('Pre_Input_Super', util.tensor2im(data[0])),
                                           ('Pre_Denoised_Super', util.tensor2im(generated.data[0])),
                                           ('Pre_Label_Super', util.tensor2im(label[0]))])             
                    visualizer.display_current_results(visuals, epoch, total_steps)
                    ### save latest model
                    model.module.save('Pre_trained', T_S="Pre")
            val_Cycle(args=opt, modelG_A=model.module.netG, dataloader_val=dataloader_val, epoch=epoch, k_fold=k_fold)
            print('saving the Pretrained model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('Pre_trained', T_S="Pre")
            model.module.save(epoch, T_S="Pre")
        else:
            for i, (data, label,img_list_Semi) in enumerate(dataloader_train): # inner loop within one epoch
                model.module.netG.train()
                model.module.netD.train()
                total_steps += opt.batch_size
                if torch.cuda.is_available() and opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()  
                data_Super = data[:opt.labeled_bs]
                label_Super = label[:opt.labeled_bs]
                data_Un = data[opt.labeled_bs:]
                label_Un = label[opt.labeled_bs:]
                losses_Super, generated_Super = model(data_Super,label_Super,Super=True)   # calculate loss functions, get gradients, update network weights
                losses_Super = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Super]
                loss_dict_Super = dict(zip(model.module.loss_names, losses_Super))
                # calculate final loss 
                loss_D_Super = (loss_dict_Super['D_fake'] + loss_dict_Super['D_real']) * 0.5
                loss_G_Super = loss_dict_Super['G_GAN'] + loss_dict_Super.get('L1_Loss', 0) + loss_dict_Super.get('SSIM_Loss', 0)

                ############### Backward Pass ####################
                # update generator weights
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G_Super.backward()
                optimizer_G.step()
                model.module.netD.train()
                model.module.netG.eval()
                optimizer_D.zero_grad()
                loss_D_Super.backward()
                optimizer_D.step()
                model.module.netG.train()
                model.module.netD.train()
                losses_Un , generated_Un = model(data_Un,label_Un,Super=False)   
                losses_Un = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Un]
                loss_dict_Un = dict(zip(model.module.loss_names, losses_Un))
                # calculate final loss 
                loss_G_Un = loss_dict_Un['G_GAN']
                loss_D_Un = loss_dict_Un['D_fake']

                ############### Backward Pass ####################
                # update generator weights
                model.module.netG.train()
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G_Un.backward()
                optimizer_G.step()
                model.module.netG.eval()
                model.module.netD.train()
                optimizer_D.zero_grad()
                loss_D_Un.backward()
                optimizer_D.step()
                save_fake = total_steps % 100 == 0
                if save_fake:
                    errors_Super = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Super.items()}
                    errors_un = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Un.items()}
                    visuals = OrderedDict([('Smi_Input_Super', util.tensor2im(data_Super[0])),
                                           ('Smi_Denoised_Super', util.tensor2im(generated_Super.data[0])),
                                           ('Smi_Label_Super', util.tensor2im(label_Super[0])),
                                           ('Smi_Input_Un', util.tensor2im(data_Un[0])),
                                           ('Smi_Denoised_Un', util.tensor2im(generated_Un.data[0])),
                                           ])
                    visualizer.display_current_results(visuals, epoch, total_steps)
                    ### save latest model
                    model.module.save('Latest_Semi', T_S="Semi")
            for i_Super, (data, label,img_list) in enumerate(dataloader_train_Super): # inner loop within one epoch
                model.module.netG.train()
                model.module.netD.train()
                total_steps += opt.batch_size
                if torch.cuda.is_available() and opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()  
                losses_Pre, generated = model(data,label)   # calculate loss functions
                losses_Pre = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Pre]
                loss_dict_Pre = dict(zip(model.module.loss_names, losses_Pre))
                # calculate final loss 
                loss_D = (loss_dict_Pre['D_fake'] + loss_dict_Pre['D_real']) * 0.5
                loss_G = loss_dict_Pre['G_GAN'] + loss_dict_Pre.get('L1_Loss', 0) + loss_dict_Pre.get('SSIM_Loss', 0)
                errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                ############### Backward Pass ####################
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                model.module.netD.train()
                model.module.netG.eval()
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                save_fake = total_steps % 100 == 0
                if save_fake:
                    visualizer.plot_current_errors(errors_Pre, total_steps)
                    errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                    visuals = OrderedDict([('Fine_Input_Super', util.tensor2im(data[0])),
                                           ('Fine_Denoised_Super', util.tensor2im(generated.data[0])),
                                           ('Finee_Label_Super', util.tensor2im(label[0]))])
                    visualizer.display_current_results(visuals, epoch, total_steps)
                    ### save latest model
                    model.module.save('Fine_trained', T_S="Fine")

            val_Cycle(args=opt, modelG_A=model.module.netG, dataloader_val=dataloader_val, epoch=epoch, k_fold=k_fold)
            print('saving the Fine model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('Fine_trained', T_S="Fine")
            model.module.save(epoch, T_S="Fine")

def patients_to_slices(k_fold=1,total_folds=['f1','f2','f3']):
    total_folds.remove("f{}".format(k_fold))
    label_num = 0
    ref_dict = {"f1": 256, "f2": 256,
                    "f3": 630,"f4": 1130}
    for fold in total_folds:
        label_num += ref_dict[fold]
    return label_num

def main(opt,k_fold):
    ################create dataset
    opt.data = r"C:\Users\abc\Desktop\BF-C2Gan\Datasets"
    opt.dataset = 'Data'
    opt.crop_height = 128
    opt.crop_width = 128
    opt.batch_size = 4
    opt.labeled_bs = 1
    opt.display_id = -1
    dataset_path = os.path.join(opt.data, opt.dataset)
    dataset_train = Slit_loader(dataset_path, scale=(opt.crop_height, opt.crop_width), k_fold_test=k_fold,
                               mode='train')
    print(len(dataset_train))
    total_slices = len(dataset_train)
    labeled_slice = patients_to_slices(k_fold, total_folds=['f1','f2','f3'])
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, opt.batch_size, opt.batch_size-opt.labeled_bs)

    dataloader_train = DataLoader(
        dataset_train,  
        batch_sampler=batch_sampler,
        num_workers=0,  
        pin_memory=True
    )
    dataset_val = Slit_loader(dataset_path, scale=(opt.crop_height, opt.crop_width), k_fold_test=k_fold, mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=opt.labeled_bs,  
        shuffle=False,
        num_workers=0, 
        pin_memory=True,
        drop_last=False
    )
    opt.dataset_Super = 'Data_Super'
    dataset_path_Super = os.path.join(opt.data, opt.dataset_Super)
    dataset_train_Super = Slit_loader(dataset_path_Super, scale=(opt.crop_height, opt.crop_width), k_fold_test=k_fold,
                                mode='train')
    print("Labeled slices is: {}".format(
        len(dataset_train_Super)))

    dataloader_train_Super = DataLoader(
        dataset_train_Super,
        batch_size=opt.labeled_bs,
        shuffle=True,
        num_workers=0, 
        pin_memory=True,  
        drop_last=True
    )
    print("Labeled slices Step: {}".format(
        len(dataloader_train_Super)))
    opt.isTrain = True
    model = create_model(opt)
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    train(opt, model, dataloader_train_Super, dataloader_train, dataloader_val,  optimizer_G,optimizer_D, k_fold=k_fold)

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.data = None
        self.batch_size = None
        self.isTrain = True
        self.dataset_Super = 'Data'
        self.gpu_ids = list(range(torch.cuda.device_count()))
        self.checkpoints_dir = "./checkpoints"
        self.no_lsgan = False
        self.tf_log = True
        self.net_work = None

if __name__ == '__main__':
    from datetime import datetime
    import socket
    from tensorboardX import SummaryWriter
    opt = TrainOptions()
    visualizer = Visualizer(opt)
    seed = 42
    u.setup_seed(seed)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')  
    log_dir = os.path.join(r"C:\Users\abc\Desktop\BF-C2Gan\logs",
                           'example_network' + '_' + '_' + current_time + '_' + socket.gethostname())  
    if not os.path.exists(os.path.dirname(log_dir)):  
        os.makedirs(os.path.dirname(log_dir))
    writer = SummaryWriter(log_dir=log_dir)  
    main(opt, k_fold=int(3))
