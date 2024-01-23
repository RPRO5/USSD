import torch
from torchstat import stat
def create_model(opt):
    from .BF-C2GAN import BF-C2GAN, InferenceModel
    if opt.isTrain:
        model = BF-C2GAN()
    else:
        model = InferenceModel()
    model.initialize(opt)
    opt.gpu_ids = list(range(torch.cuda.device_count()))
    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    return model
