import torch
from torchstat import stat
def create_model(opt):
    from .Caps_cGAN import Caps_cGAN, InferenceModel
    if opt.isTrain:
        model = Caps_cGAN()
    else:
        model = InferenceModel()
    model.initialize(opt)

    opt.gpu_ids = list(range(torch.cuda.device_count()))
    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    return model
