import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
vae123_layers = ['relu1_1', 'relu2_1', 'relu3_1']
vae345_layers = ['relu3_1', 'relu4_1', 'relu5_1']

class _VGG(nn.Module):
    def __init__(self, model):
        super(_VGG, self).__init__()

        # Load pretrained model
        features = models.vgg19(pretrained=True).features

        # Rename layers
        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

        # Disable autograd
        for param in self.features.parameters():
            param.requires_grad = False

        # Content layers
        if model == "vae-123":
            self.content_layers = vae123_layers
        elif model == "vae-345":
            self.content_layers = vae345_layers
        

    def forward(self, inputs):
        batch_size = inputs.size(0)
        all_outputs = []
        output = inputs
        for name, module in self.features.named_children():
            output = module(output)
            if name in self.content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs

class KLDLoss(nn.Module):
    def __init__(self, device, size_average=True):
        super(KLDLoss, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=size_average)
        self.device = device

    def forward(self, latent_z):
        # Shape
        batch_size, nz = latent_z.size()
        # Desired distribution
        latent_label = torch.FloatTensor(batch_size, nz).fill_(1).to(self.device)
        # KLD loss
        kld_loss = self.criterion(F.log_softmax(latent_z), latent_label)

        return kld_loss

class FLPLoss(nn.Module):
    def __init__(self, model, device, size_average=True):
        super(FLPLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=size_average)
        self.pretrained = _VGG(model).to(device)
    
    def forward(self, x, recon_x):
        x_f = self.pretrained(x)
        recon_f = self.pretrained(recon_x)
        return self._fpl(recon_f, x_f)

    def _fpl(self, recon_f, x_f):
        fpl = 0
        for _r, _x in zip(recon_f, x_f):
            fpl += self.criterion(_r, _x)
        return fpl