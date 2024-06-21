import numpy as np
import torch

from nn_denoiser.oidn.config import *
from nn_denoiser.oidn.util import *
from nn_denoiser.oidn.model import *
from nn_denoiser.oidn.color import *
from nn_denoiser.oidn.result import *

# Inference function object
class OIDN(object):
  def __init__(self):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] %s is used for denoising!' % device)

    # Initialize the model
    # - U-Net, 9 channels for network inputs
    self.model = get_model('unet', 9)
    self.model.to(device)

    self.ckpt_dir = './nn_denoiser/oidn/weights'
    checkpoint = load_checkpoint_for_ours(self.ckpt_dir, device, self.model)

    # Initialize the transfer function
    self.transfer = LogTransferFunction()

    # Set the model to evaluation mode
    self.model.eval()


  # Inference function
  def __call__(self, cNoisy, cAlbedo, cNormal):
    cNoisy = self.transfer.forward(cNoisy)

    # Filter the main feature
    netInputs = torch.concat((cNoisy, cAlbedo, cNormal), dim=1)
    netOutput = self.model(netInputs)

    # Apply the inverse transfer function
    netOutput = self.transfer.inverse(netOutput)

    return netOutput