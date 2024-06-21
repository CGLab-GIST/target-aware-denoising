import os
import numpy as np
import torch
from collections import OrderedDict

from config import *
from util import *
from result import *
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Imports TZA from PyTorch checkpoint.')

  device = init_device(cfg)

  input_filename = os.path.join(get_result_dir(cfg), cfg.result + '.tza')
  output_filename = os.path.join(get_result_dir(cfg), 'checkpoint.pth')

  input_model_state = OrderedDict()
  input_file = tza.Reader(input_filename)
  for eleName in input_file._table:
    currTensor, currLayout = input_file[eleName]
    currTensorTorch = torch.from_numpy(currTensor.copy()).to(device)
    input_model_state[eleName] = currTensorTorch

  input_checkpoint = {}
  input_checkpoint['model_state'] = input_model_state
  torch.save(input_checkpoint, output_filename)

  print('Done!')


if __name__ == '__main__':
  main()