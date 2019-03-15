#!/usr/bin/python3.5

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
# https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
#
#

import torch

# from torch import nn

import model as mod


loaded_model = mod.ResNet()
print("Children:")
loaded_model.load_state_dict(torch.load("pickles/dummy2.pt"))

child_list = list(loaded_model.children())

print("=========")
print("Children:")
print("=========")
for c in child_list:
    print(c)
    print("-----")

print("=========")

# chopped_m = nn.Sequential(*child_list)
