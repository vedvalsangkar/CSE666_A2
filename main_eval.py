#!/usr/bin/python3.5

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
# https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
#
#

import torch

from torch import cuda  # , nn

import model as mod


def get_list(model):
    return list(model.children())


def list_model():
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    load = torch.load("pickles/dummy.pt")

    print("Class: ", load["optimizer"].__class__())
    print("Type: ", type(load["optimizer"]))
    print("obj: ", load["optimizer"])

    loaded_model, _, _ = mod.get_model(device)

    loaded_model.load_state_dict()

    child_list = get_list(loaded_model)

    print("=========")
    print("Children:")
    print("=========")
    for c in child_list:
        print(c)
        print("-----")

    print("=========")


# chopped_m = nn.Sequential(*child_list)
list_model()
