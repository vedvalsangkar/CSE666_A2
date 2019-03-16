#!/usr/bin/python3.5

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
# https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
#
#

import torch
import numpy as np

from torch import cuda, nn
from PIL import Image
from torchvision import transforms

import model as mod

from classes import Helper


def get_list(model):
    return list(model.children())


def list_model():
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    helper = Helper()

    threshold = 0.75

    # load = torch.load("pickles/A2_T20190315_104245_S1.pt")
    load = torch.load("pickles/dummy.pt")

    print("Class: ", load["optimizer"].__class__())
    print("Type: ", type(load["optimizer"]))
    # print("obj: ", load["optimizer"])
    print("obj: ", load["optim_name"])

    loaded_model, _, _ = mod.get_model(device, load["optim_name"])

    loaded_model.load_state_dict(load["model"])

    loaded_model.eval()

    child_list = get_list(loaded_model)

    # print("=========")
    # print("Children:")
    # print("=========")
    # for c in child_list:
    #     print(c)
    #     print("-----")
    #
    # print("=========")

    # chopped_m = nn.Sequential(*child_list[:-1])

    # training_set, training_loader = helper.get_data(mode="train",
    #                                                 training_batch_size=10
    #                                                 )
    testing_set, testing_loader = helper.get_data(mode="test",
                                                  testing_batch_size=1
                                                  )

    correct = 0
    total = 0
    false_reject = 0
    false_accept = 0

    print("Starting eval.")

    # with torch.no_grad():
    #     for images, labels in training_loader[1]:
    #         inp = images.to(device)
    #         lab = labels.to(device)
    #         outputs = loaded_model(inp)
    #         a, predicted = torch.max(outputs.data, 1)
    #         # print(outputs)
    #         # print(outputs[0])
    #         # print(a)
    #         # print(predicted)
    #         # print(lab)
    #         # print("------")
    #         total += lab.size(0)
    #         correct += (predicted == lab).sum().item()

    to_tensor = transforms.ToTensor()

    total_len = len(testing_loader[1])

    with torch.no_grad():
        for i, (list_1, list_2, labels) in enumerate(testing_loader[1]):
            if type(list_1).__name__ != 'list' or type(list_2).__name__ != 'list':
                print("Issues with testing file at loaction {0}".format(i))
                print(list_1)
                print(list_2)
            # print(list_1)
            # print(list_2)
            # print(labels)
            # print(type(labels))
            l1_avg = np.zeros([1, loaded_model.features])
            l2_avg = np.zeros([1, loaded_model.features])
            for im in list_1:
                # print(im[0])
                image = Image.open(im[0])
                # print(image)
                tensor_img = to_tensor(image).to(device)
                output = loaded_model(tensor_img.unsqueeze(0))
                l1_avg += output.cpu().numpy()
            l1_avg /= len(list_1)
            for im in list_2:
                image = Image.open(im[0])
                tensor_img = to_tensor(image).to(device)
                output = loaded_model(tensor_img.unsqueeze(0))
                l2_avg += output.cpu().numpy()
            l2_avg /= len(list_2)

            try:
                dot = np.sum(l1_avg * l2_avg)
                a = np.sqrt(np.sum(l1_avg * l1_avg))
                b = np.sqrt(np.sum(l2_avg * l2_avg))
                # b = np.linalg.norm(l2_avg)
                cos_sim = dot / (a * b)
            except ValueError:
                print(list_1)
                print(list_2)
                continue
            # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists

            total += 1
            correct += int(cos_sim > threshold and labels.item())
            false_reject += int(cos_sim <= threshold and labels.item())
            false_accept += int(cos_sim > threshold and not labels.item())

            if (i+1) % 100 == 0:
                print("Step {0}/{1}. C|FA|FR:{2}|{3}|{4}<T>:{5}".format(i, total_len, correct, false_accept, false_reject, total))

    print("Accuracy metrics:\nTotal{0}".format(total))
    print("Correct      : {0} \t|\t {1} %".format(correct, (correct / total)))
    print("False Accept : {0} \t|\t {1} %".format(false_accept, (false_accept / total)))
    print("False Reject : {0} \t|\t {1} %".format(false_reject, (false_reject / total)))


list_model()
