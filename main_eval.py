#!/usr/bin/python3.5

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
# https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
#

import torch
import numpy as np
import pandas as pd
import argparse as ap

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
# from itertools import cycle

from torch import cuda, nn
from torchvision import transforms

import model as mod

from classes import Helper


def get_list(input_model):
    return list(input_model.children())


def load_model(filename, device):
    # device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    # load = torch.load("pickles/A2_T20190317_104005_S1.pt")
    load = torch.load("pickles/"+filename)

    '''
    Best models on GCP:

    A2_T20190317_104005_S1.pt

    A2_T20190317_082812_S4.pt
    A2_T20190317_082812_S2.pt
    '''

    loaded_model = mod.get_model_only(device, final_features=load["features"])

    loaded_model.load_state_dict(load["model"])

    return loaded_model


def evaluate(filename):
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    helper = Helper()

    testing_set, testing_loader = helper.get_data(mode="test", testing_batch_size=1)

    # correct = 0
    # total = 0
    # false_reject = 0
    # false_accept = 0

    print("Starting evaluation")

    to_tensor = transforms.ToTensor()

    true_val = {}
    score = {}
    tpr = {}
    fpr = {}
    thresh = {}
    area = {}

    for set_n in range(1, 11):

        true_val[set_n] = []
        score[set_n] = []

        model = load_model(filename + str(set_n) + ".pt", device)

        model.eval()

        total_len = len(testing_loader[set_n])

        with torch.no_grad():
            for i, (list_1, list_2, labels) in enumerate(testing_loader[set_n]):

                if type(list_1).__name__ != 'list' or type(list_2).__name__ != 'list':
                    print("Issues with testing file at location {0}".format(i))
                    print(list_1)
                    print(list_2)
                    continue

                l1_avg = np.zeros([1, model.features])
                l1 = 0

                l2_avg = np.zeros([1, model.features])
                l2 = 0

                for im in list_1:
                    try:
                        image = Image.open(im[0])
                        tensor_img = to_tensor(image).to(device)
                        output = model(tensor_img.unsqueeze(0))
                        l1_avg += output.cpu().numpy()
                        l1 += 1
                    except FileNotFoundError:
                        print("File {0} not found. Skipping.".format(im))
                l1_avg /= l1

                for im in list_2:
                    try:
                        image = Image.open(im[0])
                        tensor_img = to_tensor(image).to(device)
                        output = model(tensor_img.unsqueeze(0))
                        l2_avg += output.cpu().numpy()
                        l2 += 1
                    except FileNotFoundError:
                        print("File {0} not found. Skipping.".format(im))
                l2_avg /= l2

                s = cosine_similarity(l1_avg.reshape(1, -1), l2_avg.reshape(1, -1))[0, 0]
                score[set_n].append(s)
                true_val[set_n].append(labels.item())
                if (i + 1) % 500 == 0:
                    print("Step: {0}/{1}".format(i, total_len))
                #     print(score[1][i], true_val[1][i])

            # Code to evaluate ROC graph is taken from the official documentation.
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

            fpr[set_n], tpr[set_n], thresh[set_n] = roc_curve(np.asarray(true_val[set_n]), np.asarray(score[set_n]))
            area[set_n] = auc(fpr[set_n], tpr[set_n])

            plt.figure()
            plt.plot(fpr[set_n],
                     tpr[set_n],
                     color='darkorange',
                     lw=2,
                     label="ROC curve (area = {0:.2f})".format(area[set_n]))

            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for Split {0}'.format(set_n))
            plt.legend(loc="lower right")
            plt.savefig(fname="images/{1}ROC{0}.jpg".format(set_n, filename[:-1]))

    color_list = ["aqua", "chocolate", "brown", "navy", "lime", "olive", "silver", "gold", "pink", "magenta"]

    plt.figure()
    print("Thresholds aquired:")
    for set_n in range(1, 11):
        print("Split {0}".format(set_n), thresh[set_n])
        plt.plot(fpr[set_n],
                 tpr[set_n],
                 color=color_list[set_n - 1],
                 lw=1,
                 label="Split {0}".format(set_n))

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Consolidated)}')
    plt.legend(loc="lower right")
    plt.savefig(fname="images/{0}ROC_ALL.jpg".format(filename[:-1]))

    pd.DataFrame(true_val).to_csv(path_or_buf="results/{0}GT.csv".format(filename[:-1]))
    pd.DataFrame(score).to_csv(path_or_buf="results/{0}Score.csv".format(filename[:-1]))
    pd.DataFrame(tpr).to_csv(path_or_buf="results/{0}TPR.csv".format(filename[:-1]))
    pd.DataFrame(fpr).to_csv(path_or_buf="results/{0}FPR.csv".format(filename[:-1]))
    pd.DataFrame(thresh).to_csv(path_or_buf="results/{0}TH.csv".format(filename[:-1]))
    pd.DataFrame(area).to_csv(path_or_buf="results/{0}Area.csv".format(filename[:-1]))

    print("\n\nDone")


def single_eval(filename, split):

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    helper = Helper()

    testing_set, testing_loader = helper.get_data(mode="test", testing_batch_size=1)

    print("Starting single file evaluation")

    to_tensor = transforms.ToTensor()

    true_val = {}
    score = {}
    tpr = {}
    fpr = {}
    thresh = {}
    area = {}

    set_n = split

    true_val[set_n] = []
    score[set_n] = []

    model = load_model(filename + str(set_n) + ".pt", device)

    model.eval()

    total_len = len(testing_loader[set_n])

    with torch.no_grad():
        for i, (list_1, list_2, labels) in enumerate(testing_loader[set_n]):
            try:
                if type(list_1).__name__ != 'list' or type(list_2).__name__ != 'list':
                    print("Issues with testing file at location {0}".format(i))
                    print(list_1)
                    print(list_2)
                    continue

                l1_avg = np.zeros([1, model.features])
                l1 = 0

                l2_avg = np.zeros([1, model.features])
                l2 = 0

                for im in list_1:
                    try:
                        image = Image.open(im[0])
                        tensor_img = to_tensor(image).to(device)
                        output = model(tensor_img.unsqueeze(0))
                        l1_avg += output.cpu().numpy()
                        l1 += 1
                    except FileNotFoundError:
                        print("File {0} not found. Skipping.".format(im))
                l1_avg /= l1

                for im in list_2:
                    try:
                        image = Image.open(im[0])
                        tensor_img = to_tensor(image).to(device)
                        output = model(tensor_img.unsqueeze(0))
                        l2_avg += output.cpu().numpy()
                        l2 += 1
                    except FileNotFoundError:
                        print("File {0} not found. Skipping.".format(im))
                l2_avg /= l2

                s = cosine_similarity(l1_avg.reshape(1, -1), l2_avg.reshape(1, -1))[0, 0]
                score[set_n].append(s)
                true_val[set_n].append(labels.item())
                if (i + 1) % 500 == 0:
                    print("Step: {0}/{1}".format(i, total_len))

            except ValueError:
                print("Issues with testing file at location {0}".format(i))
                print(list_1)
                print(list_2)
            #     print(score[1][i], true_val[1][i])

        # Code to evaluate ROC graph is taken from the official documentation.
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

        fpr[set_n], tpr[set_n], thresh[set_n] = roc_curve(np.asarray(true_val[set_n]), np.asarray(score[set_n]))
        area[set_n] = auc(fpr[set_n], tpr[set_n])

        plt.figure()
        plt.plot(fpr[set_n],
                 tpr[set_n],
                 color='darkorange',
                 lw=2,
                 label="ROC curve (area = {0:.2f})".format(area[set_n]))

        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Split {0}'.format(set_n))
        plt.legend(loc="lower right")
        plt.savefig(fname="images/{1}ROC{0}.jpg".format(set_n, filename[:-1]))

        pd.DataFrame(true_val).to_csv(path_or_buf="results/{0}GT_{1}.csv".format(filename[:-1], set_n))
        pd.DataFrame(score).to_csv(path_or_buf="results/{0}Score_{1}.csv".format(filename[:-1], set_n))
        pd.DataFrame(tpr).to_csv(path_or_buf="results/{0}TPR_{1}.csv".format(filename[:-1], set_n))
        pd.DataFrame(fpr).to_csv(path_or_buf="results/{0}FPR_{1}.csv".format(filename[:-1], set_n))
        pd.DataFrame(thresh).to_csv(path_or_buf="results/{0}TH_{1}.csv".format(filename[:-1], set_n))
        pd.DataFrame(area).to_csv(path_or_buf="results/{0}Area_{1}.csv".format(filename[:-1], set_n))


if __name__ == "__main__":

    parser = ap.ArgumentParser()

    parser.add_argument("-f",
                        "--file",
                        type=str,
                        # default="",
                        help="File to process. If None, all will be processed.")

    parser.add_argument("-s",
                        "--split",
                        type=int,
                        default=2,
                        help="Split number to process.")

    args = parser.parse_args()

    if args.file is None:
        f = open("latest_t_stmp.txt", 'r')
        latest_file = f.readline()
        f.close()

        print(latest_file)

        evaluate(latest_file)

    else:

        single_eval(args.file, args.split)
