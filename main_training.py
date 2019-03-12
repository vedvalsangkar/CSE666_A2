# import pandas as pd
import torch

from torch import cuda
from torch.utils import data

import time

import model as mod
from classes import A2TrainDataSet, A2VerifyDataSet


def get_data(training_batch_size=256, testing_batch_size=1, shuffle=False):

    split_pre = "IJBA_sets/split"
    t_pre = "/train_"
    comp = "/verify_comparisons_"
    meta = "/verify_metadata_"
    ext = ".csv"
    ext2 = "_clean.csv"

    # col_names = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID']

    training_set = {}
    training_loader = {}
    testing_set = {}
    testing_loader = {}
    # comparison_set = {}
    # metadata_set = {}

    for set_n in range(1, 11):

        training_set[set_n] = A2TrainDataSet(split_pre + str(set_n) + t_pre + str(set_n) + ext2)
        testing_set[set_n] = A2VerifyDataSet(comp_file=split_pre + str(set_n) + comp + str(set_n) + ext,
                                             meta_file=split_pre + str(set_n) + meta + str(set_n) + ext)

        training_loader[set_n] = data.DataLoader(dataset=training_set[set_n],
                                                 batch_size=training_batch_size,
                                                 shuffle=shuffle)

        testing_loader[set_n] = data.DataLoader(dataset=testing_set[set_n],
                                                batch_size=testing_batch_size,
                                                shuffle=shuffle)

    return training_set, training_loader, testing_set, testing_loader


def main():

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    num_epochs = 25
    batch_size = 50

    op_dir = "pickles/"
    t_stmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    print("Starting data loading.")

    training_set, training_loader, testing_set, testing_loader = get_data(training_batch_size=batch_size,
                                                                          testing_batch_size=1)
    model, criterion, optimizer = mod.get_model(device)

    print("Finished data loading. Starting main training.")

    print(training_set[1].__getitem__(1))

    print("Finished data loading. Starting main training.")

    for set_n in range(1, 11):

        print("\nStart of split {0}\n".format(set_n))

        total_len = training_set[set_n].__len__()
        running_loss = 0.0

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(training_loader[set_n]):
            # i = 0
            # for images, labels in training_loader:
                # Change variable type to match GPU requirements
                inp = images.to(device)
                lab = labels.to(device)

                # Reset gradients before processing
                optimizer.zero_grad()

                # Get model output
                out = model(inp)

                # Calculate loss
                loss = criterion(out, lab)

                # Update weights
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0] if len(loss.data) > 1 else loss.data

                if i+1 % 10 == 0:
                    # print("Epoch " + str(epoch) + " Step " + str(i + 1) + "/" + str(total_len), end="\t")
                    print("Epoch: {0}, step: {1}/{2}".format(epoch+1, i+1, total_len), end="\t")
                    # print("Running Loss data: ", loss.data)
                    print("Running Loss (avg): ", running_loss/10)
                    running_loss = 0.0

                # i += 1

        filename = op_dir + "A2_S{0}_{1}.ckpt".format(set_n, t_stmp)
        torch.save(model.state_dict(), filename)
        print("\nFile {0} saved for split {1}".format(filename, set_n))


if __name__ == "__main__":

    main()

    # for s in ts:
    #     data = ts[s]
    #     l = data.__len__()
    #     print("Split {}.".format(s))
    #     # for i in range(l):
    #     #     _, _ = data.__getitem__(i)
    #     data.clean(s)
    #
    # cnt = 0
    #
    # tdata = testing[1]
    #
    # l = tdata.__len__()
    # for i in range(l):
    #     cnt += 1
    #     if cnt%100 == 0:
    #         t1, t2, cond = tdata.__getitem__(i)
    #         print("List for template 1: {0} having length: {1}".format(tdata.template_1[i], len(t1)))
    #         print(t1)
    #         print("List for template 2: {0} having length: {1}".format(tdata.template_2[i], len(t2)))
    #         print(t2)
    #         print("Label =", cond, "\n")



