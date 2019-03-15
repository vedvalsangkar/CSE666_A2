#!/usr/bin/python3.5

import sys
import time

import torch
from torch import cuda

import model as mod
from classes import Helper


# def get_data(mode="both", training_batch_size=256, testing_batch_size=1, shuffle=False):
#     '''
#
#     :param mode: Data load mode. 'train', 'test' or 'both'.
#     :param training_batch_size: Batch size for training.
#     :param testing_batch_size: Batch size for testing.
#     :param shuffle:
#     :return:
#     '''
#
#     split_pre = "IJBA_sets/split"
#     t_pre = "/train_"
#     comp = "/verify_comparisons_"
#     meta = "/verify_metadata_"
#     ext = ".csv"
#     ext2 = "_clean.csv"
#
#     # col_names = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID']
#
#     training_set = {}
#     training_loader = {}
#     testing_set = {}
#     testing_loader = {}
#     # comparison_set = {}
#     # metadata_set = {}
#
#     if mode == 'train' or mode == 'both':
#         for set_n in range(1, 11):
#             training_set[set_n] = A2TrainDataSet(split_pre + str(set_n) + t_pre + str(set_n) + ext2)
#             training_loader[set_n] = data.DataLoader(dataset=training_set[set_n],
#                                                      batch_size=training_batch_size,
#                                                      shuffle=shuffle)
#     if mode == 'test' or mode == 'both':
#         for set_n in range(1, 11):
#             testing_set[set_n] = A2VerifyDataSet(comp_file=split_pre + str(set_n) + comp + str(set_n) + ext,
#                                                  meta_file=split_pre + str(set_n) + meta + str(set_n) + ext)
#
#             testing_loader[set_n] = data.DataLoader(dataset=testing_set[set_n],
#                                                     batch_size=testing_batch_size,
#                                                     shuffle=shuffle)
#
#     # for set_n in range(1, 11):
#     #     training_set[set_n] = A2TrainDataSet(split_pre + str(set_n) + t_pre + str(set_n) + ext2)
#     #     testing_set[set_n] = A2VerifyDataSet(comp_file=split_pre + str(set_n) + comp + str(set_n) + ext,
#     #                                          meta_file=split_pre + str(set_n) + meta + str(set_n) + ext)
#     #
#     #     training_loader[set_n] = data.DataLoader(dataset=training_set[set_n],
#     #                                              batch_size=training_batch_size,
#     #                                              shuffle=shuffle)
#     #
#     #     testing_loader[set_n] = data.DataLoader(dataset=testing_set[set_n],
#     #                                             batch_size=testing_batch_size,
#     #                                             shuffle=shuffle)
#     if mode == 'train':
#         return training_set, training_loader
#     if mode == 'test':
#         return testing_set, testing_loader
#
#     return training_set, training_loader, testing_set, testing_loader


def main(opts):
    """
    Main training function for Assignment 2.

    :param opts: Options for batch size.
    :return: Nothing for now.

    """

    # TODO: Remove when done with tuning.

    # ''' ---------------------Parameters---------------------'''
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    optim_name = 'SGD'
    num_epochs = 50
    batch_printer = 50
    op_dir = "pickles/"
    t_stmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    # helper = Helper()

    if len(opts) > 1:
        batch_size = int(opts[1])
    else:
        batch_size = 10

    helper = Helper("log/log_" + t_stmp + ".txt")
    helper.set_local()

    # logger.log(msg=)

    helper.log(msg="Starting data loading.")

    # training_set, training_loader, testing_set, testing_loader = helper.get_data(mode="train",
    training_set, training_loader = helper.get_data(mode="train",
                                                    training_batch_size=batch_size
                                                    )
    # model, criterion, optimizer = mod.get_model(device, 'SGD')

    helper.log(msg="Finished data loading. Starting main training.")

    # Following code only for creating a dummy file for testing pipeline.
    flag_dummy = False

    # noinspection PyBroadException
    try:
        f = open(op_dir + "dummy.pt", 'r')
        f.close()
    except:
        flag_dummy = True

    # model.train(True)

    for set_n in range(1, 11):

        model, criterion, optimizer = mod.get_model(device, optim_name)
        model.train(True)
        # print(type(optimizer))

        if flag_dummy:
            helper.log(msg="\nCreating dummy file.\n")
            dummy_file = {"model": model.state_dict(),
                          "criterion": criterion.state_dict(),
                          "optimizer": optimizer.state_dict()
                          }
            torch.save(dummy_file, op_dir + "dummy.pt")

        helper.log(msg="\nStart of split {0}\n".format(set_n))

        # https://stackoverflow.com/questions/32558805/ceil-and-floor-equivalent-in-python-3-without-math-module
        total_len = -(-training_set[set_n].__len__() // batch_printer)
        # total_len = training_set[set_n].__len__()
        running_loss = 0.0

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(training_loader[set_n]):

                # Change variable type to match GPU requirements
                inp = images.to(device)
                lab = labels.to(device)

                # a = images.size()

                # Reset gradients before processing
                optimizer.zero_grad()

                # Get model output
                out = model(inp)

                # Calculate loss
                loss = criterion(out, lab)

                # Update weights
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # logger.log(msg="\rLoss = {0}        ".format(l), end="")
                if (i + 1) % batch_printer == 0:
                    # logger.log(msg="Epoch " + str(epoch) + " Step " + str(i + 1) + "/" + str(total_len), end="\t")
                    # logger.log(msg="\rEpoch: {0}, step: {1}/{2}".format(epoch+1, i+1, total_len), end="\t")
                    helper.log(msg="Split: {3}, Epoch: {0}, step: {1}/{2}  ".format(epoch + 1, i + 1, total_len, set_n),
                               end="\t")
                    # logger.log(msg="Running Loss data: ", loss.data)
                    helper.log(msg="Running Loss (avg): {0}".format(running_loss / batch_printer))
                    running_loss = 0.0

        filename = op_dir + "A2_T{1}_S{0}.pt".format(set_n, t_stmp)

        # TODO: Citing source for this idea
        save_file = {"model": model.state_dict(),
                     "criterion": criterion.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "optim_name": optim_name,
                     }

        torch.save(save_file, filename)
        helper.log(msg="\nFile {0} saved for split {1}".format(filename, set_n))

    helper.close()


if __name__ == "__main__":

    # TODO: argparser for commandline options.

    options = sys.argv

    main(options)
