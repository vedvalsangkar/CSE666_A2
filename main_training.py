#!/usr/bin/python3.5

import time
import argparse as ap
# from matplotlib import pyplot as plt

import torch
from torch import cuda

import model as mod
from classes import Helper


def main(batch_size, num_epochs, lr, file_write, flag_dummy, temperature, lr_decay, features):
    """
    Main function for training.
    :param batch_size  : Batch size to use.
    :param num_epochs  : Number of epochs for each split.
    :param lr          : Learning rate to be set at the start of each split.
    :param file_write  : Write output to stdout(default) or a file.
    :param flag_dummy  : Create a dummy file for evaluation.
    :param temperature : Set default temperature for softmax/log_softmax layer while training.
    :param lr_decay    : Learning rate decay for every drop in min loss observed.
    :param features    : Number of nodes for penultimate feature layer.
    :return:
    """

    # ''' ---------------------Parameters---------------------'''
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    # optim_name = 'SGD'
    optim_name = 'Adam'
    # optim_name = 'RMS'
    batch_printer = 50
    op_dir = "pickles/"
    t_stmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    # Creating a file to store the name of the latest models
    ff = open("latest_t_stmp.txt", 'w')
    ff.write("A2_T{0}_S".format(t_stmp))
    ff.close()

    helper = Helper("log/log_" + t_stmp + ".txt")
    helper.write_file(file_write)

    helper.log(msg="Starting data loading.")

    # training_set, training_loader, testing_set, testing_loader = helper.get_data(mode="train",
    training_set, training_loader = helper.get_data(mode="train",
                                                    training_batch_size=batch_size
                                                    )
    helper.log(msg="Finished data loading. Starting main training.")

    # Following code only for creating a dummy file for testing pipeline.
    # flag_dummy = False
    # noinspection PyBroadException
    # try:
    #     f = open(op_dir + "dummy.pt", 'r')
    #     f.close()
    # except:
    #     flag_dummy = True

    for set_n in range(1, 11):

        init_lr = lr

        model, criterion, optimizer = mod.get_model(device, optim_name, lamb=0, learning_rate=init_lr, final_features=features)
        model.train(True)
        model.set_temperature(temperature)
        # print(type(optimizer))

        if flag_dummy:
            helper.log(msg="\nCreating dummy file.\n")
            dummy_file = {"model": model.state_dict(),
                          "criterion": criterion.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "optim_name": optim_name
                          }
            torch.save(dummy_file, op_dir + "dummy.pt")
            flag_dummy = False

        helper.log(msg="\nStart of split {0}\n".format(set_n))

        # https://stackoverflow.com/questions/32558805/ceil-and-floor-equivalent-in-python-3-without-math-module
        # total_len = -(-training_set[set_n].__len__() // batch_size)
        total_len = len(training_loader[set_n])

        # total_len = training_set[set_n].__len__()
        running_loss = 0.0
        cor = 0
        tot = 0
        cor_b = 0
        tot_b = 0
        past_loss = 6.0 * batch_printer

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
                # pre = out
                # pre = pre.cpu().detach().numpy()

                # Calculate loss
                loss = criterion(out, lab)

                # Accuracy calc
                _, predicted = torch.max(out.data, 1)

                tot_b += batch_size
                cor_b += (predicted == lab).sum().item()

                tot += batch_size
                cor += (predicted == lab).sum().item()

                # Update weights
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # logger.log(msg="\rLoss = {0}        ".format(l), end="")
                if (i + 1) % batch_printer == 0 or (i + 1) == total_len:

                    # plt.imsave("img/im_{0}_{1}_{2}.jpg".format(set_n, epoch, i+1),
                    #            images[0].numpy().transpose(1, 2, 0))

                    # print(predicted, lab)
                    # logger.log(msg="Epoch " + str(epoch) + " Step " + str(i + 1) + "/" + str(total_len), end="\t")
                    # logger.log(msg="\rEpoch: {0}, step: {1}/{2}".format(epoch+1, i+1, total_len), end="\t")
                    helper.log(msg="Split: {3}, Epoch: {0}, step: {1}/{2} ".format(epoch + 1, i + 1, total_len, set_n),
                               end="\t")
                    # logger.log(msg="Running Loss data: ", loss.data)
                    helper.log(msg="Running Loss (avg): {0:.06f}, Past: {1:.06f}".format((running_loss / batch_printer),
                                                                                         (past_loss / batch_printer)),
                               end="\t")
                    helper.log(msg="Accuracy (Per {2})|(Total): {0:.03f}|{1:.03f} %".format((cor_b * 100) / tot_b,
                                                                                            (cor * 100) / tot,
                                                                                            batch_size * batch_printer),
                               end="\t")

                    if running_loss < past_loss:
                        past_loss = running_loss
                        init_lr *= lr_decay
                        for params in optimizer.param_groups:
                            params['lr'] = max(init_lr, 0.001)

                    helper.log(msg="LR: {0:.06f}".format(init_lr))

                    running_loss = 0.0
                    cor_b = 0
                    tot_b = 0
                    # optimizer.zero_grad()

        filename = op_dir + "A2_T{1}_S{0}.pt".format(set_n, t_stmp)

        # Idea for named saved file was picked up from here:
        # https://github.com/quiltdata/pytorch-examples/blob/master/imagenet/main.py
        save_file = {"model": model.state_dict(),
                     "criterion": criterion.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "optim_name": optim_name
                     }

        torch.save(save_file, filename)
        helper.log(msg="\nFile {0} saved for split {1}".format(filename, set_n))

    helper.close()


if __name__ == "__main__":

    # Idea for argparser was picked up from this site:
    # https://github.com/floydhub/imagenet/blob/master/main.py
    # Options were modified for current code to run on GCP.

    parser = ap.ArgumentParser()

    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=2,
                        help="Batch Size (default 2).")

    parser.add_argument("-e",
                        "--num-epochs",
                        type=int,
                        default=3,
                        help="Number of epochs (default 3).")

    parser.add_argument("--lr",
                        "--learning-rate",
                        type=float,
                        default=0.01,
                        help="Initial learning rate (default 0.01)")

    parser.add_argument("-t",
                        "--temperature",
                        type=float,
                        default=1.0,
                        help="Initial learning rate (default 1.0)")

    parser.add_argument("-d",
                        "--lr-decay",
                        type=float,
                        default=0.95,
                        help="Learning rate decay (default 0.95)")

    parser.add_argument("-f",
                        "--features",
                        type=int,
                        default=1024,
                        help="Number of features for penultimate layer")

    parser.add_argument("-w",
                        "--file-write",
                        action="store_true",
                        default=False,
                        help="Write to file instead of stdout.")

    parser.add_argument("-c",
                        "--create-dummy",
                        action="store_true",
                        default=False,
                        help="Write a dummy model file for evaluation.")

    args = parser.parse_args()

    main(args.batch_size, args.num_epochs, args.lr, args.file_write,
         args.create_dummy, args.temperature, args.lr_decay, args.features)
